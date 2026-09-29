"""Retrain a challenger on the most recent LABELLED stream data.

Closed-loop step between the drift trigger and the promotion gate. The
challenger's training set grows with released labels while the leakage rules
stay intact — all boundaries are chronological, in TransactionDT:

    original train split ── < ── extra labelled stream ── < ──
    retrain validation slice ── < ── gate evaluation window

The retrain validation slice (the last fully-labelled week before the gate
window) is what the cost-optimal threshold is tuned on, so the challenger's
threshold reflects the drifted distribution rather than the six-month-old
original validation split. Active drift injections are re-applied to the
stream rows (the model must adapt to what serving actually sees); labels and
cost amounts come from the ``labels`` table, which is staged pre-injection.

Registration is NOT done here — the DAG registers the run and sets the
``challenger`` alias, mirroring ``fraudops.models.train``.
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")  # keep run logs clean

import mlflow  # noqa: E402

from fraudops.data.clock import sim_day
from fraudops.data.loader import load_joined
from fraudops.data.splits import TRAIN, chronological_split
from fraudops.features.pipeline import FeaturePipeline
from fraudops.models.evaluate import compute_metrics
from fraudops.models.threshold import optimal_cost_threshold
from fraudops.models.window_eval import (
    SECONDS_PER_DAY,
    apply_injections_frame,
    fetch_active_injections,
)
from fraudops.registry.wrapper import FraudOpsModel

EXPERIMENT_NAME = "fraudops_training"


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def labelled_stream_frame(
    conn,
    data_dir: Path,
    configs_dir: Path,
) -> tuple[pd.DataFrame, int]:
    """Stream rows whose labels are released, with truth attached.

    Returns (frame, cutoff_dt) where cutoff_dt is the latest released
    transaction's TransactionDT — everything after it has no visible label
    yet and must not enter training.
    """
    splits_cfg = _load_yaml(configs_dir / "splits.yaml")
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT MAX(available_at_dt - %s * %s) FROM labels
            """,
            (
                int(_load_yaml(configs_dir / "drift.yaml").get("label_delay_days", 7)),
                SECONDS_PER_DAY,
            ),
        )
        cutoff_dt = cur.fetchone()[0]
        if not cutoff_dt:
            raise RuntimeError("no labels released yet — nothing to retrain on")
        cur.execute("SELECT transaction_id, is_fraud, amount FROM labels")
        rows = cur.fetchall()

    truth = pd.DataFrame(rows, columns=["TransactionID", "is_fraud", "amount"])
    df = load_joined(data_dir / "train_transaction.csv", data_dir / "train_identity.csv")
    stream = df[
        sim_day(df["TransactionDT"]) >= int(splits_cfg["train_days"]) + int(splits_cfg["val_days"])
    ]
    labelled = stream[stream["TransactionID"].isin(truth["TransactionID"])].merge(
        truth, on="TransactionID", how="inner", validate="one_to_one"
    )
    labelled["is_fraud"] = labelled["is_fraud"].astype("int8")
    return labelled, int(cutoff_dt)


def split_labelled_stream(
    labelled: pd.DataFrame,
    cutoff_dt: int,
    window_sim_days: int,
    val_days: int,
) -> dict[str, pd.DataFrame]:
    """Chronological extra-train / retrain-validation / gate-eval slices."""
    eval_start = cutoff_dt - window_sim_days * SECONDS_PER_DAY
    val_start = eval_start - val_days * SECONDS_PER_DAY
    slices = {
        "extra_train": labelled[labelled["TransactionDT"] <= val_start],
        "retrain_val": labelled[
            (labelled["TransactionDT"] > val_start) & (labelled["TransactionDT"] <= eval_start)
        ],
        "gate_eval": labelled[labelled["TransactionDT"] > eval_start],
    }
    counts = {name: len(part) for name, part in slices.items()}
    if counts["retrain_val"] == 0 or counts["gate_eval"] == 0:
        raise RuntimeError(
            f"labelled stream too short for retrain windows: {counts} (cutoff_dt={cutoff_dt:,})"
        )
    max_extra = slices["extra_train"]["TransactionDT"].max()
    min_val = slices["retrain_val"]["TransactionDT"].min()
    max_val = slices["retrain_val"]["TransactionDT"].max()
    min_eval = slices["gate_eval"]["TransactionDT"].min()
    assert max_extra < min_val, f"leakage: extra-train max {max_extra} >= val min {min_val}"
    assert max_val < min_eval, f"leakage: val max {max_val} >= eval min {min_eval}"
    return slices


def train_challenger(
    database_dsn: str,
    tracking_uri: str,
    data_dir: Path | str = Path("data/raw"),
    configs_dir: Path | str = Path("configs"),
    window_sim_days: int = 7,
    val_days: int = 7,
) -> dict:
    """Train + log the challenger on labelled data; return run/version info."""
    import psycopg

    data_dir, configs_dir = Path(data_dir), Path(configs_dir)
    splits_cfg = _load_yaml(configs_dir / "splits.yaml")
    costs_cfg = _load_yaml(configs_dir / "costs.yaml")
    model_cfg = _load_yaml(configs_dir / "model.yaml")
    false_alert_cost = float(costs_cfg["false_alert_cost"])
    seed = int(model_cfg["seed"])

    with psycopg.connect(database_dsn) as conn:
        labelled, cutoff_dt = labelled_stream_frame(conn, data_dir, configs_dir)
        injections = fetch_active_injections(conn)

    slices = split_labelled_stream(labelled, cutoff_dt, window_sim_days, val_days)
    print(
        f"labelled stream: {len(labelled):,} rows | extra_train "
        f"{len(slices['extra_train']):,} / retrain_val {len(slices['retrain_val']):,} "
        f"/ gate_eval {len(slices['gate_eval']):,}"
    )
    # the challenger trains on the distribution serving actually faces
    slices = {name: apply_injections_frame(part, injections) for name, part in slices.items()}

    print("loading base training split ...", flush=True)
    df = load_joined(data_dir / "train_transaction.csv", data_dir / "train_identity.csv")
    base = chronological_split(df, int(splits_cfg["train_days"]), int(splits_cfg["val_days"]))[
        TRAIN
    ]
    train_frame = pd.concat([base, slices["extra_train"]], ignore_index=True)
    print(
        f"training frame: {len(base):,} base + {len(slices['extra_train']):,} labelled stream rows"
    )

    # ------------------------------------------------------------- features
    use_v = bool(model_cfg["features"]["use_v_columns"])
    pipeline = FeaturePipeline(use_v_columns=use_v)
    X_train = pipeline.fit_transform(train_frame)
    X_val = pipeline.transform(slices["retrain_val"])
    X_eval = pipeline.transform(slices["gate_eval"])
    y_train = train_frame["isFraud"].to_numpy()
    y_val = slices["retrain_val"]["is_fraud"].to_numpy()
    y_eval = slices["gate_eval"]["is_fraud"].to_numpy()

    # ---------------------------------------------------------------- model
    print("training LightGBM challenger ...", flush=True)
    from lightgbm import LGBMClassifier

    pos = float(y_train.sum())
    scale_pos_weight = float((len(y_train) - pos) / max(pos, 1.0))
    model = LGBMClassifier(
        random_state=seed, scale_pos_weight=scale_pos_weight, **model_cfg["lgbm"]
    )
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    print(f"  trained in {time.perf_counter() - t0:.1f}s")

    scores = {
        "retrain_val": model.predict_proba(X_val)[:, 1],
        "gate_eval": model.predict_proba(X_eval)[:, 1],
    }

    # threshold on the (drifted, chronological) retrain validation slice
    threshold = optimal_cost_threshold(
        y_val,
        slices["retrain_val"]["amount"].to_numpy(dtype="float64"),
        scores["retrain_val"],
        false_alert_cost,
    )
    print(f"cost-optimal threshold (retrain validation): {threshold:.4f}")

    fpr_target = float(costs_cfg.get("recall_fpr_target", 0.01))
    metrics = {
        name: compute_metrics(
            y,
            slices[name]["amount"].to_numpy(dtype="float64"),
            scores[name],
            threshold,
            false_alert_cost,
            fpr_target,
        )
        for name, y in (("retrain_val", y_val), ("gate_eval", y_eval))
    }

    # -------------------------------------------------------------- mlflow
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    eval_days = (
        sim_day(slices["gate_eval"]["TransactionDT"]).min(),
        sim_day(slices["gate_eval"]["TransactionDT"]).max(),
    )
    metadata = {
        "threshold": float(threshold),
        "false_alert_cost": false_alert_cost,
        "train_days": splits_cfg["train_days"],
        "val_days": splits_cfg["val_days"],
        "n_features": X_train.shape[1],
        "seed": seed,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "retrain": {
            "n_labelled_stream_rows": len(labelled),
            "n_extra_train_rows": len(slices["extra_train"]),
            "cutoff_dt": cutoff_dt,
            "gate_eval_days": list(eval_days),
            "injections": [
                {"kind": i["kind"], "params": i["params"], "from_day": i["from_day"]}
                for i in injections
            ],
        },
        "metrics": {name: dict(m) for name, m in metrics.items()},
    }
    with mlflow.start_run(run_name="fraudops-challenger-retrain") as run:
        mlflow.log_params(
            {
                "run_kind": "retrain_on_drift",
                "base_train_rows": len(base),
                "extra_labelled_rows": len(slices["extra_train"]),
                "retrain_val_rows": len(slices["retrain_val"]),
                "gate_eval_rows": len(slices["gate_eval"]),
                "n_active_injections": len(injections),
                "seed": seed,
                "threshold": round(float(threshold), 6),
            }
        )
        for name in metrics:
            mlflow.log_metrics({f"{name}_{k}": v for k, v in metrics[name].items()}, step=0)
        mlflow.pyfunc.log_model(
            name="model",
            python_model=FraudOpsModel(model, pipeline, float(threshold), metadata),
        )
        run_id = run.info.run_id

    print(
        f"challenger trained: run {run_id} | retrain_val pr_auc "
        f"{metrics['retrain_val']['pr_auc']:.3f}, cost/100k "
        f"{metrics['retrain_val']['cost_per_100k']:,.0f} | gate_eval pr_auc "
        f"{metrics['gate_eval']['pr_auc']:.3f}, cost/100k "
        f"{metrics['gate_eval']['cost_per_100k']:,.0f}"
    )
    return {
        "run_id": run_id,
        "threshold": float(threshold),
        "metrics": metrics,
        "gate_eval_days": eval_days,
        "n_labelled": int(len(slices["gate_eval"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database-url",
        default=os.environ.get(
            "DATABASE_URL",
            "postgresql://fraudops:fraudops-local@localhost:5432/fraudops",
        ),
    )
    parser.add_argument(
        "--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("FRAUDOPS_DATA_DIR", "data/raw")),
    )
    parser.add_argument("--configs-dir", type=Path, default=Path("configs"))
    parser.add_argument("--window-sim-days", type=int, default=7)
    parser.add_argument("--val-days", type=int, default=7)
    args = parser.parse_args()
    train_challenger(
        database_dsn=args.database_url,
        tracking_uri=args.tracking_uri,
        data_dir=args.data_dir,
        configs_dir=args.configs_dir,
        window_sim_days=args.window_sim_days,
        val_days=args.val_days,
    )


if __name__ == "__main__":
    main()

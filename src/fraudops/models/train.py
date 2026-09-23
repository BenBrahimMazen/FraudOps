"""Baseline training run: load -> split -> features -> LightGBM -> threshold
-> evaluation -> MLflow logging -> reports.

Usage (see Makefile ``train``):

    uv run python -m fraudops.models.train

Data flow rules enforced here:
- the feature pipeline is fitted on the training split only;
- the cost-optimal threshold is selected on the validation split only;
- the replay stream is scored for honest, never-tuned-on reporting;
- the model is logged to MLflow as a pyfunc artifact that carries the
  pipeline and the threshold with it (single artifact, no serving skew).

Tracking destination: the ``MLFLOW_TRACKING_URI`` environment variable when
set (the Docker stack's MLflow server), otherwise a local SQLite store.
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

from fraudops.data.loader import load_joined
from fraudops.data.splits import STREAM, TRAIN, VALIDATION, chronological_split
from fraudops.features.pipeline import FeaturePipeline
from fraudops.models.evaluate import baseline_costs, compute_metrics
from fraudops.models.threshold import optimal_cost_threshold
from fraudops.registry.wrapper import FraudOpsModel

EXPERIMENT_NAME = "fraudops_training"


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _resolve_tracking_uri(mlflow_dir: Path) -> str:
    env = os.environ.get("MLFLOW_TRACKING_URI")
    if env:
        return env
    mlflow_dir.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{(mlflow_dir / 'mlflow.db').as_posix()}"


def train_baseline(
    configs_dir: Path = Path("configs"),
    data_dir: Path = Path("data/raw"),
    out_dir: Path = Path("models"),
    reports_dir: Path = Path("reports"),
    mlflow_dir: Path = Path("mlruns"),
) -> dict:
    splits_cfg = _load_yaml(configs_dir / "splits.yaml")
    costs_cfg = _load_yaml(configs_dir / "costs.yaml")
    model_cfg = _load_yaml(configs_dir / "model.yaml")

    false_alert_cost = float(costs_cfg["false_alert_cost"])
    fpr_target = float(costs_cfg.get("recall_fpr_target", 0.01))
    seed = int(model_cfg["seed"])

    # ---------------------------------------------------------------- data
    print("loading data ...", flush=True)
    t0 = time.perf_counter()
    df = load_joined(data_dir / "train_transaction.csv", data_dir / "train_identity.csv")
    print(f"  joined frame: {len(df):,} rows in {time.perf_counter() - t0:.1f}s")

    splits = chronological_split(df, int(splits_cfg["train_days"]), int(splits_cfg["val_days"]))
    for name, part in splits.items():
        rate = part["isFraud"].mean()
        print(f"  {name:<11} {len(part):>7,} rows | fraud rate {rate:.4f}")

    # ------------------------------------------------------------- features
    print("fitting feature pipeline on training split ...", flush=True)
    use_v = bool(model_cfg["features"]["use_v_columns"])
    pipeline = FeaturePipeline(use_v_columns=use_v)
    X: dict[str, pd.DataFrame] = {}
    for name, part in splits.items():
        X[name] = pipeline.fit_transform(part) if name == TRAIN else pipeline.transform(part)
    y = {name: part["isFraud"].to_numpy() for name, part in splits.items()}
    amt = {name: part["TransactionAmt"].to_numpy(dtype="float64") for name, part in splits.items()}
    print(f"  feature matrix: {X[TRAIN].shape[1]} features")

    # ---------------------------------------------------------------- model
    print("training LightGBM ...", flush=True)
    from lightgbm import LGBMClassifier

    pos = float(y[TRAIN].sum())
    scale_pos_weight = float((len(y[TRAIN]) - pos) / max(pos, 1.0))
    model = LGBMClassifier(
        random_state=seed,
        scale_pos_weight=scale_pos_weight,
        **model_cfg["lgbm"],
    )
    t0 = time.perf_counter()
    model.fit(X[TRAIN], y[TRAIN])
    print(f"  trained in {time.perf_counter() - t0:.1f}s")

    scores = {name: model.predict_proba(X[name])[:, 1] for name in splits}

    # ------------------------------------------------- threshold on validation
    threshold = optimal_cost_threshold(
        y[VALIDATION], amt[VALIDATION], scores[VALIDATION], false_alert_cost
    )
    print(f"cost-optimal threshold (validation): {threshold:.4f}")

    # ----------------------------------------------------------- evaluation
    metrics = {
        name: compute_metrics(
            y[name], amt[name], scores[name], threshold, false_alert_cost, fpr_target
        )
        for name in splits
    }
    baselines = {
        name: baseline_costs(y[name], amt[name], scores[name], false_alert_cost) for name in splits
    }

    # ------------------------------------------------------------- reports
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in (TRAIN, VALIDATION, STREAM):
        rows.append(
            {
                "window": name,
                **{k: round(v, 6) for k, v in metrics[name].items()},
                **{f"cost_{k}": round(v, 2) for k, v in baselines[name].items()},
            }
        )
    results = pd.DataFrame(rows)
    results.to_csv(reports_dir / "results_table.csv", index=False)

    # -------------------------------------------------------------- mlflow
    tracking_uri = _resolve_tracking_uri(mlflow_dir)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    metadata = {
        "threshold": float(threshold),
        "false_alert_cost": false_alert_cost,
        "train_days": splits_cfg["train_days"],
        "val_days": splits_cfg["val_days"],
        "n_features": X[TRAIN].shape[1],
        "seed": seed,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "metrics": {name: dict(m) for name, m in metrics.items()},
        "baselines": {name: dict(b) for name, b in baselines.items()},
    }
    with mlflow.start_run(run_name="fraudops-baseline") as run:
        mlflow.log_params(
            {
                "train_days": splits_cfg["train_days"],
                "val_days": splits_cfg["val_days"],
                "seed": seed,
                "false_alert_cost": false_alert_cost,
                "n_features": X[TRAIN].shape[1],
                "threshold": round(float(threshold), 6),
                **{f"lgbm_{k}": v for k, v in model_cfg["lgbm"].items()},
            }
        )
        for name in splits:
            mlflow.log_metrics({f"{name}_{k}": v for k, v in metrics[name].items()}, step=0)
        mlflow.log_artifact(str(reports_dir / "results_table.csv"))
        mlflow.pyfunc.log_model(
            name="model",
            python_model=FraudOpsModel(model, pipeline, float(threshold), metadata),
        )
        run_id = run.info.run_id

    print("\nresults (cost per 100k transactions):")
    print(results.to_string(index=False))
    print(f"mlflow run: {run_id} ({tracking_uri})")
    return {
        "run_id": run_id,
        "metrics": metrics,
        "baselines": baselines,
        "threshold": float(threshold),
        "results_table": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs-dir", type=Path, default=Path("configs"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("models"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    train_baseline(
        configs_dir=args.configs_dir,
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        reports_dir=args.reports_dir,
    )


if __name__ == "__main__":
    main()

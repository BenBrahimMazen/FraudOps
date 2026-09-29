"""Figure 2: cumulative expected cost — no retraining vs scheduled vs
drift-triggered retraining.

A retrospective, leakage-clean simulation over the labelled part of the
replay (sim days 150–175; later days have no released labels, so no policy
can be scored there). Two information constraints shape the result and are
encoded, not hidden:

- **the 7-day label delay** — a retrain at day T only sees labels through
  day T−7. A weekly schedule anchored at deployment produces starved ticks
  at days 157 (no labels visible at all) and 164 (the 7 visible days are
  entirely consumed by the threshold-validation slice, leaving zero extra
  training rows); the first usable tick is day 171. Starved ticks are
  skipped and logged exactly as a real scheduler would.
- **drifted labels arrive from day 172** — the ×3 shift starts at day 165,
  so its labels only become visible at day 172: neither the day-169
  triggered retrain nor the day-171 scheduled one can train on the shift
  itself. What retraining buys here is adaptation to the natural drift of
  the stream weeks plus a threshold re-tuned on recent labels. The live
  loop's promotion (which fired once the gate window was fully labelled)
  is annotated on the figure for reference, not scored as an arm.

Policies:

- **no retraining** — the frozen champion exactly as it served the stream
  (stored decisions, nothing re-scored);
- **scheduled (weekly)** — retrain every 7 sim days from deployment;
- **drift-triggered** — retrain the day the monitor's PSI first reached
  alert level on the injected feature (day 169 live; warning at 167).

Each tick rebuilds exactly what that moment could know
(``labelled_stream_frame(as_of_dt=...)`` — the same visibility rule the
release job enforces) and fits through the same ``fit_challenger`` the
live DAG uses; costs before a policy's first retrain are the frozen
champion's served costs. Every fitted arm is logged to MLflow
(experiment ``fraudops_experiments``) so the figure traces to runs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from fraudops.data.clock import sim_day
from fraudops.data.loader import load_joined
from fraudops.data.splits import TRAIN, chronological_split
from fraudops.models.retrain import fit_challenger, labelled_stream_frame
from fraudops.models.window_eval import apply_injections_frame, fetch_active_injections
from fraudops.reports.common import (
    connect,
    data_dir,
    save_figure,
    save_table,
    served_labelled_frame,
    tracking_uri,
)

SECONDS_PER_DAY = 86_400


def _cfg(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def row_costs(y: np.ndarray, amounts: np.ndarray, flagged: np.ndarray, review: float) -> np.ndarray:
    """Per-transaction cost: missed fraud = amount, false alert = review."""
    y = y.astype(bool)
    flagged = flagged.astype(bool)
    return np.where(y & ~flagged, amounts, np.where(~y & flagged, review, 0.0))


def retrain_at(
    conn,
    data: pd.DataFrame,
    data_root: Path,
    configs_dir: Path,
    retrain_day: int,
    val_days: int,
    model_cfg: dict,
    review: float,
) -> dict | None:
    """Fit one retrain exactly as visible at ``retrain_day``.

    Returns None when the tick is starved: no labels visible at all, or the
    visible labels are entirely consumed by the threshold-validation slice
    (no extra training rows). A real scheduler would skip such a tick.
    """
    as_of = retrain_day * SECONDS_PER_DAY
    try:
        labelled, cutoff = labelled_stream_frame(conn, data_root, configs_dir, as_of_dt=as_of)
    except RuntimeError:
        return None  # no labels visible at all yet
    val_start = cutoff - val_days * SECONDS_PER_DAY
    extra = labelled[labelled["TransactionDT"] <= val_start]
    val = labelled[labelled["TransactionDT"] > val_start]
    if len(extra) == 0 or len(val) == 0:
        return None  # starved: validation slice consumed everything
    assert extra["TransactionDT"].max() < val["TransactionDT"].min(), "leakage: extra >= val"
    assert val["TransactionDT"].max() < as_of, "leakage: val reaches past as_of"

    splits_cfg = _cfg(configs_dir / "splits.yaml")
    base = chronological_split(data, int(splits_cfg["train_days"]), int(splits_cfg["val_days"]))[
        TRAIN
    ]
    model, pipeline, threshold = fit_challenger(
        pd.concat([base, extra], ignore_index=True),
        val,
        val["amount"].to_numpy(dtype="float64"),
        model_cfg,
        review,
    )
    return {
        "model": model,
        "pipeline": pipeline,
        "threshold": threshold,
        "retrain_day": retrain_day,
        "n_extra": len(extra),
        "n_val": len(val),
        "cutoff": cutoff,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-version", type=int, default=1)
    parser.add_argument(
        "--schedule-cadence-days",
        type=int,
        default=7,
        help="scheduled cadence anchored at stream start",
    )
    parser.add_argument(
        "--triggered-day",
        type=int,
        default=169,
        help="live monitor: PSI alert on log_TransactionAmt",
    )
    parser.add_argument("--val-days", type=int, default=7)
    args = parser.parse_args()

    configs_dir = Path("configs")
    model_cfg = _cfg(configs_dir / "model.yaml")
    review = float(_cfg(configs_dir / "costs.yaml")["false_alert_cost"])

    served = served_labelled_frame(args.baseline_version)
    served_day = (served["sim_ts"] // SECONDS_PER_DAY).astype(int)
    stream_start, labelled_end = int(served_day.min()), int(served_day.max())

    data = load_joined(data_dir() / "train_transaction.csv", data_dir() / "train_identity.csv")
    with connect() as conn:
        injections = fetch_active_injections(conn)

        scheduled_ticks = list(
            range(
                stream_start + args.schedule_cadence_days,
                labelled_end + 1,
                args.schedule_cadence_days,
            )
        )
        policies: dict[str, dict] = {
            "scheduled": {"ticks": scheduled_ticks, "retrains": [], "skipped": []},
            "triggered": {"ticks": [args.triggered_day], "retrains": [], "skipped": []},
        }
        for name, policy in policies.items():
            for day in policy["ticks"]:
                fit = retrain_at(
                    conn, data, data_dir(), configs_dir, day, args.val_days, model_cfg, review
                )
                if fit is None:
                    policy["skipped"].append(day)
                    print(f"{name}: tick day {day} starved — skipped", flush=True)
                else:
                    policy["retrains"].append(fit)
                    print(
                        f"{name}: retrained at day {day} "
                        f"(+{fit['n_extra']:,} labelled rows, threshold {fit['threshold']:.4f})",
                        flush=True,
                    )

        with conn.cursor() as cur:
            cur.execute("SELECT transaction_id, is_fraud, amount FROM labels")
            truth = pd.DataFrame(cur.fetchall(), columns=["TransactionID", "is_fraud", "amount"])

    stream = data.merge(truth, on="TransactionID", how="inner", validate="one_to_one")
    stream = stream[sim_day(stream["TransactionDT"]) >= stream_start]
    stream = stream.sort_values("TransactionDT").reset_index(drop=True)
    stream_day = sim_day(stream["TransactionDT"])  # ndarray already
    stream = apply_injections_frame(stream, injections)

    y_all = stream["is_fraud"].to_numpy()
    amounts_all = stream["amount"].to_numpy(dtype="float64")
    served_flagged = (served["decision"] == "alert").to_numpy()
    assert np.array_equal(served["sim_ts"].to_numpy(), stream["TransactionDT"].to_numpy()), (
        "served frame and stream frame disagree"
    )

    cost_rows: dict[str, np.ndarray] = {
        "no retraining": row_costs(y_all, amounts_all, served_flagged, review)
    }
    arm_scores: dict[str, np.ndarray] = {}
    for name, policy in policies.items():
        fits = policy["retrains"]
        if not fits:
            policy["label"] = None
            continue
        flagged = served_flagged.copy()
        scores = np.full(len(stream), np.nan)
        for i, fit in enumerate(fits):
            start = int(np.searchsorted(stream_day, fit["retrain_day"], side="left"))
            end = (
                int(np.searchsorted(stream_day, fits[i + 1]["retrain_day"], side="left"))
                if i + 1 < len(fits)
                else len(stream)
            )
            X = fit["pipeline"].transform(stream.iloc[start:end])
            s = fit["model"].predict_proba(X)[:, 1]
            flagged[start:end] = s >= fit["threshold"]
            scores[start:end] = s
        arm_scores[name] = scores
        policy["label"] = f"{name} (day {fits[0]['retrain_day']})"
        cost_rows[policy["label"]] = row_costs(y_all, amounts_all, flagged, review)

    days = np.arange(stream_start, labelled_end + 1)
    cum = {}
    for label, costs in cost_rows.items():
        daily = np.array([costs[stream_day == d].sum() for d in days])
        cum[label] = np.cumsum(daily)

    from sklearn.metrics import average_precision_score

    table_rows = [
        {
            "policy": label,
            "n_retrains": 0
            if label == "no retraining"
            else len(policies[label.split(" ")[0]]["retrains"]),
            "total_cost_days_150_175": round(float(costs.sum()), 0),
            "cost_per_100k": round(float(costs.sum()) / len(costs) * 100_000, 0),
        }
        for label, costs in cost_rows.items()
    ]
    for name, policy in policies.items():
        for fit in policy["retrains"]:
            active = stream_day >= fit["retrain_day"]
            table_rows.append(
                {
                    "policy": f"  {name} active window (days {fit['retrain_day']}-{labelled_end})",
                    "n_retrains": len(policy["retrains"]),
                    "total_cost_days_150_175": round(
                        float(cost_rows[policy["label"]][active].sum()), 0
                    ),
                    "cost_per_100k": round(
                        float(cost_rows[policy["label"]][active].sum()) / active.sum() * 100_000, 0
                    ),
                    "pr_auc": round(
                        float(average_precision_score(y_all[active], arm_scores[name][active])), 4
                    ),
                    "threshold": round(fit["threshold"], 6),
                    "n_extra_train": fit["n_extra"],
                    "n_retrain_val": fit["n_val"],
                }
            )
        for day in policy["skipped"]:
            table_rows.append(
                {
                    "policy": f"  {name} tick day {day} skipped",
                    "n_retrains": 0,
                    "total_cost_days_150_175": np.nan,
                    "note": "starved: no usable labelled window (7-day delay)",
                }
            )
    table = pd.DataFrame(table_rows)
    print(table.to_string(index=False))
    save_table("retrain_comparison.csv", table)

    # MLflow traceability: one run per fitted retrain
    import mlflow

    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment("fraudops_experiments")
    for name, policy in policies.items():
        for fit in policy["retrains"]:
            active = stream_day >= fit["retrain_day"]
            with mlflow.start_run(run_name=f"exp-{name}-{fit['retrain_day']}"):
                mlflow.log_params(
                    {
                        "run_kind": "retrain_experiment",
                        "policy": name,
                        "retrain_day": fit["retrain_day"],
                        "cutoff_dt": fit["cutoff"],
                        "n_extra_train": fit["n_extra"],
                        "n_retrain_val": fit["n_val"],
                        "threshold": round(fit["threshold"], 6),
                        "seed": model_cfg["seed"],
                        "skipped_ticks": ",".join(str(d) for d in policy["skipped"]) or "none",
                    }
                )
                mlflow.log_metrics(
                    {
                        "active_cost_per_100k": float(cost_rows[policy["label"]][active].sum())
                        / active.sum()
                        * 100_000,
                        "total_cost_days_150_175": float(cost_rows[policy["label"]].sum()),
                    }
                )

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    colors = {"no retraining": "#7f7f7f", "scheduled": "#1f77b4", "triggered": "#d62728"}
    for label, series in cum.items():
        key = "no retraining" if label == "no retraining" else label.split(" ")[0]
        ax.plot(days, series / 1000, label=label, lw=1.6, color=colors.get(key))
    scheduled_first = next(
        (p["retrains"][0]["retrain_day"] for p in [policies["scheduled"]] if p["retrains"]), None
    )
    marks = [
        (165, "#2ca02c", "amount ×3 injected"),
        (args.triggered_day, "#d62728", "PSI alert → triggered retrain"),
        (175, "#9467bd", "labels end / live promotion"),
    ]
    if scheduled_first:
        marks.insert(2, (scheduled_first, "#1f77b4", "weekly schedule's first usable tick"))
    for day, color, text in marks:
        ax.axvline(day, color=color, ls="--", lw=1)
        ax.text(
            day + 0.15,
            ax.get_ylim()[1] * 0.02,
            text,
            rotation=90,
            fontsize=6.5,
            color=color,
            va="bottom",
        )
    ax.set_xlabel("simulated day (only labelled days shown — labels end at 175)")
    ax.set_ylabel("cumulative expected cost [k$]")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Retraining policies under a 7-day label delay (review cost 5.0)", fontsize=10)
    save_figure(fig, "retrain_comparison.png")


if __name__ == "__main__":
    main()

"""Figure 1: frozen-model decay over the replay period.

The champion that served the whole stream (v1) is judged only on what it
actually served: stored scores and decisions joined to the labels that were
released 7 simulated days later. Trailing 7-day windows stepped one day at a
time — the same window shape the monitor uses — give PR-AUC and expected
cost per 100k as the replay advances. No model is re-scored; this is the
deployment record.
"""

from __future__ import annotations

import argparse

import pandas as pd
from sklearn.metrics import average_precision_score

from fraudops.models.cost import cost_per_100k
from fraudops.reports.common import save_figure, save_table, served_labelled_frame

SECONDS_PER_DAY = 86_400
WINDOW_DAYS = 7
FALSE_ALERT_COST = 5.0


def decay_curve(served: pd.DataFrame) -> pd.DataFrame:
    """Rolling PR-AUC + cost per 100k, one row per trailing window."""
    day = (served["sim_ts"] // SECONDS_PER_DAY).astype(int)
    rows = []
    for end_day in range(day.min() + WINDOW_DAYS, day.max() + 1):
        mask = (day > end_day - WINDOW_DAYS) & (day <= end_day)
        part = served[mask]
        y = part["is_fraud"].to_numpy(dtype=int)
        if y.sum() == 0:
            continue  # PR-AUC undefined; cost would still be false alerts only
        scores = part["score"].to_numpy(dtype="float64")
        rows.append(
            {
                "window_end_day": end_day,
                "n": len(part),
                "fraud_rate": float(y.mean()),
                "pr_auc": float(average_precision_score(y, scores)),
                "cost_per_100k": cost_per_100k(
                    y,
                    part["amount"].to_numpy(dtype="float64"),
                    part["decision"].to_numpy() == "alert",
                    FALSE_ALERT_COST,
                ),
                "alert_rate": float((part["decision"] == "alert").mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-version", type=int, default=1, help="the frozen served model")
    parser.add_argument("--injection-day", type=int, default=165, help="vertical marker")
    args = parser.parse_args()

    served = served_labelled_frame(args.model_version)
    curve = decay_curve(served)
    print(curve.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    save_table("decay_curve.csv", curve)

    from fraudops.reports.common import model_metrics

    metrics = model_metrics(args.model_version)
    val_pr_auc = metrics.get("validation_pr_auc")
    val_cost = metrics.get("validation_cost_per_100k")

    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5.5), sharex=True)
    d = curve["window_end_day"]

    ax1.plot(d, curve["pr_auc"], marker="o", ms=3, color="#1f77b4")
    if val_pr_auc is not None:
        ax1.axhline(val_pr_auc, color="#7f7f7f", ls=":", lw=1)
        ax1.text(
            d.iloc[0],
            val_pr_auc + 0.012,
            f"original validation PR-AUC {val_pr_auc:.3f}",
            fontsize=7,
            color="#7f7f7f",
        )
    ax1.set_ylabel("PR-AUC (7-day window)")
    ax1.set_ylim(0, 1)

    ax2.plot(d, curve["cost_per_100k"], marker="o", ms=3, color="#d62728")
    if val_cost is not None:
        ax2.axhline(val_cost, color="#7f7f7f", ls=":", lw=1)
        ax2.text(
            d.iloc[0],
            val_cost * 1.05,
            f"validation cost/100k {val_cost:,.0f}",
            fontsize=7,
            color="#7f7f7f",
        )
    ax2.set_ylabel("expected cost per 100k [$]")
    ax2.set_xlabel("simulated day (window end)")

    for ax in (ax1, ax2):
        ax.axvline(args.injection_day, color="#2ca02c", ls="--", lw=1)
    ax1.text(
        args.injection_day + 0.3,
        0.97,
        f"amount ×3 injected (day {args.injection_day})",
        fontsize=7,
        color="#2ca02c",
    )
    fig.suptitle(
        f"Frozen champion v{args.model_version} decay over the replay "
        "(as served, labels lag 7 days)",
        fontsize=10,
    )
    save_figure(fig, "decay_curve.png")


if __name__ == "__main__":
    main()

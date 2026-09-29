"""Figure 3: how the cost-optimal threshold reacts to the review cost.

The business lever a fraud team actually tunes: if reviewing one legitimate
alert costs 1, 5, 10 or 25 currency units, where does the cost-optimal
operating point move? Scored on the original validation split (the set the
baseline's own threshold was tuned on) with the baseline model, exactly as
training does — the optimum marked on each curve is the same
``optimal_cost_threshold`` the trainer uses, not a re-derivation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from fraudops.data.loader import load_joined
from fraudops.data.splits import VALIDATION, chronological_split
from fraudops.models.cost import cost_per_100k, total_cost
from fraudops.models.threshold import optimal_cost_threshold
from fraudops.reports.common import data_dir, load_bundle, save_figure, save_table

REVIEW_COSTS = [1.0, 5.0, 10.0, 25.0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-version", type=int, default=1, help="the baseline model")
    args = parser.parse_args()

    configs_dir = Path("configs")
    with (configs_dir / "splits.yaml").open(encoding="utf-8") as fh:
        splits_cfg = yaml.safe_load(fh)
    val_days = int(splits_cfg["val_days"])

    bundle, version = load_bundle(args.model_version)
    df = load_joined(data_dir() / "train_transaction.csv", data_dir() / "train_identity.csv")
    val = chronological_split(df, int(splits_cfg["train_days"]), val_days)[VALIDATION]

    y = val["isFraud"].to_numpy()
    amounts = val["TransactionAmt"].to_numpy(dtype="float64")
    scores = bundle.score_raw(val)
    grid = np.unique(np.concatenate([np.linspace(0.0, 1.0, 201), np.asarray([bundle.threshold])]))

    rows = []
    for review_cost in REVIEW_COSTS:
        costs = np.array([cost_per_100k(y, amounts, scores >= t, review_cost) for t in grid])
        optimum = optimal_cost_threshold(y, amounts, scores, review_cost)
        flagged = scores >= optimum
        cost_at_opt = total_cost(y, amounts, flagged, review_cost)
        missed = total_cost(y, amounts, np.zeros_like(flagged), review_cost)  # approve-all
        rows.append(
            {
                "review_cost": review_cost,
                "optimal_threshold": round(optimum, 6),
                "cost_per_100k": round(cost_at_opt / len(y) * 100_000, 0),
                "alert_rate": round(float(flagged.mean()), 4),
                "recall": round(float((y.astype(bool) & flagged).sum() / max(y.sum(), 1)), 4),
                "missed_fraud_share_of_cost": round(
                    (cost_at_opt - review_cost * float((~y.astype(bool) & flagged).sum()))
                    / cost_at_opt,
                    3,
                ),
                "_grid_costs": costs,
                "_missed_baseline": missed / len(y) * 100_000,
            }
        )

    table = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    print(
        "\nreview cost | threshold | cost/100k | alert rate | recall\n"
        + "\n".join(
            f"{r['review_cost']:>11.0f} | {r['optimal_threshold']:.6f} | "
            f"{r['cost_per_100k']:>9,.0f} | {r['alert_rate']:>10.3f} | {r['recall']:.3f}"
            for r in table
        )
    )
    save_table("threshold_sensitivity.csv", pd.DataFrame(table))

    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(rows)))
    for row, color in zip(rows, colors, strict=True):
        label = f"review cost {row['review_cost']:.0f}"
        ax1.plot(grid, row["_grid_costs"], color=color, lw=1.4, label=label)
        opt_y = row["cost_per_100k"]
        ax1.plot(row["optimal_threshold"], opt_y, "v", color=color, ms=6)
    ax1.set_xlabel("decision threshold")
    ax1.set_ylabel("expected cost per 100k [$]")
    ax1.set_title("cost vs threshold, by review cost", fontsize=9)
    ax1.legend(fontsize=7)

    rc = [r["review_cost"] for r in rows]
    ax2.plot(rc, [r["optimal_threshold"] for r in rows], "o-", color="#1f77b4")
    ax2.set_xlabel("false-alert review cost [$]")
    ax2.set_ylabel("cost-optimal threshold", color="#1f77b4")
    ax2b = ax2.twinx()
    ax2b.plot(rc, [r["cost_per_100k"] for r in rows], "s--", color="#d62728")
    ax2b.set_ylabel("cost at optimum per 100k [$]", color="#d62728")
    ax2b.grid(False)
    ax2.set_title("operating point vs review cost", fontsize=9)
    fig.suptitle(
        f"Threshold sensitivity — model v{version}, validation split "
        f"(days {splits_cfg['train_days']}–{splits_cfg['train_days'] + val_days})",
        fontsize=10,
    )
    save_figure(fig, "threshold_sensitivity.png")


if __name__ == "__main__":
    main()

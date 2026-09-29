"""Figure 4: detection lag across injected shifts.

How long does the monitor take to notice each synthetic shift? Six
scenarios are replayed OFFLINE through the exact monitor semantics: the
champion's own reference (top-20 features + score distribution, fitted on
the training split), trailing 7-day windows stepped one simulated day,
PSI with warning 0.1 / alert 0.2 — the same machinery that ran live.

Only the scenario's own injection is applied (the live ×3 shift is
excluded), so each row isolates that shift's detection lag. The live
replay's ×3 measurement (warning day 167, alert day 169) is the
cross-check the offline numbers must reproduce.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from fraudops.data.clock import sim_day
from fraudops.data.loader import load_joined
from fraudops.models.window_eval import apply_injections_frame
from fraudops.monitoring.drift import evaluate_drift
from fraudops.monitoring.monitor import build_reference
from fraudops.reports.common import data_dir, load_bundle, save_figure, save_table

SECONDS_PER_DAY = 86_400
WINDOW_DAYS = 7
WARNING = 0.1
ALERT = 0.2

SCENARIOS = [
    (
        "amount ×1.5",
        {"kind": "amount_factor", "params": {"factor": 1.5}, "from_day": 165},
        "log_TransactionAmt",
    ),
    (
        "amount ×2",
        {"kind": "amount_factor", "params": {"factor": 2.0}, "from_day": 165},
        "log_TransactionAmt",
    ),
    (
        "amount ×3 (live)",
        {"kind": "amount_factor", "params": {"factor": 3.0}, "from_day": 165},
        "log_TransactionAmt",
    ),
    (
        "amount ×5",
        {"kind": "amount_factor", "params": {"factor": 5.0}, "from_day": 165},
        "log_TransactionAmt",
    ),
    ("nullify C13", {"kind": "nullify", "params": {"columns": ["C13"]}, "from_day": 165}, "C13"),
    (
        "nullify P_emaildomain",
        {"kind": "nullify", "params": {"columns": ["P_emaildomain"]}, "from_day": 165},
        "P_emaildomain",
    ),
]


def _feature_values(X: pd.DataFrame, name: str) -> np.ndarray:
    col = X[name]
    if str(col.dtype) == "category":
        return np.asarray([None if pd.isna(v) else str(v) for v in col], dtype=object)
    return col.to_numpy(dtype="float64")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-version",
        type=int,
        default=1,
        help="the model that served the replay (its reference detected)",
    )
    args = parser.parse_args()

    configs_dir = Path("configs")
    with (configs_dir / "splits.yaml").open(encoding="utf-8") as fh:
        splits_cfg = yaml.safe_load(fh)
    stream_start = int(splits_cfg["train_days"]) + int(splits_cfg["val_days"])

    bundle, version = load_bundle(args.model_version)
    reference = build_reference(bundle, data_dir(), configs_dir, top_k=20, version=version)
    ref_features = {name: reference.feature_reference[name] for name in reference.feature_names}
    ref_score = reference.score_reference

    df = load_joined(data_dir() / "train_transaction.csv", data_dir() / "train_identity.csv")
    stream = df[sim_day(df["TransactionDT"]) >= stream_start].reset_index(drop=True)
    day = (stream["TransactionDT"] // SECONDS_PER_DAY).astype(int).to_numpy()
    end_days = list(range(stream_start + WINDOW_DAYS, int(day.max()) + 1))

    rows = []
    for label, injection, affected in SCENARIOS:
        injected = apply_injections_frame(stream, [injection])
        X = bundle.pipeline.transform(injected)
        scores = bundle.model.booster_.predict(X)
        affected_values = _feature_values(X, affected)

        psis, score_psis = {}, {}
        for end in end_days:
            mask = (day > end - WINDOW_DAYS) & (day <= end)
            result = evaluate_drift(
                affected, ref_features[affected], affected_values[mask], WARNING, ALERT, bins=10
            )
            psis[end] = result
            score_psis[end] = evaluate_drift(
                "score", ref_score, scores[mask], WARNING, ALERT, bins=10
            ).psi

        from_day = int(injection["from_day"])
        first_warning = next((e for e in end_days if psis[e].level.value != "ok"), None)
        first_alert = next((e for e in end_days if psis[e].level.value == "alert"), None)
        # a channel already drifting naturally cannot measure THIS shift's lag
        masked = first_warning is not None and first_warning < from_day

        def _lag(first: int | None, _masked: bool = masked, _from: int = from_day) -> float:
            if first is None or _masked:
                return float("nan")
            return float(first - _from)

        rows.append(
            {
                "scenario": label,
                "affected_feature": affected,
                "from_day": from_day,
                "first_warning_day": first_warning,
                "lag_warning_days": _lag(first_warning),
                "first_alert_day": first_alert,
                "lag_alert_days": _lag(first_alert),
                "masked_by_natural_drift": masked,
                "max_psi": round(max(p.psi for p in psis.values()), 3),
                "max_score_psi": round(max(score_psis.values()), 4),
            }
        )
        lag_note = "MASKED by natural drift (already alerting pre-injection)" if masked else ""
        print(
            f"{label:>22}: warning day {first_warning} (lag {_lag(first_warning):.0f}), "
            f"alert day {first_alert} (lag {_lag(first_alert):.0f}), "
            f"max PSI {rows[-1]['max_psi']}, max score PSI {rows[-1]['max_score_psi']} {lag_note}"
        )

    table = pd.DataFrame(rows)
    save_table("detection_lag.csv", table)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    y_pos = np.arange(len(rows))[::-1]
    warn = [r["lag_warning_days"] for r in rows]
    alert = [r["lag_alert_days"] for r in rows]
    labels = [r["scenario"] + (" (masked)" if r["masked_by_natural_drift"] else "") for r in rows]
    ax.barh(y_pos + 0.2, warn, height=0.38, color="#ff7f0e", label="lag to warning (PSI > 0.1)")
    ax.barh(y_pos - 0.2, alert, height=0.38, color="#d62728", label="lag to alert (PSI > 0.2)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("simulated days from injection (day 165) to first detection")
    ax.text(
        0.98,
        0.04,
        '"masked": the feature was already in natural-drift alert\n'
        "before the injection — this shift's lag is not measurable",
        transform=ax.transAxes,
        ha="right",
        fontsize=6.5,
        color="#7f7f7f",
    )
    for y, w, a in zip(y_pos, warn, alert, strict=True):
        if not np.isnan(w):
            ax.text(w + 0.1, y + 0.2, f"{w:.0f}d", va="center", fontsize=7)
        if not np.isnan(a):
            ax.text(a + 0.1, y - 0.2, f"{a:.0f}d", va="center", fontsize=7)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title(
        f"Detection lag by injected shift — offline replay of monitor semantics "
        f"(v{version} reference, 7-day windows)",
        fontsize=9,
    )
    save_figure(fig, "detection_lag.png")


if __name__ == "__main__":
    main()

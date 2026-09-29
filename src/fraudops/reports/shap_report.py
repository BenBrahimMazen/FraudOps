"""Figure 6: SHAP summary + example reason codes on the current champion.

Explainability evidence for the serving layer: a beeswarm of TreeSHAP
contributions over a sample of the most recent fully-labelled window (the
drifted distribution the champion actually faces), and three worked
reason-code examples — a clear fraud, a clear legitimate transaction and a
borderline case — produced by the same native-TreeSHAP path the API uses.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

SAMPLE_ROWS = 5_000
SEED = 42
REASONS_PATH = Path("reports/figures/reason_codes.md")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-sim-days", type=int, default=7)
    args = parser.parse_args()

    import shap

    from fraudops.models.window_eval import fetch_labelled_window
    from fraudops.reports.common import connect, data_dir, load_bundle, save_figure
    from fraudops.serving.reasons import reasons_from_contributions

    bundle, version = load_bundle()  # current champion
    with connect() as conn:
        window, info = fetch_labelled_window(conn, data_dir(), args.window_sim_days)
    print(f"window {info['desc']} | champion v{version} threshold {bundle.threshold:.4f}")

    sample = window.sample(n=min(SAMPLE_ROWS, len(window)), random_state=SEED)
    X = bundle.pipeline.transform(sample)
    explainer = shap.TreeExplainer(bundle.model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):  # LGBM binary: [class0, class1]
        shap_values = shap_values[1]
    shap_values = np.asarray(shap_values)

    shap.summary_plot(shap_values, X, max_display=15, show=False, plot_size=(8, 6))
    import matplotlib.pyplot as plt

    fig = plt.gcf()
    fig.suptitle(
        f"TreeSHAP contributions — champion v{version}, {info['desc']}", fontsize=10, y=1.02
    )
    save_figure(fig, "shap_summary.png")

    # ----------------------------- reason codes: clear fraud / clear legit / borderline
    y = window["is_fraud"].to_numpy()
    X_all = bundle.pipeline.transform(window)
    scores = bundle.model.booster_.predict(X_all)
    contribs = bundle.model.booster_.predict(X_all, pred_contrib=True)

    picks = {
        "highest score (alerted)": int(np.argmax(scores)),
        "borderline (nearest threshold)": int(np.argmin(np.abs(scores - bundle.threshold))),
        "lowest score (approved)": int(np.argmin(scores)),
    }
    lines = [
        f"# Reason codes — champion v{version}, window {info['desc']}",
        "",
        "Top-3 features by |contribution| (log-odds, positive pushes toward fraud),",
        "exactly as `/score` computes them.",
        "",
    ]
    for title, idx in picks.items():
        reasons = reasons_from_contributions(contribs[idx, :-1], X_all.iloc[idx : idx + 1])
        lines.append(
            f"## {title} — score {scores[idx]:.3f}, actual {'FRAUD' if y[idx] else 'legitimate'}"
        )
        lines.append("")
        lines.append("| feature | value | contribution (log-odds) |")
        lines.append("|---|---|---|")
        for r in reasons:
            lines.append(f"| `{r['feature']}` | {r['value']} | {r['contribution']:+.3f} |")
        lines.append("")
    out = REASONS_PATH
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

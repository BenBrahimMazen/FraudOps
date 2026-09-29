"""Evidently HTML reports for the replayed window (on demand).

Human-readable snapshots complementing the always-on PSI/KS metrics:
``make report-evidently`` writes ``reports/evidently/evidently_day<simday>.html``.
Kept out of the hot monitoring loop — Evidently's value here is the report, not
the arithmetic (our own tested PSI module feeds Prometheus and the trigger).

The reference comes from ``monitor.build_reference`` — the same code path the
live monitor uses — so the report compares the window against the champion's
training reference through the real feature pipeline, engineered features
included. Written against evidently 0.7 (``evidently.presets``).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import psycopg

from fraudops.monitoring.monitor import build_reference, fetch_window
from fraudops.serving.model_loader import ModelHolder


def build_report(
    database_dsn: str,
    tracking_uri: str,
    model_name: str,
    data_dir: Path,
    configs_dir: Path,
    output_dir: Path,
    window_sim_days: int = 7,
    features: int = 10,
) -> Path:
    from evidently.future.report import Report
    from evidently.presets.drift import DataDriftPreset

    holder = ModelHolder(tracking_uri=tracking_uri, model_name=model_name)
    loaded = holder.load()

    conn = psycopg.connect(database_dsn)
    window = fetch_window(conn, window_sim_days)
    if not window:
        raise SystemExit("no predictions yet — replay something first")

    reference = build_reference(
        loaded.bundle, data_dir, configs_dir, top_k=features, version=loaded.version
    )

    # current window: one value per (transaction, top-K feature); every scored
    # transaction stores all K features, so all lists share one length
    by_feature: dict[str, list] = {name: [] for name in reference.feature_names}
    for feature, value_num, value_text, _sim_ts in window["features"]:
        col = by_feature.get(feature)
        if col is not None:
            col.append(value_text if value_num is None else value_num)
    n_rows = min(len(v) for v in by_feature.values())
    current_df = pd.DataFrame({name: values[:n_rows] for name, values in by_feature.items()})
    current_df["score"] = [p[1] for p in window["predictions"]][:n_rows]

    reference_df = pd.DataFrame(
        {name: list(values) for name, values in reference.feature_reference.items()}
    )
    reference_df["score"] = reference.score_reference

    # The monitor's reference stores categorical dtype as strings while the
    # consumer persists numeric categories as value_num floats — coerce each
    # column to one type across both frames or evidently 0.7 rejects the pair
    for col in current_df.columns:
        if pd.api.types.is_numeric_dtype(current_df[col]):
            reference_df[col] = pd.to_numeric(reference_df[col], errors="coerce")
        else:
            reference_df[col] = reference_df[col].astype("string")
            current_df[col] = current_df[col].astype("string")

    # evidently 0.7: run() returns a Snapshot which owns the HTML export
    snapshot = Report([DataDriftPreset()]).run(reference_data=reference_df, current_data=current_df)
    day = int(window["max_dt"] // 86_400)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"evidently_day{day}.html"
    snapshot.save_html(str(path))
    print(
        f"evidently report written: {path} "
        f"(reference champion v{loaded.version}, {len(reference_df):,} train rows vs "
        f"{n_rows:,} window rows, {len(current_df.columns)} columns)"
    )
    return path


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
        "--model-name", default=os.environ.get("FRAUDOPS_MODEL_NAME", "fraudops-lightgbm")
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path(os.environ.get("FRAUDOPS_DATA_DIR", "data/raw"))
    )
    parser.add_argument("--configs-dir", type=Path, default=Path("configs"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/evidently"))
    parser.add_argument("--window-sim-days", type=int, default=7)
    parser.add_argument("--features", type=int, default=10)
    args = parser.parse_args()
    build_report(
        database_dsn=args.database_url,
        tracking_uri=args.tracking_uri,
        model_name=args.model_name,
        data_dir=args.data_dir,
        configs_dir=args.configs_dir,
        output_dir=args.output_dir,
        window_sim_days=args.window_sim_days,
        features=args.features,
    )


if __name__ == "__main__":
    main()

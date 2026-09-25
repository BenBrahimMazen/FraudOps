"""Evidently HTML reports for the replayed window (on demand).

Human-readable snapshots complementing the always-on PSI/KS metrics:
``make report-evidently`` writes ``reports/evidently/<sim_day>.html``. Kept
out of the hot monitoring loop — Evidently's value here is the report, not
the arithmetic (our own tested PSI module feeds Prometheus and the trigger).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
import yaml

from fraudops.data.clock import sim_day
from fraudops.data.loader import load_joined
from fraudops.storage import ensure_schema


def build_report(
    database_dsn: str,
    data_dir: Path,
    configs_dir: Path,
    output_dir: Path,
    window_sim_days: int = 7,
    features: int = 10,
) -> Path:
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report

    with (configs_dir / "splits.yaml").open(encoding="utf-8") as fh:
        splits_cfg = yaml.safe_load(fh)

    conn = psycopg.connect(database_dsn)
    ensure_schema(conn)
    with conn.cursor() as cur:
        cur.execute("SELECT COALESCE(MAX(sim_ts), 0) FROM predictions")
        max_dt = cur.fetchone()[0]
        if not max_dt:
            raise SystemExit("no predictions yet — replay something first")
        cur.execute(
            """
            SELECT pf.feature, pf.value_num, pf.value_text
            FROM prediction_features pf
            WHERE pf.sim_ts >= %s
            """,
            (max_dt - window_sim_days * 86_400,),
        )
        rows = cur.fetchall()
        cur.execute(
            "SELECT fraud_probability FROM predictions WHERE sim_ts >= %s",
            (max_dt - window_sim_days * 86_400,),
        )
        window_scores = [r[0] for r in cur.fetchall()]

    by_feature: dict[str, list] = {}
    for feature, value_num, value_text in rows:
        by_feature.setdefault(feature, []).append(value_text if value_num is None else value_num)
    window_df = pd.DataFrame({name: values for name, values in list(by_feature.items())[:features]})
    window_df["score"] = window_scores

    # reference: the training split through the same feature columns
    df = load_joined(data_dir / "train_transaction.csv", data_dir / "train_identity.csv")
    train = df[sim_day(df["TransactionDT"]) < int(splits_cfg["train_days"])]
    train = train.sample(n=min(len(window_df) * 4, len(train)), random_state=42)
    reference_df = window_df.iloc[0:0].copy()
    for col in window_df.columns:
        if col == "score":
            continue
        if col.startswith(("log_", "sim_")):
            continue
        raw = train.get(col)
        reference_df[col] = raw.astype(object).where(raw.notna()) if raw is not None else np.nan

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df, current_data=window_df)
    day = int(max_dt // 86_400)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"evidently_day{day}.html"
    report.save_html(str(path))
    print(f"evidently report written: {path}")
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
        "--data-dir", type=Path, default=Path(os.environ.get("FRAUDOPS_DATA_DIR", "data/raw"))
    )
    parser.add_argument("--configs-dir", type=Path, default=Path("configs"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/evidently"))
    parser.add_argument("--window-sim-days", type=int, default=7)
    parser.add_argument("--features", type=int, default=10)
    args = parser.parse_args()
    build_report(
        database_dsn=args.database_url,
        data_dir=args.data_dir,
        configs_dir=args.configs_dir,
        output_dir=args.output_dir,
        window_sim_days=args.window_sim_days,
        features=args.features,
    )


if __name__ == "__main__":
    main()

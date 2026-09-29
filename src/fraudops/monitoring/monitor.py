"""Drift monitor: windows of scored traffic -> PSI/KS/performance -> alerts.

A long-running service (``python -m fraudops.monitoring.monitor``):

1. builds the drift REFERENCE from the training split with the current
   champion (top-K features by importance + the score distribution),
   re-building it whenever the champion changes;
2. every cycle: gathers the last ``window_sim_days`` of predictions and
   stored feature values, computes feature PSI/KS + score PSI, evaluates the
   rolling performance on RELEASED labels only (the label-delay contract),
   writes ``monitoring_results`` / ``drift_alerts`` and evaluates the
   retrain trigger (logged here; the closed-loop DAG consumes it in the
   next phase);
3. exposes Prometheus metrics on :9101.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
import yaml
from psycopg.types.json import Json
from sklearn.metrics import average_precision_score

from fraudops.data.clock import sim_day
from fraudops.data.loader import load_joined
from fraudops.models.evaluate import compute_metrics
from fraudops.monitoring.drift import DriftResult, Level, evaluate_drift
from fraudops.monitoring.trigger import (
    PerformanceSnapshot,
    TriggerConfig,
    should_retrain,
)
from fraudops.registry.wrapper import FraudOpsModel
from fraudops.serving.model_loader import ModelHolder
from fraudops.storage import ensure_schema

logger = logging.getLogger("fraudops.monitor")


@dataclass
class Reference:
    champion_version: int
    feature_names: list[str]
    feature_reference: dict[str, np.ndarray]
    score_reference: np.ndarray
    pr_auc: float
    cost_per_100k: float


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    defaults = {
        "top_k_features": 20,
        "psi": {"warning": 0.1, "alert": 0.2, "bins": 10},
        "window_sim_days": 7,
        "min_window_predictions": 200,
    }
    for key, value in defaults.items():
        cfg.setdefault(key, value)
    return cfg


def build_reference(
    bundle: FraudOpsModel,
    data_dir: Path,
    configs_dir: Path,
    top_k: int,
    version: int,
    sample_rows: int = 100_000,
) -> Reference:
    """Top-K features + score distribution + performance on the training split."""
    with (configs_dir / "splits.yaml").open(encoding="utf-8") as fh:
        splits_cfg = yaml.safe_load(fh)
    df = load_joined(data_dir / "train_transaction.csv", data_dir / "train_identity.csv")
    train = df[sim_day(df["TransactionDT"]) < int(splits_cfg["train_days"])]
    train = train.sample(n=min(sample_rows, len(train)), random_state=42)

    X = bundle.pipeline.transform(train)
    scores = bundle.model.booster_.predict(X)
    booster = bundle.model.booster_
    importance = booster.feature_importance(importance_type="gain")
    ranked = [
        name
        for _, name in sorted(zip(importance, booster.feature_name(), strict=True), reverse=True)
    ][:top_k]

    feature_reference: dict[str, np.ndarray] = {}
    for name in ranked:
        values = X[name].to_numpy()
        if str(X[name].dtype) == "category":
            values = np.asarray([None if pd.isna(v) else str(v) for v in values], dtype=object)
        else:
            values = np.asarray(values, dtype="float64")
        feature_reference[name] = values

    metrics = compute_metrics(
        train["isFraud"].to_numpy(),
        train["TransactionAmt"].to_numpy(dtype="float64"),
        scores,
        bundle.threshold,
        bundle.metadata.get("false_alert_cost", 5.0),
    )
    return Reference(
        champion_version=version,
        feature_names=ranked,
        feature_reference=feature_reference,
        score_reference=scores.astype("float64"),
        pr_auc=metrics["pr_auc"],
        cost_per_100k=metrics["cost_per_100k"],
    )


def fetch_window(conn, window_sim_days: int, label_delay_days: int = 7) -> dict:
    """Predictions, stored features and released labels of the last window.

    The labelled-performance lookback reaches ``window + label_delay`` days
    back: a prediction becomes labelable exactly ``label_delay`` days after it
    was scored, so with a 7-day window and a 7-day delay the trailing window
    alone is NEVER fully labelled (its earliest row is the only one whose
    label exists) — performance monitoring silently got an empty set until
    this lookback accounted for the delay.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COALESCE(MAX(sim_ts), 0) FROM predictions
            """
        )
        max_dt = cur.fetchone()[0]
        if not max_dt:
            return {}
        window_start = max_dt - window_sim_days * 86_400
        cur.execute(
            """
            SELECT transaction_id, fraud_probability, decision, sim_ts, model_version
            FROM predictions
            WHERE sim_ts IS NOT NULL AND sim_ts >= %s
            """,
            (window_start,),
        )
        preds = cur.fetchall()
        cur.execute(
            """
            SELECT pf.feature, pf.value_num, pf.value_text, pf.sim_ts
            FROM prediction_features pf
            WHERE pf.sim_ts >= %s
            """,
            (window_start,),
        )
        feats = cur.fetchall()
        cur.execute(
            """
            SELECT p.fraud_probability, p.decision, p.threshold, l.is_fraud, l.amount
            FROM predictions p
            JOIN labels l ON l.transaction_id = p.transaction_id
            WHERE p.sim_ts IS NOT NULL AND p.sim_ts >= %s
              AND l.available_at_dt <= %s
            """,
            # the most recent predictions whose labels can exist: the window,
            # shifted back by the label delay (the available_at filter below
            # still decides which are actually released)
            (max_dt - (window_sim_days + label_delay_days) * 86_400, max_dt),
        )
        labelled = cur.fetchall()
    return {
        "max_dt": max_dt,
        "predictions": preds,
        "features": feats,
        "labelled": labelled,
    }


def evaluate_window(
    window: dict, reference: Reference, cfg: dict
) -> tuple[list[DriftResult], DriftResult, PerformanceSnapshot | None]:
    """Feature drifts + score drift + performance snapshot for one window."""
    psi_cfg = cfg["psi"]
    feature_results: list[DriftResult] = []
    by_feature: dict[str, list[tuple]] = {}
    for feature, value_num, value_text, _ in window["features"]:
        by_feature.setdefault(feature, []).append((value_num, value_text))
    for name in reference.feature_names:
        rows = by_feature.get(name, [])
        if len(rows) < cfg["min_window_predictions"]:
            continue
        numeric = [r[0] for r in rows if r[0] is not None]
        texts = [r[1] for r in rows if r[1] is not None]
        ref = reference.feature_reference[name]
        if ref.dtype == object:
            actual = np.asarray(texts, dtype=object)
        else:
            actual = np.asarray(numeric, dtype="float64")
        if actual.size == 0:
            continue
        feature_results.append(
            evaluate_drift(
                name,
                ref,
                actual,
                warning=psi_cfg["warning"],
                alert=psi_cfg["alert"],
                bins=psi_cfg["bins"],
            )
        )

    scores = np.asarray([p[1] for p in window["predictions"]], dtype="float64")
    score_result = evaluate_drift(
        "score",
        reference.score_reference,
        scores,
        warning=psi_cfg["warning"],
        alert=psi_cfg["alert"],
        bins=psi_cfg["bins"],
    )

    perf = None
    labelled = window["labelled"]
    if labelled:
        y = np.array([row[3] for row in labelled], dtype=int)
        p = np.array([row[0] for row in labelled], dtype="float64")
        decision = np.array([row[1] for row in labelled], dtype=object)
        from fraudops.models.cost import cost_per_100k

        cost = cost_per_100k(
            y,
            np.array([row[4] for row in labelled], dtype="float64"),
            decision == "alert",
            5.0,
        )
        pr_auc = float(average_precision_score(y, p)) if y.sum() > 0 else None
        perf = PerformanceSnapshot(pr_auc=pr_auc, cost_per_100k=cost, n_labelled=len(labelled))
    return feature_results, score_result, perf


def persist_results(
    conn,
    sim_day_value: int,
    feature_results: list[DriftResult],
    score_result: DriftResult,
    perf: PerformanceSnapshot | None,
) -> list[tuple[str, str, str, str]]:
    """Write monitoring_results + drift_alerts; return the alert tuples."""
    alerts: list[tuple[str, str, str, str]] = []
    rows = []
    for result in feature_results + [score_result]:
        rows.append(
            (
                sim_day_value,
                "feature" if result.name != "score" else "score",
                result.name,
                result.psi,
                result.ks_pvalue,
                result.level.value,
            )
        )
        if result.level in (Level.WARNING, Level.ALERT):
            alerts.append(
                (
                    result.name,
                    result.level.value,
                    f"{result.name} PSI {result.psi:.3f} ({result.level.value})",
                )
            )
    if perf is not None:
        rows.append(
            (
                sim_day_value,
                "performance",
                "labelled_window",
                None,
                None,
                "ok",
                # values live in details: the retrain DAG re-derives the trigger
                # from these persisted rows when it wakes up (psycopg needs an
                # explicit Json adapter for the JSONB column)
                Json(
                    {
                        "pr_auc": perf.pr_auc,
                        "cost_per_100k": perf.cost_per_100k,
                        "n_labelled": perf.n_labelled,
                    }
                ),
            )
        )
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO monitoring_results
                (sim_day, kind, name, psi, ks_pvalue, level, details)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            [r + (None,) if len(r) == 6 else r for r in rows],
        )
        for name, level, message in alerts:
            cur.execute(
                """
                INSERT INTO drift_alerts (kind, name, level, message)
                VALUES (%s, %s, %s, %s)
                """,
                ("drift", name, level, message),
            )
    conn.commit()
    return alerts


def _metrics_registry():
    from prometheus_client import CollectorRegistry, Gauge

    registry = CollectorRegistry()
    gauges = {
        "psi": Gauge(
            "fraudops_drift_psi",
            "PSI per monitored feature/score",
            labelnames=["name"],
            registry=registry,
        ),
        "level": Gauge(
            "fraudops_drift_level",
            "0 ok / 1 warning / 2 alert",
            labelnames=["name"],
            registry=registry,
        ),
        "pr_auc": Gauge("fraudops_pr_auc", "PR-AUC on the labelled window", registry=registry),
        "cost": Gauge(
            "fraudops_cost_per_100k",
            "Expected cost per 100k (window)",
            registry=registry,
        ),
        "champion": Gauge("fraudops_champion_version", "Champion model version", registry=registry),
        "trigger": Gauge(
            "fraudops_retrain_trigger",
            "1 when the retrain trigger fires",
            registry=registry,
        ),
    }
    return registry, gauges


def run_monitor(
    database_dsn: str,
    tracking_uri: str,
    model_name: str,
    data_dir: Path,
    configs_dir: Path,
    metrics_port: int = 9101,
    cycle_seconds: float = 15.0,
) -> None:
    cfg = load_config(configs_dir / "drift.yaml")
    trigger_cfg = TriggerConfig(**cfg.get("trigger", {}))

    from prometheus_client import start_http_server

    registry, gauges = _metrics_registry()
    start_http_server(metrics_port, registry=registry)

    holder = ModelHolder(tracking_uri=tracking_uri, model_name=model_name)
    conn: psycopg.Connection | None = None

    reference: Reference | None = None
    while True:
        try:
            if conn is None or conn.closed:
                conn = psycopg.connect(database_dsn)
                ensure_schema(conn)
            # first pass loads the champion; later passes follow promotions
            # (a new champion means the reference below is rebuilt too)
            holder.reload_if_changed()
            if holder.last_error:
                logger.warning("champion load failed: %s", holder.last_error)
            loaded = holder.require()
            if reference is None or reference.champion_version != loaded.version:
                logger.info("building drift reference for champion v%s ...", loaded.version)
                reference = build_reference(
                    loaded.bundle,
                    data_dir,
                    configs_dir,
                    int(cfg["top_k_features"]),
                    loaded.version,
                )
                logger.info(
                    "reference ready: %d features, train pr_auc=%.3f",
                    len(reference.feature_names),
                    reference.pr_auc,
                )
            gauges["champion"].set(loaded.version)

            window = fetch_window(
                conn,
                int(cfg["window_sim_days"]),
                label_delay_days=int(cfg.get("label_delay_days", 7)),
            )
            # close the read transaction NOW: an idle-in-transaction reader
            # holds an AccessShare lock that blocks every other service's
            # ensure_schema (AccessExclusive) for as long as we sleep
            conn.commit()
            if not window or len(window["predictions"]) < cfg["min_window_predictions"]:
                logger.info(
                    "waiting for predictions (window has %s)", len(window.get("predictions", []))
                )
                time.sleep(cycle_seconds)
                continue

            feature_results, score_result, perf = evaluate_window(window, reference, cfg)
            sim_day_value = int(window["max_dt"] // 86_400)
            alerts = persist_results(conn, sim_day_value, feature_results, score_result, perf)

            for result in feature_results + [score_result]:
                gauges["psi"].labels(name=result.name).set(result.psi)
                gauges["level"].labels(name=result.name).set(
                    {"ok": 0, "warning": 1, "alert": 2}[result.level.value]
                )
            if perf is not None:
                if perf.pr_auc is not None:
                    gauges["pr_auc"].set(perf.pr_auc)
                gauges["cost"].set(perf.cost_per_100k)

            decision = should_retrain(
                score_result,
                feature_results,
                perf,
                reference_pr_auc=reference.pr_auc,
                reference_cost_per_100k=reference.cost_per_100k,
                config=trigger_cfg,
            )
            gauges["trigger"].set(1 if decision.retrain else 0)
            status = (
                f"day {sim_day_value}: score PSI {score_result.psi:.3f}"
                f" [{score_result.level.value}],"
                f" {sum(1 for r in feature_results if r.level is Level.ALERT)}"
                f" feature alerts" + (f", pr_auc {perf.pr_auc:.3f}" if perf and perf.pr_auc else "")
            )
            if decision.retrain:
                logger.warning("RETRAIN TRIGGERED: %s | %s", "; ".join(decision.reasons), status)
            elif alerts:
                logger.warning("drift alerts: %s | %s", "; ".join(a[2] for a in alerts), status)
            else:
                logger.info(status)
        except psycopg.OperationalError as exc:
            conn = None  # postgres restarted — reconnect next cycle
            logger.warning("postgres unavailable, retrying: %s", exc)
        except Exception:  # noqa: BLE001 — the monitor keeps running
            logger.exception("monitor cycle failed")
            if conn is not None and not conn.closed:
                try:
                    conn.rollback()  # clear an aborted transaction, if any
                except psycopg.Error:
                    conn = None
        time.sleep(cycle_seconds)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database-url",
        default=os.environ.get(
            "DATABASE_URL", "postgresql://fraudops:fraudops-local@localhost:5432/fraudops"
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
    parser.add_argument("--metrics-port", type=int, default=9101)
    parser.add_argument("--cycle-seconds", type=float, default=15.0)
    args = parser.parse_args()
    run_monitor(
        database_dsn=args.database_url,
        tracking_uri=args.tracking_uri,
        model_name=args.model_name,
        data_dir=args.data_dir,
        configs_dir=args.configs_dir,
        metrics_port=args.metrics_port,
        cycle_seconds=args.cycle_seconds,
    )


if __name__ == "__main__":
    main()

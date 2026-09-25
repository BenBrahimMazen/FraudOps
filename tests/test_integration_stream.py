"""Integration: a tiny replay through producer -> kafka -> consumer -> storage.

Requires the live full stack (Kafka + Postgres + MLflow with a champion);
skipped cleanly everywhere else (CI, bare clones). Run locally with:

    docker compose --profile full up -d && pytest tests/test_integration_stream.py
"""

from __future__ import annotations

import socket
from pathlib import Path

import psycopg
import pytest

KAFKA = ("localhost", 9092)
DSN = "postgresql://fraudops:fraudops-local@localhost:5432/fraudops"
MLFLOW = "http://localhost:5000"


def _reachable(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _champion_exists() -> bool:
    import os

    os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
    from fraudops.registry import client as rc

    cli = rc.configure(MLFLOW)
    return rc.get_version_by_alias(cli, "champion") is not None


pytestmark = pytest.mark.skipif(
    not (_reachable(*KAFKA) and _reachable("localhost", 5432) and _reachable("localhost", 5000)),
    reason="full stack not running (kafka/postgres/mlflow)",
)


def test_tiny_replay_flows_through_scoring_and_storage() -> None:
    if not _champion_exists():
        pytest.skip("no champion registered in the stack's mlflow")

    from fraudops.storage import ensure_schema
    from fraudops.streaming.consumer import run_consumer
    from fraudops.streaming.replay import run_replay

    conn = psycopg.connect(DSN)
    ensure_schema(conn)
    with conn.cursor() as cur:
        cur.execute("SELECT COALESCE(MAX(id), 0) FROM predictions")
        baseline_predictions = cur.fetchone()[0]

    summary = run_replay(
        bootstrap_servers="localhost:9092",
        data_dir=Path("data/raw"),
        configs_dir=Path("configs"),
        day_seconds=0.001,  # as fast as possible
        max_events=300,
        database_dsn=DSN,
        quiet=True,
    )
    assert summary["events"] == 300

    result = run_consumer(
        bootstrap_servers="localhost:9092",
        tracking_uri=MLFLOW,
        model_name="fraudops-lightgbm",
        database_dsn=DSN,
        stop_after_messages=300,
    )
    assert result["scored"] >= 300

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM predictions WHERE id > %s", (baseline_predictions,))
        stored = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM prediction_features")
        features = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM labels_pending")
        pending = cur.fetchone()[0]
        cur.execute("SELECT transaction_dt FROM sim_clock WHERE id = 1")
        clock = cur.fetchone()
    conn.close()

    assert stored >= 300
    assert features >= 300 * 20  # top-k features per scored transaction
    assert pending >= 300  # labels staged with availability in the future
    assert clock is not None and clock[0] > 0  # the sim clock advanced

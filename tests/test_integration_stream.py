"""Integration: a tiny replay through producer -> kafka -> consumer -> storage.

Requires the live full stack (Kafka + Postgres + MLflow with a champion);
skipped cleanly everywhere else (CI, bare clones). Run locally with:

    docker compose --profile full up -d && pytest tests/test_integration_stream.py
"""

from __future__ import annotations

import socket
from pathlib import Path
from uuid import uuid4

import psycopg
import pytest

KAFKA = ("localhost", 9092)
# host port 5433 (compose publishes postgres there; a native PostgreSQL often
# already owns 5432 on dev machines)
DSN = "postgresql://fraudops:fraudops-local@localhost:5433/fraudops"
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
    not (_reachable(*KAFKA) and _reachable("localhost", 5433) and _reachable("localhost", 5000)),
    reason="full stack not running (kafka/postgres/mlflow)",
)


def test_tiny_replay_flows_through_scoring_and_storage(monkeypatch) -> None:
    # loading the champion pulls its artifact from MinIO (S3 protocol): the
    # host shell does not source .env (compose-only), and conftest's autouse
    # _clean_env scrub deletes the endpoint var — so set the whole local
    # stack env explicitly for this test
    monkeypatch.setenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "fraudops")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "fraudops-local")

    if not _champion_exists():
        pytest.skip("no champion registered in the stack's mlflow")

    from fraudops.storage import ensure_schema
    from fraudops.streaming.consumer import run_consumer
    from fraudops.streaming.replay import run_replay

    # isolate from any live replay: a shared topic/group once let a test
    # consumer eat live offsets (messages committed but never stored)
    run_id = uuid4().hex[:8]
    topic = f"transactions-test-{run_id}"
    group = f"fraudops-test-{run_id}"

    conn = psycopg.connect(DSN)
    ensure_schema(conn)
    with conn.cursor() as cur:
        cur.execute("SELECT COALESCE(MAX(id), 0) FROM predictions")
        baseline_predictions = cur.fetchone()[0]
        cur.execute("SELECT (SELECT COUNT(*) FROM labels_pending) + (SELECT COUNT(*) FROM labels)")
        baseline_labels = cur.fetchone()[0]
    # close the read transaction: run_replay opens its OWN connection and runs
    # ensure_schema, whose DDL would otherwise queue behind this idle reader
    conn.commit()

    def _cleanup_db() -> None:
        # remove everything this test wrote: the same 300 transaction ids are
        # replayed on every run, and label staging is ON CONFLICT DO NOTHING,
        # so leftovers would make the next run stage nothing and fail
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM prediction_features
                WHERE transaction_id IN
                    (SELECT transaction_id FROM predictions WHERE id > %s)
                """,
                (baseline_predictions,),
            )
            cur.execute(
                """
                DELETE FROM labels_pending
                WHERE transaction_id IN
                    (SELECT transaction_id FROM predictions WHERE id > %s)
                """,
                (baseline_predictions,),
            )
            cur.execute(
                """
                DELETE FROM labels
                WHERE transaction_id IN
                    (SELECT transaction_id FROM predictions WHERE id > %s)
                """,
                (baseline_predictions,),
            )
            cur.execute("DELETE FROM predictions WHERE id > %s", (baseline_predictions,))
        conn.commit()

    try:
        summary = run_replay(
            bootstrap_servers="localhost:9092",
            data_dir=Path("data/raw"),
            configs_dir=Path("configs"),
            day_seconds=0.001,  # as fast as possible
            max_events=300,
            database_dsn=DSN,
            topic=topic,
            quiet=True,
        )
        assert summary["events"] == 300

        result = run_consumer(
            bootstrap_servers="localhost:9092",
            tracking_uri=MLFLOW,
            model_name="fraudops-lightgbm",
            database_dsn=DSN,
            stop_after_messages=300,
            group_id=group,
            topic=topic,
        )
        assert result["scored"] >= 300

        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM predictions WHERE id > %s", (baseline_predictions,))
            stored = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM prediction_features")
            features = cur.fetchone()[0]
            # labels race the live label-release job: with the sim clock already
            # past their availability they move to `labels` immediately, so count
            # both tables against the pre-test baseline
            cur.execute(
                "SELECT (SELECT COUNT(*) FROM labels_pending) + (SELECT COUNT(*) FROM labels)"
            )
            labels_total = cur.fetchone()[0]
            cur.execute("SELECT transaction_dt FROM sim_clock WHERE id = 1")
            clock = cur.fetchone()

        assert stored >= 300
        assert features >= 300 * 20  # top-k features per scored transaction
        assert labels_total - baseline_labels >= 300  # every replayed label staged
        assert clock is not None and clock[0] > 0  # the sim clock advanced
    finally:
        _cleanup_db()
        conn.close()
        # best-effort teardown: remove the test topic so repeated runs stay isolated
        from confluent_kafka.admin import AdminClient

        try:
            for _, fut in (
                AdminClient({"bootstrap.servers": "localhost:9092"}).delete_topics([topic]).items()
            ):
                fut.result(timeout=10)
        except Exception as exc:  # noqa: BLE001 — leftover test topics are harmless
            print(f"warning: could not delete test topic {topic}: {exc}")

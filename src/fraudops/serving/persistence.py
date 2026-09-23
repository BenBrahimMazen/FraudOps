"""Buffered, non-blocking persistence of scored transactions to Postgres.

/scscore latency must not include a database round-trip: handlers enqueue,
a background worker batches inserts. Bounded queue — under sustained DB
outage the service keeps scoring and drops (counted), never blocks.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass

import psycopg

logger = logging.getLogger("fraudops.persistence")

DDL = """
CREATE TABLE IF NOT EXISTS predictions (
    id                 BIGSERIAL PRIMARY KEY,
    transaction_id     BIGINT,
    scored_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    model_version      INTEGER  NOT NULL,
    fraud_probability  DOUBLE PRECISION NOT NULL,
    decision           TEXT     NOT NULL,
    threshold          DOUBLE PRECISION NOT NULL,
    latency_ms         DOUBLE PRECISION
);
CREATE INDEX IF NOT EXISTS predictions_scored_at_idx ON predictions (scored_at);
"""

_INSERT = (
    "INSERT INTO predictions (transaction_id, model_version, fraud_probability,"
    " decision, threshold, latency_ms) VALUES (%s, %s, %s, %s, %s, %s)"
)


@dataclass(frozen=True)
class Prediction:
    transaction_id: int | None
    model_version: int
    fraud_probability: float
    decision: str
    threshold: float
    latency_ms: float
    scored_at: float  # time.time() at enqueue


class PredictionSink:
    """Queue + background writer. Dropping beats blocking the scorer."""

    def __init__(
        self,
        dsn: str | None,
        queue_size: int = 10_000,
        flush_interval: float = 2.0,
        batch_size: int = 200,
    ) -> None:
        self.dsn = dsn
        self.queue: queue.Queue[Prediction] = queue.Queue(maxsize=queue_size)
        self.flush_interval = flush_interval
        self.batch_size = batch_size
        self.persisted = 0
        self.dropped = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._conn: psycopg.Connection | None = None

    # ------------------------------------------------------------------ api
    @property
    def enabled(self) -> bool:
        return self.dsn is not None

    def submit(self, prediction: Prediction) -> None:
        try:
            self.queue.put_nowait(prediction)
        except queue.Full:
            self.dropped += 1

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="prediction-writer", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def flush_now(self, timeout: float = 3.0) -> None:
        """Best-effort drain (used by tests)."""
        deadline = time.monotonic() + timeout
        while not self.queue.empty() and time.monotonic() < deadline:
            self._drain_once()

    # -------------------------------------------------------------- worker
    def _run(self) -> None:
        while not self._stop.wait(self.flush_interval):
            self._drain_once()
        self._drain_once()  # final drain on shutdown

    def _drain_once(self) -> None:
        if self.queue.empty():
            return
        batch: list[Prediction] = []
        while len(batch) < self.batch_size and not self.queue.empty():
            try:
                batch.append(self.queue.get_nowait())
            except queue.Empty:
                break
        if not batch:
            return
        try:
            self._ensure_connection()
            assert self._conn is not None
            with self._conn.cursor() as cur:
                cur.executemany(
                    _INSERT,
                    [
                        (
                            p.transaction_id,
                            p.model_version,
                            p.fraud_probability,
                            p.decision,
                            p.threshold,
                            p.latency_ms,
                        )
                        for p in batch
                    ],
                )
            self._conn.commit()
            self.persisted += len(batch)
        except Exception as exc:  # noqa: BLE001 — persistence must not kill scoring
            logger.warning("prediction batch dropped (%d rows): %s", len(batch), exc)
            self.dropped += len(batch)
            self._conn = None

    def _ensure_connection(self) -> None:
        if self._conn is None or self._conn.closed:
            self._conn = psycopg.connect(self.dsn, autocommit=False)
            self._conn.execute(DDL)
            self._conn.commit()

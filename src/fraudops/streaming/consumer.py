"""Scoring consumer: Kafka transactions -> champion model -> predictions.

Shares the exact serving library (FeaturePipeline + champion bundle via the
ModelHolder), scores micro-batches, persists predictions (with sim_ts) and
the top-K model features per transaction (drift input), and advances the
simulated clock. Hot-reload works the same as in the API: the holder's
poller swaps the champion without stopping consumption.
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
import time

import numpy as np
import pandas as pd

from fraudops.serving.model_loader import ModelHolder
from fraudops.storage import ensure_schema

logger = logging.getLogger("fraudops.consumer")
KAFKA_TOPIC = "transactions"


class BatchWriter:
    """Buffered inserts for predictions + features + clock updates."""

    def __init__(self, dsn: str, flush_rows: int = 200, flush_seconds: float = 2.0) -> None:
        import psycopg

        self._connect = lambda: psycopg.connect(dsn)
        self.conn = None
        self.flush_rows = flush_rows
        self.flush_seconds = flush_seconds
        self.predictions: list[tuple] = []
        self.features: list[tuple] = []
        self._last_flush = time.monotonic()
        # flushes also run on the watchdog thread, so serialise them
        self._flush_lock = threading.Lock()

    def _ensure(self):
        if self.conn is None or self.conn.closed:
            self.conn = self._connect()
            ensure_schema(self.conn)
        return self.conn

    def add_prediction(self, row: tuple) -> None:
        self.predictions.append(row)

    def add_features(self, rows: list[tuple]) -> None:
        self.features.extend(rows)

    def maybe_flush(self, force: bool = False) -> None:
        with self._flush_lock:
            due = (
                force
                or len(self.predictions) >= self.flush_rows
                or (self.predictions and time.monotonic() - self._last_flush >= self.flush_seconds)
            )
            if not due:
                return
            conn = self._ensure()
            with conn.cursor() as cur:
                if self.predictions:
                    cur.executemany(
                        """
                        INSERT INTO predictions
                            (transaction_id, model_version, fraud_probability,
                             decision, threshold, latency_ms, sim_ts)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """,
                        self.predictions,
                    )
                if self.features:
                    cur.executemany(
                        """
                        INSERT INTO prediction_features
                            (transaction_id, sim_ts, feature, value_num, value_text)
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        self.features,
                    )
                max_dt = max((p[6] for p in self.predictions if p[6] is not None), default=None)
                if max_dt is not None:
                    cur.execute(
                        """
                        INSERT INTO sim_clock (id, transaction_dt, updated_at)
                        VALUES (1, %s, now())
                        ON CONFLICT (id) DO UPDATE
                        SET transaction_dt = GREATEST(sim_clock.transaction_dt,
                                                       EXCLUDED.transaction_dt),
                            updated_at = now()
                        """,
                        (int(max_dt),),
                    )
            conn.commit()
            self.predictions.clear()
            self.features.clear()
            self._last_flush = time.monotonic()

    def close(self) -> None:
        self.maybe_flush(force=True)
        if self.conn is not None:
            self.conn.close()


def top_feature_names(bundle, k: int) -> list[str]:
    """Top-k features by LightGBM gain importance, restricted to columns the
    pipeline emits (numeric features + categorical ones we can read back)."""
    booster = bundle.model.booster_
    importance = booster.feature_importance(importance_type="gain")
    names = booster.feature_name()
    ranked = [name for _, name in sorted(zip(importance, names, strict=True), reverse=True)]
    return ranked[:k]


def feature_row_values(X: pd.DataFrame, names: list[str], transaction_id, sim_ts):
    """(feature, numeric-or-text value) tuples for one transformed row."""
    row = X.iloc[0]
    for name in names:
        value = row[name]
        if value is None or (isinstance(value, float) and np.isnan(value)):
            yield (transaction_id, sim_ts, name, None, None)
        elif isinstance(value, np.floating | float | np.integer | int):
            yield (transaction_id, sim_ts, name, float(value), None)
        else:
            yield (transaction_id, sim_ts, name, None, str(value))


def run_consumer(
    bootstrap_servers: str,
    tracking_uri: str,
    model_name: str,
    database_dsn: str,
    top_k: int = 20,
    batch_max: int = 100,
    batch_seconds: float = 1.0,
    poll_seconds: float = 30.0,
    stop_after_messages: int | None = None,
    group_id: str = "fraudops-scorer",
    topic: str = KAFKA_TOPIC,
) -> dict:
    from confluent_kafka import Consumer

    holder = ModelHolder(
        tracking_uri=tracking_uri, model_name=model_name, poll_seconds=poll_seconds
    )
    loaded = holder.load()
    logger.info("champion v%s loaded (threshold %.4f)", loaded.version, loaded.bundle.threshold)

    writer = BatchWriter(database_dsn)

    # DB durability must not depend on the main loop's health: poll(0.2) once
    # blocked for 8+ minutes on a librdkafka-internal futex (SIGINT traceback
    # pinned it at the poll call), and the last buffered predictions reached
    # Postgres only thanks to the shutdown force-flush. A daemon thread keeps
    # flushing whatever has been scored, every few seconds, regardless.
    flush_stop = threading.Event()

    def _flush_watchdog() -> None:
        while not flush_stop.wait(5.0):
            try:
                writer.maybe_flush()
            except Exception:  # noqa: BLE001 — never let the watchdog die
                logger.exception("watchdog flush failed")

    flusher = threading.Thread(target=_flush_watchdog, name="flush-watchdog", daemon=True)
    flusher.start()

    consumer = Consumer(
        {
            "bootstrap.servers": bootstrap_servers,
            "group.id": group_id,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": True,
        }
    )
    # subscribe to the GIVEN topic: the live consumer and tests must be able
    # to point at different topics — a test group with earliest reset once
    # re-consumed the whole live topic and duplicated its predictions
    consumer.subscribe([topic])
    holder.start_polling()

    buffer: list[dict] = []
    last_batch = time.monotonic()
    total = 0
    failed_batches = 0
    idle_cycles = 0

    def _drain(buf: list[dict]) -> None:
        """Score one buffer; a poison batch is logged and skipped, never fatal.

        One malformed event must not wedge the consumer forever: a crash here
        would restart into the same offset and die in a loop. Drops are
        counted, like the API's prediction sink.
        """
        nonlocal total, failed_batches
        try:
            _score_batch(buf, holder, writer, top_k)
            total += len(buf)
        except Exception:  # noqa: BLE001 — keep consuming after bad input
            failed_batches += 1
            ids = [e.get("TransactionID") for e in buf[:5]]
            logger.exception(
                "poison batch skipped (%d events, first ids %s) — batch #%d dropped",
                len(buf),
                ids,
                failed_batches,
            )

    try:
        while True:
            message = consumer.poll(0.2)
            if message is None:
                idle_cycles += 1
                if buffer and time.monotonic() - last_batch >= batch_seconds:
                    _drain(buffer)
                    buffer.clear()
                    last_batch = time.monotonic()
                # a partial tail must not wait for the next message (that may
                # be hours away on a quiet topic) — flush it once we're idle
                writer.maybe_flush()
                # integration mode: stop once the expected batch arrived
                if (
                    stop_after_messages is not None
                    and total >= stop_after_messages
                    and idle_cycles > 5
                ):
                    break
                continue
            idle_cycles = 0
            if message.error():
                logger.warning("kafka: %s", message.error())
                continue
            event = json.loads(message.value())
            buffer.append(event)
            if len(buffer) >= batch_max:
                _drain(buffer)
                buffer.clear()
                last_batch = time.monotonic()
                if total % 5000 < batch_max:
                    logger.info("scored %s events", f"{total:,}")
            writer.maybe_flush()
    finally:
        if buffer:
            _drain(buffer)
        flush_stop.set()
        flusher.join(timeout=10)
        writer.close()
        consumer.close()
        holder.stop_polling()
    return {"scored": total, "failed_batches": failed_batches}


def _score_batch(events: list[dict], holder, writer: BatchWriter, top_k: int) -> None:
    loaded = holder.require()
    frame = pd.DataFrame(events)
    t0 = time.perf_counter()
    X = loaded.bundle.pipeline.transform(frame)
    scores = loaded.bundle.model.booster_.predict(X)
    latency_ms = (time.perf_counter() - t0) / max(len(events), 1) * 1000

    names = top_feature_names(loaded.bundle, top_k)
    for i, event in enumerate(events):
        transaction_id = event.get("TransactionID")
        sim_ts = event.get("TransactionDT")
        score = float(scores[i])
        decision = "alert" if score >= loaded.bundle.threshold else "approve"
        writer.add_prediction(
            (
                transaction_id,
                loaded.version,
                score,
                decision,
                loaded.bundle.threshold,
                latency_ms,
                sim_ts,
            )
        )
        writer.add_features(list(feature_row_values(X.iloc[[i]], names, transaction_id, sim_ts)))
    writer.maybe_flush()


def main() -> None:
    import os

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bootstrap-servers", default=os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    )
    parser.add_argument(
        "--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    )
    parser.add_argument(
        "--model-name", default=os.environ.get("FRAUDOPS_MODEL_NAME", "fraudops-lightgbm")
    )
    parser.add_argument(
        "--database-url",
        default=os.environ.get(
            "DATABASE_URL",
            "postgresql://fraudops:fraudops-local@localhost:5432/fraudops",
        ),
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--group-id", default="fraudops-scorer")
    parser.add_argument("--topic", default=KAFKA_TOPIC)
    args = parser.parse_args()
    run_consumer(
        bootstrap_servers=args.bootstrap_servers,
        tracking_uri=args.tracking_uri,
        model_name=args.model_name,
        database_dsn=args.database_url,
        top_k=args.top_k,
        group_id=args.group_id,
        topic=args.topic,
    )


if __name__ == "__main__":
    main()

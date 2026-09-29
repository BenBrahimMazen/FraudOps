"""Replay producer: publish the stream window to Kafka in TransactionDT order.

Speed: ``--day-seconds`` real seconds per simulated day (default 30: the
~33-day stream window replays in about 16 minutes). ``--max-events`` caps
the run for fast tests. Labels are never published on the topic: they are staged into
``labels_pending`` with their simulated availability time, and the label
release job makes them visible only after the configured delay.

Drift injection: the producer periodically reads ``drift_injections`` from
Postgres and applies active injections (amount multiplier, feature nulling)
to events whose simulated day has passed ``from_sim_day`` — so shifts can be
injected into a live replay (``make drift``) without restarting it. Runs
without Postgres too (no injections then).
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import yaml

from fraudops.data.clock import sim_day
from fraudops.data.loader import load_joined
from fraudops.storage import ensure_schema

if TYPE_CHECKING:
    from psycopg import Connection

KAFKA_TOPIC = "transactions"
SECONDS_PER_DAY = 86_400


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


class Injector:
    """Reads active injections from Postgres; no-ops when Postgres is absent."""

    def __init__(self, dsn: str | None, poll_seconds: float = 5.0) -> None:
        self.dsn = dsn
        self.poll_seconds = poll_seconds
        self._conn: Connection | None = None
        self._next_poll = 0.0
        self._injections: list[dict] = []

    def _refresh(self) -> None:
        if not self.dsn:
            return  # no Postgres configured: injections stay off
        try:
            import psycopg

            conn = self._conn
            if conn is None or conn.closed:
                conn = psycopg.connect(self.dsn, autocommit=True)
                ensure_schema(conn)
                self._conn = conn
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT kind, params, from_sim_day FROM drift_injections"
                    " WHERE active ORDER BY id"
                )
                self._injections = [
                    {"kind": kind, "params": params, "from_day": from_day}
                    for kind, params, from_day in cur.fetchall()
                ]
        except Exception:  # noqa: BLE001 — replay must not die on db issues
            self._conn = None
            self._injections = []

    def apply(self, event: dict, current_day: int) -> dict:
        """Return the (possibly shifted) event for the current sim day."""
        if time.monotonic() >= self._next_poll:
            self._refresh()
            self._next_poll = time.monotonic() + self.poll_seconds
        for injection in self._injections:
            if current_day < injection["from_day"]:
                continue
            if injection["kind"] == "amount_factor":
                factor = float(injection["params"].get("factor", 1.0))
                amount = event.get("TransactionAmt")
                if amount is not None:
                    event = {**event, "TransactionAmt": amount * factor}
            elif injection["kind"] == "nullify":
                columns = injection["params"].get("columns", [])
                event = {k: (None if k in columns else v) for k, v in event.items()}
        return event


def _row_to_event(row: pd.Series) -> dict:
    """JSON-native event dict from a source row.

    pandas 3 yields plain Python scalars from ``iterrows``/``items`` while
    pandas 2 yields numpy scalars — accept both, and never let a number fall
    through to ``str()`` (the consumer's feature pipeline needs real numerics;
    only the API's Pydantic layer would silently coerce strings back).
    """
    out: dict[str, object] = {}
    for key, value in row.items():
        if isinstance(value, bool | np.bool_):
            out[key] = bool(value)
        elif pd.isna(value):
            out[key] = None
        elif isinstance(value, int | np.integer):
            out[key] = int(value)
        elif isinstance(value, float | np.floating):
            out[key] = float(value)
        else:
            out[key] = str(value)
    return out


def run_replay(
    bootstrap_servers: str,
    data_dir: Path = Path("data/raw"),
    configs_dir: Path = Path("configs"),
    day_seconds: float = 30.0,
    max_events: int | None = None,
    database_dsn: str | None = None,
    topic: str = KAFKA_TOPIC,
    quiet: bool = False,
) -> dict:
    from confluent_kafka import Producer

    splits_cfg = load_config(configs_dir / "splits.yaml")
    label_delay_days = int(load_config(configs_dir / "drift.yaml").get("label_delay_days", 7))

    df = load_joined(data_dir / "train_transaction.csv", data_dir / "train_identity.csv")
    stream = df[
        sim_day(df["TransactionDT"]) >= int(splits_cfg["train_days"]) + int(splits_cfg["val_days"])
    ]
    stream = stream.sort_values("TransactionDT").reset_index(drop=True)
    if max_events is not None:
        stream = stream.head(max_events)

    injector = Injector(database_dsn)
    pg = None
    if database_dsn:
        import psycopg

        pg = psycopg.connect(database_dsn)
        ensure_schema(pg)

    producer = Producer({"bootstrap.servers": bootstrap_servers})
    speedup = SECONDS_PER_DAY / max(day_seconds, 1e-9)
    delivered = 0
    started = time.perf_counter()

    def _flush_labels(rows: pd.DataFrame) -> None:
        if pg is None:
            return
        avail = available_at_dt_bulk(rows["TransactionDT"], label_delay_days)
        with pg.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO labels_pending (transaction_id, is_fraud, amount, available_at_dt)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (transaction_id) DO NOTHING
                """,
                list(
                    zip(
                        rows["TransactionID"].astype("int64"),
                        rows["isFraud"].astype("int64"),
                        rows["TransactionAmt"].astype("float64"),
                        avail,
                        strict=True,
                    )
                ),
            )
        pg.commit()

    def available_at_dt_bulk(dts: pd.Series, delay_days: int) -> np.ndarray:
        return (dts.to_numpy(dtype="int64") + delay_days * SECONDS_PER_DAY).tolist()

    batch_frames: list[pd.DataFrame] = []
    last_dt: int | None = None
    next_send_at = time.monotonic()

    for _, row in stream.iterrows():
        dt = int(row["TransactionDT"])
        if last_dt is not None:
            sim_gap = dt - last_dt
            wait = sim_gap / speedup
            if wait > 0:
                deadline = next_send_at + wait
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    time.sleep(min(remaining, 0.5))
                next_send_at = deadline
        else:
            next_send_at = time.monotonic()

        event = _row_to_event(row)
        event.pop("isFraud", None)  # labels NEVER travel on the topic
        event = injector.apply(event, int(dt // SECONDS_PER_DAY))
        # NO timestamp arg: Kafka reads it as epoch-ms, but dt is dataset-epoch
        # SECONDS (0..15.8M) — stamped 1970-01-01, every segment looked older
        # than retention and the 5-min sweep deleted them mid-replay, silently
        # skipping offsets under the lagging consumer. Sim time travels in the
        # payload (TransactionDT); broker-side timestamps stay wall-clock.
        producer.produce(
            topic,
            key=str(event["TransactionID"]).encode(),
            value=json.dumps(event).encode(),
        )
        delivered += 1
        batch_frames.append(row)
        if len(batch_frames) >= 500:
            _flush_labels(pd.DataFrame(batch_frames))
            batch_frames = []
        if not quiet and delivered % 2000 == 0:
            producer.poll(0)
            day = dt // SECONDS_PER_DAY
            print(
                f"  {delivered:,} events | sim day {day} | "
                f"{delivered / (time.perf_counter() - started):,.0f} ev/s",
                flush=True,
            )
        if delivered % 1000 == 0:
            producer.poll(0)

    if batch_frames:
        _flush_labels(pd.DataFrame(batch_frames))
    producer.flush(30)
    if pg is not None:
        pg.close()

    elapsed = time.perf_counter() - started
    summary = {
        "events": delivered,
        "elapsed_s": round(elapsed, 1),
        "events_per_s": round(delivered / max(elapsed, 1e-9), 1),
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    print(f"replay done: {json.dumps(summary)}", flush=True)
    return summary


def main() -> None:
    import os

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--configs-dir", type=Path, default=Path("configs"))
    parser.add_argument(
        "--day-seconds", type=float, default=30.0, help="real seconds per simulated day (speed-up)"
    )
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument(
        "--database-url",
        default=os.environ.get("DATABASE_URL"),
        help="Postgres for label staging + drift injections (env: DATABASE_URL); "
        "without it no labels are staged and drift injections are not applied",
    )
    parser.add_argument("--topic", default=KAFKA_TOPIC)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    run_replay(
        bootstrap_servers=args.bootstrap_servers,
        data_dir=args.data_dir,
        configs_dir=args.configs_dir,
        day_seconds=args.day_seconds,
        max_events=args.max_events,
        database_dsn=args.database_url,
        topic=args.topic,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()

"""Delayed labels: availability logic and the release job.

Fraud confirmations arrive late: a transaction's true label becomes visible
only after ``label_delay_days`` simulated days. The replay producer stages
every label in ``labels_pending`` with ``available_at_dt = TransactionDT +
delay``; the release job moves rows whose availability has passed the
simulated clock into ``labels``. Performance monitoring only ever joins
against ``labels`` — that is what makes the delay real for the metrics.

``available_at_dt`` and ``release_due`` are pure and unit-tested against the
simulated clock.
"""

from __future__ import annotations

import argparse
import os
import time

import psycopg

from fraudops.storage import ensure_schema

SECONDS_PER_DAY = 86_400


def available_at_dt(transaction_dt: int, label_delay_days: int) -> int:
    """Simulated timestamp at which the label becomes visible."""
    return int(transaction_dt) + label_delay_days * SECONDS_PER_DAY


def release_due(pending_available_at: int, clock_dt: int) -> bool:
    """True when the simulated clock has passed the label's availability."""
    return pending_available_at <= clock_dt


def release_due_labels(conn: psycopg.Connection) -> int:
    """Move all due pending labels into the labels table; returns count."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO labels (transaction_id, is_fraud, amount, available_at_dt)
            SELECT p.transaction_id, p.is_fraud, p.amount, p.available_at_dt
            FROM labels_pending p
            CROSS JOIN sim_clock c
            WHERE p.available_at_dt <= c.transaction_dt
            ON CONFLICT (transaction_id) DO NOTHING
            """
        )
        moved = cur.rowcount
        cur.execute(
            """
            DELETE FROM labels_pending p
            USING labels l
            WHERE l.transaction_id = p.transaction_id
            """
        )
    conn.commit()
    return moved


def sim_clock(conn: psycopg.Connection) -> int | None:
    with conn.cursor() as cur:
        cur.execute("SELECT transaction_dt FROM sim_clock WHERE id = 1")
        row = cur.fetchone()
    return row[0] if row else None


def advance_clock(conn: psycopg.Connection, transaction_dt: int) -> None:
    """Move the simulated clock forward (never backward)."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO sim_clock (id, transaction_dt, updated_at)
            VALUES (1, %s, now())
            ON CONFLICT (id) DO UPDATE
            SET transaction_dt = GREATEST(sim_clock.transaction_dt, EXCLUDED.transaction_dt),
                updated_at = now()
            """,
            (int(transaction_dt),),
        )
    conn.commit()


def run_job(dsn: str, poll_seconds: float = 10.0) -> None:
    """Label-release service loop."""
    conn: psycopg.Connection | None = None
    while True:
        try:
            if conn is None or conn.closed:
                conn = psycopg.connect(dsn)
                ensure_schema(conn)
            moved = release_due_labels(conn)
            if moved:
                print(f"released {moved} labels at clock {sim_clock(conn)}", flush=True)
        except psycopg.Error as exc:
            # connection loss AND retryable failures (e.g. ensure_schema hitting
            # the lock timeout while another service reads) — drop and retry
            print(f"postgres error, retrying: {exc}", flush=True)
            conn = None
        time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database-url",
        default=os.environ.get(
            "DATABASE_URL", "postgresql://fraudops:fraudops-local@localhost:5432/fraudops"
        ),
    )
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    args = parser.parse_args()
    run_job(args.database_url, args.poll_seconds)


if __name__ == "__main__":
    main()

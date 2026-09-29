"""Postgres schema for the streaming/monitoring tables.

One place that knows every table. Idempotent: safe to run at every service
startup. The predictions table grows a sim_ts column (transaction time in
the simulation) on top of the serving-phase schema.
"""

from __future__ import annotations

import psycopg

SCHEMA = """
-- scored transactions (serving phase + streaming phase columns).
-- The ALTER runs BEFORE the index that uses the column: databases created
-- in the serving phase lack sim_ts, and an index on a missing column would
-- abort this whole script before the migration applies.
CREATE TABLE IF NOT EXISTS predictions (
    id                 BIGSERIAL PRIMARY KEY,
    transaction_id     BIGINT,
    scored_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    model_version      INTEGER  NOT NULL,
    fraud_probability  DOUBLE PRECISION NOT NULL,
    decision           TEXT     NOT NULL,
    threshold          DOUBLE PRECISION NOT NULL,
    latency_ms         DOUBLE PRECISION,
    sim_ts             BIGINT
);
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS sim_ts BIGINT;
CREATE INDEX IF NOT EXISTS predictions_scored_at_idx ON predictions (scored_at);
CREATE INDEX IF NOT EXISTS predictions_sim_ts_idx ON predictions (sim_ts);

-- ground truth, staged by the replay producer with an availability time
CREATE TABLE IF NOT EXISTS labels_pending (
    transaction_id  BIGINT PRIMARY KEY,
    is_fraud        SMALLINT NOT NULL,
    amount          DOUBLE PRECISION NOT NULL,
    available_at_dt BIGINT NOT NULL
);

-- labels the release job has made visible (performance monitoring may only
-- ever join against this table)
CREATE TABLE IF NOT EXISTS labels (
    transaction_id  BIGINT PRIMARY KEY,
    is_fraud        SMALLINT NOT NULL,
    amount          DOUBLE PRECISION NOT NULL,
    available_at_dt BIGINT NOT NULL,
    released_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- the simulated clock advances with the stream (max TransactionDT seen)
CREATE TABLE IF NOT EXISTS sim_clock (
    id             INTEGER PRIMARY KEY CHECK (id = 1),
    transaction_dt BIGINT NOT NULL,
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- drift injections the replay producer applies from a given simulated day
CREATE TABLE IF NOT EXISTS drift_injections (
    id           SERIAL PRIMARY KEY,
    kind         TEXT NOT NULL,
    params       JSONB NOT NULL DEFAULT '{}',
    from_sim_day INTEGER NOT NULL,
    active       BOOLEAN NOT NULL DEFAULT TRUE,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- every monitoring check (feature PSI, score PSI, performance windows)
CREATE TABLE IF NOT EXISTS monitoring_results (
    id               BIGSERIAL PRIMARY KEY,
    checked_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    sim_day          INTEGER,
    kind             TEXT NOT NULL,
    name             TEXT NOT NULL,
    psi              DOUBLE PRECISION,
    ks_pvalue        DOUBLE PRECISION,
    level            TEXT NOT NULL,
    details          JSONB
);

-- alerts raised (warning/alert level results the operator should see)
CREATE TABLE IF NOT EXISTS drift_alerts (
    id         BIGSERIAL PRIMARY KEY,
    raised_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    kind       TEXT NOT NULL,
    name       TEXT NOT NULL,
    level      TEXT NOT NULL,
    message    TEXT NOT NULL
);

-- top-K model-feature values per scored transaction (drift input)
CREATE TABLE IF NOT EXISTS prediction_features (
    id             BIGSERIAL PRIMARY KEY,
    transaction_id BIGINT,
    sim_ts         BIGINT,
    feature        TEXT NOT NULL,
    value_num      DOUBLE PRECISION,
    value_text     TEXT
);
CREATE INDEX IF NOT EXISTS prediction_features_feature_idx
    ON prediction_features (feature, sim_ts);
"""


def ensure_schema(conn: psycopg.Connection) -> None:
    """Create/migrate every table (idempotent).

    Runs under a short ``lock_timeout``: the DDL needs an exclusive lock on
    ``predictions``, and a reader with an open transaction would otherwise
    park every service startup behind it indefinitely. A timeout raises
    (``psycopg.errors.LockNotAvailable``) and the caller's retry loop tries
    again on its next poll.
    """
    with conn.cursor() as cur:
        cur.execute("SET lock_timeout = '10s'")
        cur.execute(SCHEMA)
    conn.commit()

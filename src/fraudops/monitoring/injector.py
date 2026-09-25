"""Drift injector CLI: apply synthetic shifts to a LIVE replay.

Writes an injection row to Postgres; the replay producer picks it up within
a few seconds and shifts events whose simulated day has passed
``from_sim_day``. Two kinds:

    python -m fraudops.monitoring.injector amount-factor --factor 3 --from-day 165
    python -m fraudops.monitoring.injector nullify --columns card4,P_emaildomain --from-day 170
    python -m fraudops.monitoring.injector list
    python -m fraudops.monitoring.injector deactivate --id 1

Detection lag is measurable: compare the injection's from_day with the sim
day of the first monitoring alert that follows it.
"""

from __future__ import annotations

import argparse
import json

import psycopg

from fraudops.storage import ensure_schema


def add_injection(conn, kind: str, params: dict, from_day: int) -> int:
    ensure_schema(conn)
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO drift_injections (kind, params, from_sim_day)
            VALUES (%s, %s, %s)
            RETURNING id
            """,
            (kind, json.dumps(params), from_day),
        )
        injection_id = cur.fetchone()[0]
    conn.commit()
    return injection_id


def list_injections(conn) -> list[tuple]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, kind, params, from_sim_day, active FROM drift_injections ORDER BY id"
        )
        return cur.fetchall()


def deactivate(conn, injection_id: int) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE drift_injections SET active = FALSE WHERE id = %s",
            (injection_id,),
        )
    conn.commit()


def main() -> None:
    import os

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database-url",
        default=os.environ.get(
            "DATABASE_URL",
            "postgresql://fraudops:fraudops-local@localhost:5432/fraudops",
        ),
    )
    sub = parser.add_subparsers(dest="action", required=True)

    p = sub.add_parser("amount-factor", help="multiply TransactionAmt from a sim day")
    p.add_argument("--factor", type=float, required=True)
    p.add_argument("--from-day", type=int, required=True)

    p = sub.add_parser("nullify", help="null out features from a sim day")
    p.add_argument("--columns", required=True, help="comma-separated feature names")
    p.add_argument("--from-day", type=int, required=True)

    sub.add_parser("list", help="all injections")
    p = sub.add_parser("deactivate", help="deactivate an injection")
    p.add_argument("--id", type=int, required=True)

    args = parser.parse_args()
    conn = psycopg.connect(args.database_url)
    if args.action == "amount-factor":
        injection_id = add_injection(conn, "amount_factor", {"factor": args.factor}, args.from_day)
        print(f"injection {injection_id}: amount x{args.factor} from sim day {args.from_day}")
    elif args.action == "nullify":
        columns = [c.strip() for c in args.columns.split(",") if c.strip()]
        injection_id = add_injection(conn, "nullify", {"columns": columns}, args.from_day)
        print(f"injection {injection_id}: nullify {columns} from sim day {args.from_day}")
    elif args.action == "list":
        for row in list_injections(conn):
            print(row)
    elif args.action == "deactivate":
        deactivate(conn, args.id)
        print(f"injection {args.id} deactivated")
    conn.close()


if __name__ == "__main__":
    main()

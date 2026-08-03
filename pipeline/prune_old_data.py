"""Prunes data older than a rolling retention window, so storage stays roughly
flat forever as the accelerated clock keeps advancing (each cron tick invents
another simulated week — with no pruning, fact_deliveries and feature_snapshots
would grow without bound and eventually exceed Neon's free-tier storage cap,
same failure mode as the original historical seed).

"as_of" is the latest *simulated* order_ts already in fact_deliveries, not
wall-clock time — this system's "now" is always defined by the data itself,
same convention batch_features.py uses.

fact_predictions.delivery_id has no ON DELETE CASCADE, so dependents are
deleted before the fact_deliveries rows they point to.

Usage:
    python pipeline/prune_old_data.py
    python pipeline/prune_old_data.py --retention-days 730
"""

from __future__ import annotations

import argparse
import os

import psycopg2
from dotenv import load_dotenv

load_dotenv()

DEFAULT_RETENTION_DAYS = 730  # ~2 years


def get_connection():
    return psycopg2.connect(os.environ["POSTGRES_URL"])


def prune(conn, retention_days: int):
    with conn.cursor() as cur:
        cur.execute("SELECT MAX(order_ts) FROM fact_deliveries")
        as_of = cur.fetchone()[0]
        if as_of is None:
            print("fact_deliveries is empty — nothing to prune.")
            return

        cutoff_expr = "%s - make_interval(days => %s)"

        cur.execute(
            f"""DELETE FROM fact_predictions
                WHERE delivery_id IN (
                    SELECT delivery_id FROM fact_deliveries WHERE order_ts < {cutoff_expr}
                )""",
            (as_of, retention_days),
        )
        n_predictions = cur.rowcount

        cur.execute(
            f"DELETE FROM fact_deliveries WHERE order_ts < {cutoff_expr}",
            (as_of, retention_days),
        )
        n_deliveries = cur.rowcount

        cur.execute(
            f"DELETE FROM feature_snapshots WHERE computed_at < {cutoff_expr}",
            (as_of, retention_days),
        )
        n_snapshots = cur.rowcount

    conn.commit()
    print(
        f"Pruned {n_deliveries} deliveries, {n_predictions} predictions, "
        f"{n_snapshots} feature snapshots older than {retention_days}d "
        f"(as of {as_of.date()})."
    )


def main():
    parser = argparse.ArgumentParser(description="Prune data older than a rolling retention window.")
    parser.add_argument(
        "--retention-days", type=int, default=DEFAULT_RETENTION_DAYS,
        help=f"Rolling window size in days (default {DEFAULT_RETENTION_DAYS} = 2 years)",
    )
    args = parser.parse_args()

    conn = get_connection()
    prune(conn, args.retention_days)
    conn.close()


if __name__ == "__main__":
    main()

"""Computes rolling features from Postgres history and writes them to both
the online store (Redis) and the offline store (feature_snapshots table) —
see feature_store.py. Run this after new data lands (as part of the
pipeline tick) to keep both stores current with the same numbers.

Usage:
    python features/batch_features.py
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from feature_store import get_pg_connection, get_redis_client, write_offline_snapshots, write_online

TRAILING_WINDOW_DAYS = 30


def compute_dasher_features(conn, as_of: dt.datetime) -> list[dict]:
    cutoff = as_of - dt.timedelta(days=TRAILING_WINDOW_DAYS)
    with conn.cursor() as cur:
        cur.execute(
            """SELECT dasher_id, AVG(actual_delivery_min), COUNT(*)
               FROM fact_deliveries
               WHERE order_ts >= %s AND order_ts < %s
               GROUP BY dasher_id""",
            (cutoff, as_of),
        )
        rows = cur.fetchall()
    features = []
    for dasher_id, avg_delivery_min, n in rows:
        features.append({"entity_type": "dasher", "entity_id": dasher_id,
                          "feature_name": "avg_delivery_time_30d", "value": float(avg_delivery_min)})
        features.append({"entity_type": "dasher", "entity_id": dasher_id,
                          "feature_name": "delivery_count_30d", "value": float(n)})
    return features


def compute_merchant_features(conn, as_of: dt.datetime) -> list[dict]:
    cutoff = as_of - dt.timedelta(days=TRAILING_WINDOW_DAYS)
    with conn.cursor() as cur:
        cur.execute(
            """SELECT merchant_id, AVG(prep_time_min),
                      100.0 * SUM(CASE WHEN is_late THEN 1 ELSE 0 END) / COUNT(*)
               FROM fact_deliveries
               WHERE order_ts >= %s AND order_ts < %s
               GROUP BY merchant_id""",
            (cutoff, as_of),
        )
        rows = cur.fetchall()
    features = []
    for merchant_id, avg_prep, late_rate in rows:
        features.append({"entity_type": "merchant", "entity_id": merchant_id,
                          "feature_name": "avg_prep_time_30d", "value": float(avg_prep)})
        features.append({"entity_type": "merchant", "entity_id": merchant_id,
                          "feature_name": "late_rate_30d", "value": float(late_rate)})
    return features


def compute_zone_features(conn, as_of: dt.datetime) -> list[dict]:
    cutoff = as_of - dt.timedelta(days=TRAILING_WINDOW_DAYS)
    with conn.cursor() as cur:
        cur.execute(
            """SELECT zone_id, AVG(travel_time_min), COUNT(*)
               FROM fact_deliveries
               WHERE order_ts >= %s AND order_ts < %s
               GROUP BY zone_id""",
            (cutoff, as_of),
        )
        rows = cur.fetchall()
    features = []
    for zone_id, avg_travel, n in rows:
        features.append({"entity_type": "zone", "entity_id": zone_id,
                          "feature_name": "avg_travel_time_30d", "value": float(avg_travel)})
        features.append({"entity_type": "zone", "entity_id": zone_id,
                          "feature_name": "order_volume_30d", "value": float(n)})
    return features


def get_latest_order_ts(conn) -> dt.datetime:
    with conn.cursor() as cur:
        cur.execute("SELECT MAX(order_ts) FROM fact_deliveries")
        row = cur.fetchone()
    if row is None or row[0] is None:
        raise RuntimeError("fact_deliveries is empty — run historical_generator.py first.")
    return row[0]


def main():
    conn = get_pg_connection()
    redis_client = get_redis_client()

    as_of = get_latest_order_ts(conn)
    print(f"Computing features as of {as_of.isoformat()} (trailing {TRAILING_WINDOW_DAYS} days)...")

    all_features = (
        compute_dasher_features(conn, as_of)
        + compute_merchant_features(conn, as_of)
        + compute_zone_features(conn, as_of)
    )

    for f in all_features:
        write_online(redis_client, f["entity_type"], f["entity_id"], f["feature_name"], f["value"])

    snapshot_rows = [
        (f["entity_type"], f["entity_id"], f["feature_name"], round(f["value"], 4), as_of)
        for f in all_features
    ]
    write_offline_snapshots(conn, snapshot_rows)

    n_dashers = len({f["entity_id"] for f in all_features if f["entity_type"] == "dasher"})
    n_merchants = len({f["entity_id"] for f in all_features if f["entity_type"] == "merchant"})
    n_zones = len({f["entity_id"] for f in all_features if f["entity_type"] == "zone"})

    conn.close()
    print(f"Computed {len(all_features)} features across {n_dashers} dashers, "
          f"{n_merchants} merchants, {n_zones} zones.")
    print("Wrote to Redis (online) and feature_snapshots (offline, Postgres).")


if __name__ == "__main__":
    main()

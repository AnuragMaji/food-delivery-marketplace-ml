"""Thin wrapper over the two feature stores: Redis (online — fast lookup at
prediction time) and the feature_snapshots Postgres table (offline — dated
history, used for training/auditing). Both are written from the same
computation in batch_features.py, so online and offline never disagree
about what a feature means.
"""

from __future__ import annotations

import datetime as dt
import os

import psycopg2
import psycopg2.extras
import redis
from dotenv import load_dotenv

load_dotenv()


def get_redis_client() -> redis.Redis:
    return redis.from_url(os.environ["REDIS_URL"], decode_responses=True)


def get_pg_connection():
    return psycopg2.connect(os.environ["POSTGRES_URL"])


def online_key(entity_type: str, entity_id: int, feature_name: str) -> str:
    return f"feature:{entity_type}:{entity_id}:{feature_name}"


def write_online(redis_client: redis.Redis, entity_type: str, entity_id: int, feature_name: str, value: float):
    redis_client.set(online_key(entity_type, entity_id, feature_name), value)


def write_online_batch(redis_client: redis.Redis, entries: list[tuple[str, int, str, float]]):
    """entries: (entity_type, entity_id, feature_name, value). Batches every
    write into a single Redis pipeline (one round trip) instead of one
    round trip per feature — matters a lot once Redis is remote (e.g.
    Upstash): thousands of individual .set() calls each pay real network
    latency, turning a sub-2-second local job into a many-minute one."""
    if not entries:
        return
    pipe = redis_client.pipeline(transaction=False)
    for entity_type, entity_id, feature_name, value in entries:
        pipe.set(online_key(entity_type, entity_id, feature_name), value)
    pipe.execute()


def read_online(redis_client: redis.Redis, entity_type: str, entity_id: int, feature_name: str) -> float | None:
    raw = redis_client.get(online_key(entity_type, entity_id, feature_name))
    return float(raw) if raw is not None else None


def write_offline_snapshots(conn, rows: list[tuple]):
    """rows: (entity_type, entity_id, feature_name, feature_value, computed_at).
    Idempotent — re-running for the same (entity, feature, timestamp) is a no-op."""
    if not rows:
        return
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            """INSERT INTO feature_snapshots (entity_type, entity_id, feature_name, feature_value, computed_at)
               VALUES %s ON CONFLICT (entity_type, entity_id, feature_name, computed_at) DO NOTHING""",
            rows,
        )
    conn.commit()


def read_latest_offline_snapshot(conn, entity_type: str, entity_id: int, feature_name: str,
                                  as_of: dt.datetime) -> float | None:
    """Point-in-time-correct lookup: the most recent snapshot at or before as_of —
    what training should use, never a snapshot from after the fact."""
    with conn.cursor() as cur:
        cur.execute(
            """SELECT feature_value FROM feature_snapshots
               WHERE entity_type=%s AND entity_id=%s AND feature_name=%s AND computed_at <= %s
               ORDER BY computed_at DESC LIMIT 1""",
            (entity_type, entity_id, feature_name, as_of),
        )
        row = cur.fetchone()
    return float(row[0]) if row else None

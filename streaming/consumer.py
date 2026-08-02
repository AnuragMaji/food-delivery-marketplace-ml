"""Consumes new messages from the Redpanda topic deliveries.raw, computes
windowed aggregate metrics into Redis (orders/min-equivalent per batch,
average ETA, late rate, per-zone demand), and writes the raw batch to a
JSON file for the contract gate (contracts/validate_batch.py) to pick up.

This is a batch-style consumer, not a long-running service: it polls until
no new messages arrive for a short timeout, then exits — matching the
pipeline's tick-based (cron-driven) nature rather than continuous streaming.
A consumer group with auto-commit means each run only picks up messages
published since the last run, not the whole topic history.

Usage:
    python streaming/consumer.py --output consumed_batch.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import defaultdict

import redis
from dotenv import load_dotenv
from kafka import KafkaConsumer

load_dotenv()

CONSUMER_GROUP = "deliveries-consumer-group"
POLL_TIMEOUT_MS = 5000  # stop after 5s of no new messages


def get_consumer() -> KafkaConsumer:
    brokers = os.environ["REDPANDA_BROKERS"].split(",")
    topic = os.environ.get("DELIVERIES_TOPIC", "deliveries.raw")
    return KafkaConsumer(
        topic,
        bootstrap_servers=brokers,
        group_id=CONSUMER_GROUP,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        consumer_timeout_ms=POLL_TIMEOUT_MS,
    )


def get_redis_client() -> redis.Redis:
    return redis.from_url(os.environ["REDIS_URL"])


def consume_available(consumer: KafkaConsumer) -> list[dict]:
    """Reads all currently-available messages. The for-loop ends on its own
    once consumer_timeout_ms elapses with no new messages (kafka-python
    raises StopIteration internally to end the iteration)."""
    return [message.value for message in consumer]


def compute_window_metrics(records: list[dict]) -> dict:
    """Aggregation over this batch: total orders, orders per zone, average
    ETA, late rate — the kind of rolling metric a streaming consumer
    maintains, here computed per-batch rather than incrementally."""
    by_zone = defaultdict(int)
    etas = []
    order_dates = []
    late_count = 0
    for r in records:
        by_zone[r["zone_id"]] += 1
        etas.append(r["promised_eta_min"])
        order_dates.append(r["order_ts"])
        if r.get("is_late"):
            late_count += 1

    return {
        "total_orders": len(records),
        "orders_by_zone": dict(by_zone),
        "avg_promised_eta_min": round(statistics.mean(etas), 2),
        "late_rate_pct": round(100.0 * late_count / len(records), 2),
        "date_range_start": min(order_dates),
        "date_range_end": max(order_dates),
    }


def write_metrics_to_redis(redis_client: redis.Redis, metrics: dict):
    redis_client.set("metrics:latest_batch:total_orders", metrics["total_orders"])
    redis_client.set("metrics:latest_batch:avg_promised_eta_min", metrics["avg_promised_eta_min"])
    redis_client.set("metrics:latest_batch:late_rate_pct", metrics["late_rate_pct"])
    # ISO date strings, not full timestamps — the dashboard shows this as a
    # human date range ("covers Sep 6-12, 2026"), not a precise instant.
    redis_client.set("metrics:latest_batch:date_range_start", metrics["date_range_start"][:10])
    redis_client.set("metrics:latest_batch:date_range_end", metrics["date_range_end"][:10])
    for zone_id, count in metrics["orders_by_zone"].items():
        redis_client.set(f"metrics:zone:{zone_id}:latest_batch_orders", count)
        redis_client.incrby(f"metrics:zone:{zone_id}:orders_total", count)
    redis_client.incrby("metrics:orders_total", metrics["total_orders"])


def main():
    parser = argparse.ArgumentParser(description="Consume new delivery records from Redpanda and aggregate into Redis.")
    parser.add_argument("--output", type=str, default="consumed_batch.json", help="Path to write the consumed raw batch")
    args = parser.parse_args()

    consumer = get_consumer()
    print("Polling for new messages...")
    records = consume_available(consumer)
    consumer.close()
    print(f"Consumed {len(records)} new records.")

    if records:
        redis_client = get_redis_client()
        metrics = compute_window_metrics(records)
        write_metrics_to_redis(redis_client, metrics)
        print(f"Wrote rolling metrics to Redis: {metrics}")

    with open(args.output, "w") as f:
        json.dump(records, f)
    print(f"Wrote {len(records)} raw records to {args.output}")


if __name__ == "__main__":
    main()

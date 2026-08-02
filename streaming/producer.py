"""Publishes a batch of raw delivery records (e.g. data-gen/realtime_generator.py's
JSON output) to the Redpanda topic deliveries.raw.

Usage:
    python streaming/producer.py --input realtime_batch.json
"""

from __future__ import annotations

import argparse
import json
import os

from dotenv import load_dotenv
from kafka import KafkaProducer

load_dotenv()


def get_producer() -> KafkaProducer:
    brokers = os.environ["REDPANDA_BROKERS"].split(",")
    return KafkaProducer(
        bootstrap_servers=brokers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )


def publish_batch(producer: KafkaProducer, topic: str, records: list[dict]) -> int:
    for record in records:
        producer.send(topic, value=record)
    producer.flush()
    return len(records)


def main():
    parser = argparse.ArgumentParser(description="Publish a batch of delivery records to Redpanda.")
    parser.add_argument("--input", type=str, required=True, help="Path to the JSON batch to publish")
    args = parser.parse_args()

    with open(args.input) as f:
        records = json.load(f)

    topic = os.environ.get("DELIVERIES_TOPIC", "deliveries.raw")
    producer = get_producer()
    n = publish_batch(producer, topic, records)
    producer.close()
    print(f"Published {n} records to topic '{topic}'.")


if __name__ == "__main__":
    main()

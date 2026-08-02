"""Data contract gate: validates a batch of raw delivery records (e.g. the
JSON output of data-gen/realtime_generator.py) against the Great Expectations
suite in expectations/deliveries_suite.py. Passing rows are written to a
validated-batch JSON file, ready for the preprocessing/cleanup step. Failing
rows are inserted into deliveries_quarantine with the specific expectations
they violated, and dropped from the output batch.

Usage:
    python contracts/validate_batch.py --input realtime_batch.json
    python contracts/validate_batch.py --input realtime_batch.json --output validated.json --batch-id week-2026-08-09
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import great_expectations as ge
import pandas as pd
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent / "expectations"))
from deliveries_suite import apply_deliveries_suite

load_dotenv()


def get_connection():
    return psycopg2.connect(os.environ["POSTGRES_URL"])


def validate_batch(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Returns (passing_records, quarantine_entries). Each quarantine entry
    is {"raw_record": ..., "failed_expectations": [...]}."""
    if not records:
        return [], []

    df = pd.DataFrame(records)

    context = ge.get_context(mode="ephemeral")
    datasource = context.sources.add_pandas("deliveries_datasource")
    data_asset = datasource.add_dataframe_asset(name="deliveries_batch")
    batch_request = data_asset.build_batch_request(dataframe=df)
    context.add_or_update_expectation_suite("deliveries_suite")
    validator = context.get_validator(batch_request=batch_request, expectation_suite_name="deliveries_suite")

    results = apply_deliveries_suite(validator)

    failed_indices: dict[int, list[str]] = {}
    for check_name, result in results:
        if not result.success:
            for idx in (result.result.get("unexpected_index_list") or []):
                failed_indices.setdefault(idx, []).append(check_name)

    passing_records, quarantine_entries = [], []
    for idx, record in enumerate(records):
        if idx in failed_indices:
            quarantine_entries.append({"raw_record": record, "failed_expectations": failed_indices[idx]})
        else:
            passing_records.append(record)

    return passing_records, quarantine_entries


def insert_quarantine(conn, quarantine_entries: list[dict], batch_id: str):
    if not quarantine_entries:
        return
    rows = [
        (psycopg2.extras.Json(e["raw_record"]), psycopg2.extras.Json(e["failed_expectations"]), batch_id)
        for e in quarantine_entries
    ]
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            "INSERT INTO deliveries_quarantine (raw_record, failed_expectations, batch_id) VALUES %s",
            rows,
        )
    conn.commit()


def main():
    parser = argparse.ArgumentParser(description="Validate a batch of raw delivery records against the data contract.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input JSON batch (list of delivery records)")
    parser.add_argument("--output", type=str, default="validated_batch.json", help="Path to write the passing records")
    parser.add_argument("--batch-id", type=str, default=None, help="Identifier for this batch (default: input filename)")
    args = parser.parse_args()

    with open(args.input) as f:
        records = json.load(f)

    batch_id = args.batch_id or Path(args.input).stem
    passing, quarantine = validate_batch(records)

    print(f"Validated {len(records)} records: {len(passing)} passed, {len(quarantine)} quarantined.")

    if quarantine:
        conn = get_connection()
        insert_quarantine(conn, quarantine, batch_id)
        conn.close()
        print(f"  wrote {len(quarantine)} quarantined rows to deliveries_quarantine (batch_id={batch_id})")

    with open(args.output, "w") as f:
        json.dump(passing, f)
    print(f"Wrote {len(passing)} passing records to {args.output}")


if __name__ == "__main__":
    main()

"""Preprocessing/cleanup: the last step before a validated batch lands in
fact_deliveries (the "silver" layer). Takes the contract gate's *passing*
output (contracts/validate_batch.py) and applies:
  - dedup (defensive — same user/merchant/dasher/timestamp combination)
  - null imputation (defensive — this generator never produces nulls in
    these fields today, but a real upstream feed could)
  - type coercion (JSON round-trip -> proper Python/Postgres types)
  - outlier clipping (tighter, realistic bounds than the contract gate's
    generous sanity bounds, informed by data-gen/distributions.py's actual
    sampling parameters)
then bulk-inserts the cleaned batch into fact_deliveries.

Usage:
    python pipeline/preprocess.py --input validated_batch.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import statistics

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()

# Tighter, realistic bounds than the contract gate's generous sanity bounds —
# meant to clip genuine tail-noise, not central-tendency data.
CLIP_BOUNDS = {
    "prep_time_min": (3, 90),
    "travel_time_min": (1, 120),
    "pickup_wait_min": (0, 30),
    "tip_amount": (0, 100),
    "subtotal": (0, 300),
    "total_amount": (0, 300),
    "item_count": (1, 20),
    "refund_amount": (0, 300),
}

# Nullable numeric fields that get median-imputed if null.
IMPUTABLE_NUMERIC_FIELDS = ["actual_delivery_min", "pickup_wait_min", "travel_time_min"]

DEDUP_KEY_FIELDS = ["user_id", "merchant_id", "dasher_id", "order_ts"]

FACT_DELIVERIES_INSERT = """
    INSERT INTO fact_deliveries (
        user_id, merchant_id, dasher_id, zone_id, promo_id, date_key, order_ts,
        promised_eta_min, actual_delivery_min, prep_time_min, pickup_wait_min,
        travel_time_min, subtotal, discount_amount, delivery_fee, tip_amount,
        total_amount, item_count, is_late, issue_type, refund_amount, source
    ) VALUES %s
"""


def get_connection():
    return psycopg2.connect(os.environ["POSTGRES_URL"])


def dedup(records: list[dict]) -> tuple[list[dict], int]:
    seen = set()
    deduped = []
    for r in records:
        key = tuple(r[f] for f in DEDUP_KEY_FIELDS)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    return deduped, len(records) - len(deduped)


def impute_nulls(records: list[dict]) -> int:
    n_imputed = 0
    for field in IMPUTABLE_NUMERIC_FIELDS:
        values = [r[field] for r in records if r.get(field) is not None]
        if not values:
            continue
        fallback = round(statistics.median(values), 2)
        for r in records:
            if r.get(field) is None:
                r[field] = fallback
                n_imputed += 1
    return n_imputed


def clip_outliers(records: list[dict]) -> int:
    n_clipped = 0
    for r in records:
        for field, (lo, hi) in CLIP_BOUNDS.items():
            value = r.get(field)
            if value is None:
                continue
            clipped = max(lo, min(hi, value))
            if clipped != value:
                r[field] = clipped
                n_clipped += 1
    return n_clipped


def preprocess_batch(records: list[dict]) -> tuple[list[dict], dict]:
    records, n_dupes = dedup(records)
    n_imputed = impute_nulls(records)
    n_clipped = clip_outliers(records)
    stats = {"duplicates_removed": n_dupes, "nulls_imputed": n_imputed, "values_clipped": n_clipped}
    return records, stats


def coerce_types(records: list[dict]) -> list[tuple]:
    """Converts JSON-native records back to Postgres-ready tuples, in
    fact_deliveries column order."""
    rows = []
    for r in records:
        order_ts = dt.datetime.fromisoformat(r["order_ts"])
        rows.append((
            int(r["user_id"]), int(r["merchant_id"]), int(r["dasher_id"]), int(r["zone_id"]),
            int(r["promo_id"]) if r.get("promo_id") is not None else None,
            int(r["date_key"]), order_ts,
            float(r["promised_eta_min"]),
            float(r["actual_delivery_min"]) if r.get("actual_delivery_min") is not None else None,
            float(r["prep_time_min"]),
            float(r["pickup_wait_min"]) if r.get("pickup_wait_min") is not None else None,
            float(r["travel_time_min"]) if r.get("travel_time_min") is not None else None,
            float(r["subtotal"]), float(r["discount_amount"]), float(r["delivery_fee"]),
            float(r["tip_amount"]), float(r["total_amount"]), int(r["item_count"]),
            bool(r["is_late"]) if r.get("is_late") is not None else None,
            r.get("issue_type"), float(r["refund_amount"]), r["source"],
        ))
    return rows


def insert_fact_deliveries(conn, rows: list[tuple]) -> list[int]:
    """Returns the delivery_ids Postgres actually assigned, so callers (the
    pipeline orchestrator) know exactly which new deliveries to score."""
    if not rows:
        return []
    with conn.cursor() as cur:
        inserted = psycopg2.extras.execute_values(cur, FACT_DELIVERIES_INSERT + " RETURNING delivery_id", rows, fetch=True)
    conn.commit()
    return [row[0] for row in inserted]


def main():
    parser = argparse.ArgumentParser(description="Clean a validated batch and load it into fact_deliveries.")
    parser.add_argument("--input", type=str, required=True, help="Path to the validated JSON batch")
    parser.add_argument("--output-ids", type=str, default=None,
                         help="Optional path to write the list of inserted delivery_ids as JSON")
    args = parser.parse_args()

    with open(args.input) as f:
        records = json.load(f)

    cleaned, stats = preprocess_batch(records)
    print(f"Preprocessed {len(records)} records: {stats['duplicates_removed']} duplicates removed, "
          f"{stats['nulls_imputed']} nulls imputed, {stats['values_clipped']} values clipped.")

    rows = coerce_types(cleaned)
    conn = get_connection()
    inserted_ids = insert_fact_deliveries(conn, rows)
    conn.close()
    print(f"Inserted {len(rows)} rows into fact_deliveries.")

    if args.output_ids:
        with open(args.output_ids, "w") as f:
            json.dump(inserted_ids, f)


if __name__ == "__main__":
    main()

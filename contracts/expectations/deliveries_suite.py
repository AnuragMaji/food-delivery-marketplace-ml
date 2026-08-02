"""Great Expectations suite definition for a batch of raw delivery records —
the data contract gate applied before rows are allowed to land in
fact_deliveries. Checks are intentionally scoped to what's cheap and robust
to verify at ingestion time: not-null on required fields, sane value ranges,
and set-membership on categorical fields. Not attempting to re-derive
business-logic consistency (e.g. total_amount reconciling exactly against
its components) — that's what preprocessing/cleanup and downstream
analytics are for, not the contract gate.
"""

from __future__ import annotations

NOT_NULL_COLUMNS = [
    "user_id", "merchant_id", "dasher_id", "zone_id", "date_key", "order_ts",
    "promised_eta_min", "prep_time_min", "subtotal", "total_amount",
    "item_count", "source",
]

# column: (min_value, max_value) — generous sanity bounds, not tight business rules
RANGE_CHECKS = {
    "prep_time_min": (0, 180),
    "travel_time_min": (0, 180),
    "pickup_wait_min": (0, 60),
    "promised_eta_min": (0, 300),
    "actual_delivery_min": (0, 300),
    "subtotal": (0, 1000),
    "total_amount": (0, 1000),
    "delivery_fee": (0, 20),
    "tip_amount": (0, 500),
    "item_count": (1, 50),
    "refund_amount": (0, 1000),
}

SET_CHECKS = {
    "issue_type": [None, "misplaced", "not_delivered", "wrong_order"],
    "source": ["historical", "realtime"],
}


def apply_deliveries_suite(validator) -> list[tuple[str, object]]:
    """Applies the full deliveries expectation suite to a GE validator bound
    to a batch DataFrame. Returns a list of (check_name, ExpectationValidationResult)
    so callers can inspect row-level failures via result.result["unexpected_index_list"]."""
    result_format = {"result_format": "COMPLETE"}
    results = []

    for column in NOT_NULL_COLUMNS:
        r = validator.expect_column_values_to_not_be_null(column, result_format=result_format)
        results.append((f"not_null:{column}", r))

    for column, (min_value, max_value) in RANGE_CHECKS.items():
        r = validator.expect_column_values_to_be_between(
            column, min_value=min_value, max_value=max_value, result_format=result_format)
        results.append((f"range:{column}", r))

    for column, value_set in SET_CHECKS.items():
        r = validator.expect_column_values_to_be_in_set(column, value_set=value_set, result_format=result_format)
        results.append((f"set:{column}", r))

    return results

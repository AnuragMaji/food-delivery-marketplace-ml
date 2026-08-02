"""Accelerated realtime generator: produces ~1 simulated week of new
marketplace activity per invocation, resuming from wherever dim_date last
left off (each invocation is a fresh, stateless process — no in-memory state
persists between runs, since this is eventually driven by a cron tick).

Per the bronze -> contract-gate -> silver architecture, this script does NOT
insert delivery records directly into fact_deliveries — it only *produces*
the raw batch (written to a JSON file). The batch is meant to flow through
producer.py -> Redpanda -> the (not yet built) Great Expectations contract
gate before landing in Postgres. dim_date, weather, and a light entity
trickle ARE inserted directly here, since those aren't part of the
contract-gated fact pipeline.

Usage:
    python data-gen/realtime_generator.py
    python data-gen/realtime_generator.py --weeks 2 --output batch.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import psycopg2.extras

sys.path.insert(0, str(Path(__file__).parent))
import distributions as dist
import weather_generator as wx
from historical_generator import (
    CHURN_SHAPE,
    DASHER_CHURN_SCALE_DAYS,
    MERCHANT_CHURN_SCALE_DAYS,
    PROMISED_ETA_BUFFER_MIN,
    USER_CHURN_SCALE_DAYS,
    VEHICLE_WEIGHTS,
    apply_promo_discount,
    get_connection,
    insert_dim_date,
    pick_promo,
)

BATCH_DAYS = 7

# Light ongoing trickle during the realtime phase (post-launch organic
# growth) — much smaller than the historical S-curve, just enough that the
# population isn't permanently fixed after the historical seed.
TRICKLE_USERS_PER_WEEK = {"high": 8, "medium": 4, "low": 1}
TRICKLE_MERCHANT_PROB = {"high": 0.6, "medium": 0.3, "low": 0.1}
TRICKLE_DASHERS_PER_WEEK = {"high": 3, "medium": 1, "low": 0}
TRICKLE_DASHER_PROB_LOW = 0.3  # low-tier zones get a probabilistic extra dasher instead of a guaranteed 0

# No natural "window end" in an open-ended realtime sim — cap Weibull churn
# sampling against a far horizon instead.
CHURN_HORIZON_YEARS = 5

TRAILING_WINDOW_DAYS = 180

# Expected orders/hour = this rate * a zone's current active-user weight sum
# * time-of-day/weather/event multipliers. Calibrated for continuity against
# the historical generator's last observed week (which used a flat
# BASE_ORDER_RATE per density tier): backing out "rate per unit of active
# weight" from that data landed at ~0.0128-0.0130 consistently across all
# three density tiers — i.e. density-tier differences were already fully
# explained by proportionally more/fewer active users, so one constant
# replaces the three, and demand now responds to trickle-driven growth.
REALTIME_RATE_PER_ACTIVE_WEIGHT = 0.0129


def get_current_sim_date(conn) -> dt.date:
    with conn.cursor() as cur:
        cur.execute("SELECT MAX(full_date) FROM dim_date")
        row = cur.fetchone()
    if row is None or row[0] is None:
        raise RuntimeError("dim_date is empty — run historical_generator.py first to seed the marketplace.")
    return row[0]


def get_zone_name_to_id(conn) -> dict[str, int]:
    with conn.cursor() as cur:
        cur.execute("SELECT zone_id, zone_name FROM dim_zones")
        return {name: zid for zid, name in cur.fetchall()}


def get_events_df(conn) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(
            """SELECT event_type, start_date, end_date, affected_state,
                      demand_multiplier, order_value_multiplier, description
               FROM dim_external_events"""
        )
        rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=[
        "event_type", "start_date", "end_date", "affected_state",
        "demand_multiplier", "order_value_multiplier", "description",
    ])
    df["demand_multiplier"] = df["demand_multiplier"].astype(float)
    df["order_value_multiplier"] = df["order_value_multiplier"].astype(float)
    return df


def insert_weather_for_range(conn, rng, zone_name_to_id, start_date, end_date):
    weather_df = wx.generate_weather_series(dist.ZONES, start_date, end_date, rng)
    weather_rows = [
        (zone_name_to_id[r["zone_name"]], r["date"], r["precipitation_mm"], r["temp_high_c"], r["temp_low_c"],
         r["wind_speed_kmh"], r["condition"], r["is_flash_flood"], r["is_heatwave"])
        for r in weather_df.to_dict("records")
    ]
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            """INSERT INTO fact_weather_daily (zone_id, date, precipitation_mm, temp_high_c, temp_low_c, wind_speed_kmh, condition, is_flash_flood, is_heatwave)
               VALUES %s""",
            weather_rows,
        )
    conn.commit()
    return wx.weather_lookup(weather_df)


def insert_trickle_entities(conn, rng, zone_name_to_id, batch_start, batch_end):
    """Light ongoing acquisition during the realtime phase — simplified (no
    S-curve needed at this point), just steady organic growth. New entities'
    activation_date typically falls mid-batch, so they naturally become
    eligible starting the *following* batch's active-pool query, not this one."""
    churn_horizon = batch_end + dt.timedelta(days=365 * CHURN_HORIZON_YEARS)
    n_users = n_merchants = n_dashers = 0

    with conn.cursor() as cur:
        for zone in dist.ZONES:
            density = zone["density_tier"]
            zone_id = zone_name_to_id[zone["zone_name"]]

            for _ in range(TRICKLE_USERS_PER_WEEK[density]):
                signup_date = batch_start + dt.timedelta(days=int(rng.integers(0, BATCH_DAYS)))
                activation_date = min(churn_horizon, signup_date + dt.timedelta(days=int(rng.integers(1, 6))))
                churn_date = dist.sample_churn_date(rng, activation_date, churn_horizon, shape=CHURN_SHAPE, scale_days=USER_CHURN_SCALE_DAYS)
                activity_weight = round(dist.sample_activity_weight(rng), 3)
                status = "active" if churn_date is None else "inactive"
                cur.execute(
                    """INSERT INTO dim_users (signup_date, activation_date, churn_date, activity_weight,
                                               home_zone_id, acquisition_channel, is_subscriber, status)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
                    (signup_date, activation_date, churn_date, activity_weight, zone_id,
                     str(rng.choice(dist.ACQUISITION_CHANNELS)), bool(rng.random() < 0.35), status),
                )
                n_users += 1

            if rng.random() < TRICKLE_MERCHANT_PROB[density]:
                cuisine = str(rng.choice(dist.CUISINES))
                baseline_prep = float(rng.uniform(15, 35))
                onboarded_date = batch_start + dt.timedelta(days=int(rng.integers(0, BATCH_DAYS)))
                activation_date = min(churn_horizon, onboarded_date + dt.timedelta(days=int(rng.integers(3, 15))))
                churn_date = dist.sample_churn_date(rng, activation_date, churn_horizon, shape=CHURN_SHAPE, scale_days=MERCHANT_CHURN_SCALE_DAYS)
                activity_weight = round(dist.sample_activity_weight(rng), 3)
                cur.execute(
                    """INSERT INTO dim_merchants (name, cuisine_type, zone_id, baseline_prep_time_min, rating, is_active,
                                                   onboarded_date, activation_date, churn_date, activity_weight)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                    (f"{zone['zone_name']} {cuisine.title()} Kitchen New", cuisine, zone_id,
                     round(baseline_prep, 2), round(dist.sample_rating(rng), 1), churn_date is None,
                     onboarded_date, activation_date, churn_date, activity_weight),
                )
                n_merchants += 1

            n_dashers_this_zone = TRICKLE_DASHERS_PER_WEEK[density]
            if n_dashers_this_zone == 0 and rng.random() < TRICKLE_DASHER_PROB_LOW:
                n_dashers_this_zone = 1
            for _ in range(n_dashers_this_zone):
                signup_date = batch_start + dt.timedelta(days=int(rng.integers(0, BATCH_DAYS)))
                activation_date = min(churn_horizon, signup_date + dt.timedelta(days=int(rng.integers(2, 10))))
                churn_date = dist.sample_churn_date(rng, activation_date, churn_horizon, shape=CHURN_SHAPE, scale_days=DASHER_CHURN_SCALE_DAYS)
                activity_weight = round(dist.sample_activity_weight(rng), 3)
                vehicle = str(rng.choice(dist.VEHICLE_TYPES, p=VEHICLE_WEIGHTS[density]))
                status = "active" if churn_date is None else "inactive"
                cur.execute(
                    """INSERT INTO dim_dashers (signup_date, activation_date, churn_date, activity_weight,
                                                 vehicle_type, home_zone_id, rating, status)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
                    (signup_date, activation_date, churn_date, activity_weight, vehicle, zone_id,
                     round(dist.sample_rating(rng), 1), status),
                )
                n_dashers += 1
    conn.commit()
    print(f"  trickle acquisition: +{n_users} users, +{n_merchants} merchants, +{n_dashers} dashers")


def weighted_sample(rng, ids_weights: list[tuple[int, float]], n: int) -> list[int]:
    ids = np.array([iw[0] for iw in ids_weights])
    weights = np.array([iw[1] for iw in ids_weights], dtype=float)
    probs = weights / weights.sum()
    idxs = rng.choice(len(ids), size=n, p=probs)
    return ids[idxs].tolist()


def get_trailing_stats(conn, user_ids, as_of_date, window_days=TRAILING_WINDOW_DAYS):
    if not user_ids:
        return {}
    cutoff = as_of_date - dt.timedelta(days=window_days)
    with conn.cursor() as cur:
        cur.execute(
            """SELECT user_id, COUNT(*), COALESCE(SUM(total_amount), 0)
               FROM fact_deliveries
               WHERE user_id = ANY(%s) AND order_ts >= %s AND order_ts < %s
               GROUP BY user_id""",
            (user_ids, cutoff, as_of_date),
        )
        rows = cur.fetchall()
    return {uid: (cnt, float(gmv)) for uid, cnt, gmv in rows}


def get_active_pools(conn, zone_id, as_of_date):
    with conn.cursor() as cur:
        cur.execute(
            """SELECT user_id, activity_weight, activation_date FROM dim_users
               WHERE home_zone_id=%s AND activation_date<=%s AND (churn_date IS NULL OR churn_date > %s)""",
            (zone_id, as_of_date, as_of_date),
        )
        users = [(r[0], float(r[1]), r[2]) for r in cur.fetchall()]

        cur.execute(
            """SELECT merchant_id, activity_weight, cuisine_type, baseline_prep_time_min FROM dim_merchants
               WHERE zone_id=%s AND activation_date<=%s AND (churn_date IS NULL OR churn_date > %s)""",
            (zone_id, as_of_date, as_of_date),
        )
        merchants = {
            r[0]: {"weight": float(r[1]), "cuisine_type": r[2], "baseline_prep_time_min": float(r[3])}
            for r in cur.fetchall()
        }

        cur.execute(
            """SELECT dasher_id, activity_weight FROM dim_dashers
               WHERE home_zone_id=%s AND activation_date<=%s AND (churn_date IS NULL OR churn_date > %s)""",
            (zone_id, as_of_date, as_of_date),
        )
        dashers = [(r[0], float(r[1])) for r in cur.fetchall()]

        cur.execute(
            """SELECT promo_id, promo_type, discount_pct, flat_amount, merchant_id, start_date, end_date
               FROM dim_promotions WHERE end_date >= %s""",
            (as_of_date,),
        )
        promo_rows = cur.fetchall()

    promos_by_merchant, platform_promos = {}, []
    for promo_id, promo_type, discount_pct, flat_amount, merchant_id, start_date, end_date in promo_rows:
        promo = {
            "promo_id": promo_id, "promo_type": promo_type,
            "discount_pct": float(discount_pct) if discount_pct is not None else None,
            "flat_amount": float(flat_amount) if flat_amount is not None else None,
            "start_date": start_date, "end_date": end_date,
        }
        if merchant_id is None:
            platform_promos.append(promo)
        else:
            promos_by_merchant.setdefault(merchant_id, []).append(promo)

    return users, merchants, dashers, promos_by_merchant, platform_promos


def generate_week_for_zone(conn, rng, zone, zone_id, batch_start, weather_lut, events_df):
    density = zone["density_tier"]
    tz = ZoneInfo(zone["timezone"])

    users, merchants, dashers, promos_by_merchant, platform_promos = get_active_pools(conn, zone_id, batch_start)
    if not users or not merchants or not dashers:
        return []

    user_ids = [u[0] for u in users]
    user_activation = {u[0]: u[2] for u in users}
    user_weights = [(u[0], u[1]) for u in users]
    merchant_weights = [(mid, m["weight"]) for mid, m in merchants.items()]
    dasher_weights = [(d[0], d[1]) for d in dashers]
    active_user_weight_sum = sum(w for _, w in user_weights)

    trailing = get_trailing_stats(conn, user_ids, batch_start)
    trailing_count = dict.fromkeys(user_ids, 0)
    trailing_gmv = dict.fromkeys(user_ids, 0.0)
    for uid, (cnt, gmv) in trailing.items():
        trailing_count[uid] = cnt
        trailing_gmv[uid] = gmv

    batch_records = []
    for day_offset in range(BATCH_DAYS):
        current = batch_start + dt.timedelta(days=day_offset)
        weather_row = weather_lut.get((zone["zone_name"], current))
        weather_demand_mult, travel_mean_mult, travel_sd_mult, weather_issue_mult = 1.0, 1.0, 1.0, 1.0
        if weather_row is not None:
            weather_demand_mult = wx.demand_weather_multiplier(
                weather_row["condition"], weather_row["is_flash_flood"], weather_row["is_heatwave"])
            travel_mean_mult, travel_sd_mult = wx.travel_time_weather_multipliers(
                weather_row["condition"], weather_row["is_flash_flood"], weather_row["is_heatwave"])
            weather_issue_mult = wx.issue_rate_multiplier(
                weather_row["condition"], weather_row["is_flash_flood"], weather_row["is_heatwave"])

        active_events = wx.active_events_for(events_df, zone["state"], current)
        event_demand_mult, event_value_mult = wx.combined_event_multipliers(active_events)

        for hour in range(24):
            local_dt = dt.datetime(current.year, current.month, current.day, hour, tzinfo=tz)
            weight = dist.hourly_order_weight(local_dt.astimezone(dt.timezone.utc), zone["timezone"])
            expected = REALTIME_RATE_PER_ACTIVE_WEIGHT * active_user_weight_sum * weight * weather_demand_mult * event_demand_mult
            n_orders = int(rng.poisson(expected))
            if n_orders == 0:
                continue

            daypart = dist.daypart_for_hour(hour)
            user_picks = weighted_sample(rng, user_weights, n_orders)
            merchant_picks = weighted_sample(rng, merchant_weights, n_orders)
            dasher_picks = weighted_sample(rng, dasher_weights, n_orders)

            for i in range(n_orders):
                user_id, merchant_id, dasher_id = user_picks[i], merchant_picks[i], dasher_picks[i]
                merchant = merchants[merchant_id]

                order_local_dt = local_dt + dt.timedelta(seconds=int(rng.integers(0, 3600)))
                order_ts = order_local_dt.astimezone(dt.timezone.utc)
                date_key = int(current.strftime("%Y%m%d")) * 100 + hour

                prep_time = dist.sample_prep_time_min(rng, merchant["baseline_prep_time_min"], daypart)
                travel_time = dist.sample_travel_time_min(rng, density, travel_mean_mult, travel_sd_mult)
                pickup_wait = dist.sample_pickup_wait_min(rng)

                subtotal = dist.sample_order_value(rng, merchant["cuisine_type"]) * event_value_mult
                promo = pick_promo(rng, merchant_id, current, promos_by_merchant, platform_promos)
                discount_amount, waives_delivery_fee = apply_promo_discount(promo, subtotal)

                delivery_fee = 0.0 if waives_delivery_fee else round(float(rng.uniform(1.99, 5.99)), 2)
                tip_amount = dist.sample_tip_amount(rng, subtotal - discount_amount)
                item_count = dist.sample_item_count(rng)

                promised_eta_min = round(prep_time + travel_time + PROMISED_ETA_BUFFER_MIN, 2)
                actual_delivery_min = round(prep_time + pickup_wait + travel_time + float(rng.normal(0, 3)), 2)
                is_late = actual_delivery_min > promised_eta_min
                total_amount = round(subtotal - discount_amount + delivery_fee + tip_amount, 2)

                tenure_days = (current - user_activation[user_id]).days
                loyalty_score = dist.compute_loyalty_score(tenure_days, trailing_count[user_id], trailing_gmv[user_id])

                issue_type = dist.sample_order_issue(rng, weather_issue_mult)
                refund_amount = dist.compute_refund_amount(issue_type, total_amount, loyalty_score) if issue_type else 0.0

                trailing_count[user_id] += 1
                trailing_gmv[user_id] += total_amount

                batch_records.append({
                    "user_id": user_id, "merchant_id": merchant_id, "dasher_id": dasher_id, "zone_id": zone_id,
                    "promo_id": promo["promo_id"] if promo else None, "date_key": date_key,
                    "order_ts": order_ts.isoformat(),
                    "promised_eta_min": promised_eta_min, "actual_delivery_min": actual_delivery_min,
                    "prep_time_min": round(prep_time, 2), "pickup_wait_min": round(pickup_wait, 2),
                    "travel_time_min": round(travel_time, 2), "subtotal": round(subtotal, 2),
                    "discount_amount": discount_amount, "delivery_fee": delivery_fee, "tip_amount": tip_amount,
                    "total_amount": total_amount, "item_count": item_count, "is_late": is_late,
                    "issue_type": issue_type, "refund_amount": refund_amount, "source": "realtime",
                })

    return batch_records


def main():
    parser = argparse.ArgumentParser(description="Generate one accelerated week of realtime marketplace activity.")
    parser.add_argument("--weeks", type=int, default=1, help="Number of accelerated weeks to generate (default 1)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: nondeterministic)")
    parser.add_argument("--output", type=str, default="realtime_batch.json", help="Output JSON path for the generated batch")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    conn = get_connection()

    zone_name_to_id = get_zone_name_to_id(conn)
    sim_date = get_current_sim_date(conn)

    all_records = []
    for week in range(args.weeks):
        batch_start = sim_date + dt.timedelta(days=1)
        batch_end = batch_start + dt.timedelta(days=BATCH_DAYS - 1)
        print(f"Week {week + 1}/{args.weeks}: generating {batch_start} to {batch_end}...")

        insert_dim_date(conn, batch_start, batch_end)
        weather_lut = insert_weather_for_range(conn, rng, zone_name_to_id, batch_start, batch_end)
        insert_trickle_entities(conn, rng, zone_name_to_id, batch_start, batch_end)
        events_df = get_events_df(conn)

        week_records = []
        for zone in dist.ZONES:
            zone_id = zone_name_to_id[zone["zone_name"]]
            week_records.extend(generate_week_for_zone(conn, rng, zone, zone_id, batch_start, weather_lut, events_df))

        print(f"  generated {len(week_records)} delivery records")
        all_records.extend(week_records)
        sim_date = batch_end

    with open(args.output, "w") as f:
        json.dump(all_records, f)
    print(f"Wrote {len(all_records)} total delivery records to {args.output}")

    conn.close()


if __name__ == "__main__":
    main()

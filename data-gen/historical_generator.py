"""One-time historical seed: inserts all dimension tables, weather, external
events, and 2-3 years of fact_deliveries into Postgres.

Usage:
    python data-gen/historical_generator.py --years 3
    python data-gen/historical_generator.py --start-date 2026-07-01 --end-date 2026-07-30  # smoke test
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from collections import deque
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
import distributions as dist
import weather_generator as wx

load_dotenv()

SEED = 42
BATCH_SIZE = 20_000

# Per-density-tier dimension sizing
USERS_PER_ZONE = {"high": 2000, "medium": 1000, "low": 400}
MERCHANTS_PER_ZONE = {"high": 150, "medium": 80, "low": 30}
DASHERS_PER_ZONE = {"high": 300, "medium": 150, "low": 60}
TOTAL_PROMOTIONS = 300

# Base expected orders/hour at demand-weight == 1.0 and 100% of the eventual
# user base active. Retuned (via dry run) after adding growth/churn/S-curve
# lag, since the combined effect suppresses the *effective* active population
# well below 100% most of the window (a real 3-year run measured 1.37M rows
# against a 6.0/3.0/1.2 baseline — this ~4.4x bump targets ~6M).
BASE_ORDER_RATE = {"high": 26.2, "medium": 13.1, "low": 5.24}

VEHICLE_WEIGHTS = {"high": [0.55, 0.30, 0.15], "medium": [0.35, 0.30, 0.35], "low": [0.15, 0.20, 0.65]}

PROMO_TYPES = ["discount_pct", "flat_discount", "free_delivery", "bogo"]
PROMO_USE_PROBABILITY = 0.30
PLATFORM_WIDE_PROMO_PROBABILITY = 0.40

# Promised ETA = prep + travel + this buffer. Tuned so is_late lands around a
# realistic ~15% (real delivery apps deliberately pad estimates); 5.0 min was
# too tight against pickup-wait variance and produced ~40% late.
PROMISED_ETA_BUFFER_MIN = 8.0

# --- Growth / churn / heterogeneity tuning -----------------------------
# Acquisition timing: logistic (S-curve) growth per zone. seed_fraction of
# the eventual population already exists at window start; USER_LAG_DAYS
# shifts user acquisition later than merchant/dasher acquisition, so supply
# (restaurants + drivers) comes online before demand (customers) follows.
SEED_FRACTION = 0.3
SCURVE_STEEPNESS = 6.0
USER_LAG_DAYS = 45.0

# Churn: Weibull hazard tenure sampling (shape < 1 = decreasing hazard rate,
# the standard retention-curve shape). Scale differs by entity type — a
# restaurant business is far stickier than an individual customer or a gig
# dasher.
CHURN_SHAPE = 0.75
USER_CHURN_SCALE_DAYS = 600.0
MERCHANT_CHURN_SCALE_DAYS = 1500.0
DASHER_CHURN_SCALE_DAYS = 450.0

FACT_DELIVERIES_INSERT = """
    INSERT INTO fact_deliveries (
        user_id, merchant_id, dasher_id, zone_id, promo_id, date_key, order_ts,
        promised_eta_min, actual_delivery_min, prep_time_min, pickup_wait_min,
        travel_time_min, subtotal, discount_amount, delivery_fee, tip_amount,
        total_amount, item_count, is_late, issue_type, refund_amount, source
    ) VALUES %s
"""


class UserHistory:
    """Rolling ~180-day order history per user (trailing order count + GMV),
    used to compute a loyalty score that scales refund generosity when an
    order issue occurs. Scoped per-zone since users don't relocate yet
    (Phase B item) — every user_id only ever appears in one zone's pool."""

    WINDOW_DAYS = 180

    def __init__(self):
        self.orders = deque()  # (date, amount)
        self.gmv_sum = 0.0

    def _evict_stale(self, current: dt.date):
        cutoff = current - dt.timedelta(days=self.WINDOW_DAYS)
        while self.orders and self.orders[0][0] < cutoff:
            _, amount = self.orders.popleft()
            self.gmv_sum -= amount

    def snapshot(self, current: dt.date) -> tuple[int, float]:
        """Trailing (order_count, gmv) as of current, BEFORE today's order."""
        self._evict_stale(current)
        return len(self.orders), self.gmv_sum

    def record(self, current: dt.date, amount: float):
        self.orders.append((current, amount))
        self.gmv_sum += amount


class ActivePool:
    """Tracks currently-active entity ids with heterogeneity weights.
    O(1) add/remove (swap-with-last) and batched weighted sampling — used so
    a multi-year run doesn't degrade as entities join/churn over time."""

    def __init__(self):
        self.ids: list[int] = []
        self.weights: list[float] = []
        self.index_of: dict[int, int] = {}
        self.weight_sum = 0.0

    def add(self, entity_id: int, weight: float):
        self.index_of[entity_id] = len(self.ids)
        self.ids.append(entity_id)
        self.weights.append(weight)
        self.weight_sum += weight

    def remove(self, entity_id: int):
        idx = self.index_of.pop(entity_id, None)
        if idx is None:
            return
        last = len(self.ids) - 1
        self.weight_sum -= self.weights[idx]
        if idx != last:
            self.ids[idx] = self.ids[last]
            self.weights[idx] = self.weights[last]
            self.index_of[self.ids[idx]] = idx
        self.ids.pop()
        self.weights.pop()

    def __len__(self):
        return len(self.ids)

    def pick(self, rng, n: int) -> list[int]:
        """Weighted sample of n entity ids (with replacement). Probabilities
        are recomputed fresh from the current weights each call (not the
        incrementally-tracked weight_sum) to avoid floating-point drift
        across thousands of add/remove operations breaking numpy's
        sum-to-1 check."""
        weights_arr = np.asarray(self.weights)
        probs = weights_arr / weights_arr.sum()
        idxs = rng.choice(len(self.ids), size=n, p=probs)
        return np.asarray(self.ids)[idxs].tolist()


def get_connection():
    return psycopg2.connect(os.environ["POSTGRES_URL"])


def insert_zones(conn) -> dict[str, int]:
    zone_name_to_id = {}
    with conn.cursor() as cur:
        for zone in dist.ZONES:
            cur.execute(
                """INSERT INTO dim_zones (zone_name, city, state, region, timezone, density_tier, latitude, longitude)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s) RETURNING zone_id""",
                (zone["zone_name"], zone["city"], zone["state"], zone["region"], zone["timezone"],
                 zone["density_tier"], zone["latitude"], zone["longitude"]),
            )
            zone_name_to_id[zone["zone_name"]] = cur.fetchone()[0]
    conn.commit()
    print(f"  inserted {len(zone_name_to_id)} zones")
    return zone_name_to_id


def insert_users(conn, rng, zone_name_to_id, start_date, end_date) -> dict[str, list[tuple]]:
    """Returns zone_name -> list of (activation_date, user_id, activity_weight,
    churn_date) sorted by activation_date."""
    span_days = max(1, (end_date - start_date).days)
    rows = []
    zone_order = []
    for zone in dist.ZONES:
        n = USERS_PER_ZONE[zone["density_tier"]]
        zone_id = zone_name_to_id[zone["zone_name"]]
        for _ in range(n):
            signup_date = dist.sample_acquisition_date(
                rng, start_date, span_days, seed_fraction=SEED_FRACTION,
                steepness=SCURVE_STEEPNESS, lag_days=USER_LAG_DAYS)
            activation_date = min(end_date, signup_date + dt.timedelta(days=int(rng.integers(1, 6))))
            churn_date = dist.sample_churn_date(
                rng, activation_date, end_date, shape=CHURN_SHAPE, scale_days=USER_CHURN_SCALE_DAYS)
            activity_weight = round(dist.sample_activity_weight(rng), 3)
            status = "active" if churn_date is None else "inactive"
            rows.append((
                signup_date, activation_date, churn_date, activity_weight, zone_id,
                str(rng.choice(dist.ACQUISITION_CHANNELS)), bool(rng.random() < 0.35), status,
            ))
            zone_order.append(zone["zone_name"])

    with conn.cursor() as cur:
        results = psycopg2.extras.execute_values(
            cur,
            """INSERT INTO dim_users (signup_date, activation_date, churn_date, activity_weight,
                                       home_zone_id, acquisition_channel, is_subscriber, status)
               VALUES %s RETURNING user_id""",
            rows, fetch=True,
        )
    conn.commit()

    zone_users: dict[str, list[tuple]] = {z["zone_name"]: [] for z in dist.ZONES}
    for (user_id,), zone_name, row in zip(results, zone_order, rows):
        _, activation_date, churn_date, activity_weight = row[0], row[1], row[2], row[3]
        zone_users[zone_name].append((activation_date, user_id, activity_weight, churn_date))
    for zone_name in zone_users:
        zone_users[zone_name].sort(key=lambda t: t[0])
    print(f"  inserted {len(rows)} users")
    return zone_users


def insert_merchants(conn, rng, zone_name_to_id, start_date, end_date) -> dict[str, list[tuple]]:
    """Returns zone_name -> list of (activation_date, merchant_id, activity_weight,
    churn_date, cuisine_type, baseline_prep_time_min) sorted by activation_date."""
    span_days = max(1, (end_date - start_date).days)
    rows = []
    zone_order = []
    for zone in dist.ZONES:
        n = MERCHANTS_PER_ZONE[zone["density_tier"]]
        zone_id = zone_name_to_id[zone["zone_name"]]
        for i in range(n):
            cuisine = str(rng.choice(dist.CUISINES))
            baseline_prep = float(rng.uniform(15, 35))
            onboarded_date = dist.sample_acquisition_date(
                rng, start_date, span_days, seed_fraction=SEED_FRACTION, steepness=SCURVE_STEEPNESS, lag_days=0.0)
            activation_date = min(end_date, onboarded_date + dt.timedelta(days=int(rng.integers(3, 15))))
            churn_date = dist.sample_churn_date(
                rng, activation_date, end_date, shape=CHURN_SHAPE, scale_days=MERCHANT_CHURN_SCALE_DAYS)
            activity_weight = round(dist.sample_activity_weight(rng), 3)
            is_active = churn_date is None
            rows.append((
                f"{zone['zone_name']} {cuisine.title()} Kitchen {i + 1}", cuisine, zone_id,
                round(baseline_prep, 2), round(dist.sample_rating(rng), 1), is_active,
                onboarded_date, activation_date, churn_date, activity_weight,
            ))
            zone_order.append((zone["zone_name"], cuisine, baseline_prep))

    with conn.cursor() as cur:
        results = psycopg2.extras.execute_values(
            cur,
            """INSERT INTO dim_merchants (name, cuisine_type, zone_id, baseline_prep_time_min, rating, is_active,
                                           onboarded_date, activation_date, churn_date, activity_weight)
               VALUES %s RETURNING merchant_id""",
            rows, fetch=True,
        )
    conn.commit()

    zone_merchants: dict[str, list[tuple]] = {z["zone_name"]: [] for z in dist.ZONES}
    for (merchant_id,), (zone_name, cuisine, baseline_prep), row in zip(results, zone_order, rows):
        activation_date, churn_date, activity_weight = row[7], row[8], row[9]
        zone_merchants[zone_name].append((activation_date, merchant_id, activity_weight, churn_date, cuisine, baseline_prep))
    for zone_name in zone_merchants:
        zone_merchants[zone_name].sort(key=lambda t: t[0])
    print(f"  inserted {len(rows)} merchants")
    return zone_merchants


def insert_dashers(conn, rng, zone_name_to_id, start_date, end_date) -> dict[str, list[tuple]]:
    """Returns zone_name -> list of (activation_date, dasher_id, activity_weight,
    churn_date) sorted by activation_date."""
    span_days = max(1, (end_date - start_date).days)
    rows = []
    zone_order = []
    for zone in dist.ZONES:
        n = DASHERS_PER_ZONE[zone["density_tier"]]
        zone_id = zone_name_to_id[zone["zone_name"]]
        weights_v = VEHICLE_WEIGHTS[zone["density_tier"]]
        for _ in range(n):
            signup_date = dist.sample_acquisition_date(
                rng, start_date, span_days, seed_fraction=SEED_FRACTION, steepness=SCURVE_STEEPNESS, lag_days=0.0)
            activation_date = min(end_date, signup_date + dt.timedelta(days=int(rng.integers(2, 10))))
            churn_date = dist.sample_churn_date(
                rng, activation_date, end_date, shape=CHURN_SHAPE, scale_days=DASHER_CHURN_SCALE_DAYS)
            activity_weight = round(dist.sample_activity_weight(rng), 3)
            vehicle = str(rng.choice(dist.VEHICLE_TYPES, p=weights_v))
            status = "active" if churn_date is None else "inactive"
            rows.append((
                signup_date, activation_date, churn_date, activity_weight, vehicle, zone_id,
                round(dist.sample_rating(rng), 1), status,
            ))
            zone_order.append(zone["zone_name"])

    with conn.cursor() as cur:
        results = psycopg2.extras.execute_values(
            cur,
            """INSERT INTO dim_dashers (signup_date, activation_date, churn_date, activity_weight,
                                         vehicle_type, home_zone_id, rating, status)
               VALUES %s RETURNING dasher_id""",
            rows, fetch=True,
        )
    conn.commit()

    zone_dashers: dict[str, list[tuple]] = {z["zone_name"]: [] for z in dist.ZONES}
    for (dasher_id,), zone_name, row in zip(results, zone_order, rows):
        _, activation_date, churn_date, activity_weight = row[0], row[1], row[2], row[3]
        zone_dashers[zone_name].append((activation_date, dasher_id, activity_weight, churn_date))
    for zone_name in zone_dashers:
        zone_dashers[zone_name].sort(key=lambda t: t[0])
    print(f"  inserted {len(rows)} dashers")
    return zone_dashers


def insert_promotions(conn, rng, all_merchant_ids, start_date, end_date):
    """Returns (promos_by_merchant, platform_promos) keyed for fast lookup
    during fact generation; each promo dict carries what's needed to compute
    a discount amount."""
    rows = []
    span_days = max(1, (end_date - start_date).days)
    for _ in range(TOTAL_PROMOTIONS):
        is_platform_wide = rng.random() < PLATFORM_WIDE_PROMO_PROBABILITY
        merchant_id = None if is_platform_wide else int(rng.choice(all_merchant_ids))
        promo_type = str(rng.choice(PROMO_TYPES))
        discount_pct = round(float(rng.uniform(10, 40)), 2) if promo_type == "discount_pct" else None
        flat_amount = round(float(rng.uniform(3, 10)), 2) if promo_type == "flat_discount" else None
        promo_start = start_date + dt.timedelta(days=int(rng.integers(0, span_days)))
        duration = int(rng.integers(7, 28))
        promo_end = min(end_date, promo_start + dt.timedelta(days=duration))
        rows.append((promo_type, discount_pct, flat_amount, merchant_id, promo_start, promo_end))

    with conn.cursor() as cur:
        results = psycopg2.extras.execute_values(
            cur,
            """INSERT INTO dim_promotions (promo_type, discount_pct, flat_amount, merchant_id, start_date, end_date)
               VALUES %s RETURNING promo_id""",
            rows, fetch=True,
        )
    conn.commit()

    promos_by_merchant: dict[int, list[dict]] = {}
    platform_promos: list[dict] = []
    for (promo_id,), (promo_type, discount_pct, flat_amount, merchant_id, promo_start, promo_end) in zip(results, rows):
        promo = {
            "promo_id": promo_id, "promo_type": promo_type, "discount_pct": discount_pct,
            "flat_amount": flat_amount, "start_date": promo_start, "end_date": promo_end,
        }
        if merchant_id is None:
            platform_promos.append(promo)
        else:
            promos_by_merchant.setdefault(merchant_id, []).append(promo)
    print(f"  inserted {len(rows)} promotions")
    return promos_by_merchant, platform_promos


def insert_dim_date(conn, start_date, end_date):
    rows = []
    current = start_date
    while current <= end_date:
        for hour in range(24):
            date_key = int(current.strftime("%Y%m%d")) * 100 + hour
            rows.append((
                date_key, current, current.weekday(), current.weekday() >= 5,
                dist.is_holiday(current), current.month, current.year, hour,
                dist.daypart_for_hour(hour),
            ))
        current += dt.timedelta(days=1)

    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            """INSERT INTO dim_date (date_key, full_date, day_of_week, is_weekend, is_holiday, month, year, hour, daypart)
               VALUES %s""",
            rows,
        )
    conn.commit()
    print(f"  inserted {len(rows)} dim_date rows")


def insert_weather_and_events(conn, rng, zone_name_to_id, start_date, end_date):
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
    print(f"  inserted {len(weather_rows)} weather rows")

    events_df = wx.build_external_events(start_date, end_date)
    event_rows = [
        (r["event_type"], r["start_date"], r["end_date"], r["affected_state"], r["demand_multiplier"], r["order_value_multiplier"], r["description"])
        for r in events_df.to_dict("records")
    ]
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            """INSERT INTO dim_external_events (event_type, start_date, end_date, affected_state, demand_multiplier, order_value_multiplier, description)
               VALUES %s""",
            event_rows,
        )
    conn.commit()
    print(f"  inserted {len(event_rows)} external events")

    return wx.weather_lookup(weather_df), events_df


def pick_promo(rng, merchant_id, date, promos_by_merchant, platform_promos):
    candidates = [p for p in promos_by_merchant.get(merchant_id, []) if p["start_date"] <= date <= p["end_date"]]
    candidates += [p for p in platform_promos if p["start_date"] <= date <= p["end_date"]]
    if candidates and rng.random() < PROMO_USE_PROBABILITY:
        return candidates[int(rng.integers(0, len(candidates)))]
    return None


def apply_promo_discount(promo, subtotal) -> tuple[float, bool]:
    """Returns (discount_amount, waives_delivery_fee)."""
    if promo is None:
        return 0.0, False
    if promo["promo_type"] == "discount_pct":
        return round(subtotal * (promo["discount_pct"] / 100.0), 2), False
    if promo["promo_type"] == "flat_discount":
        return round(promo["flat_amount"], 2), False
    if promo["promo_type"] == "free_delivery":
        return 0.0, True
    if promo["promo_type"] == "bogo":
        return round(subtotal * 0.5, 2), False
    return 0.0, False


def _advance_zone_pools(current, zone_state):
    """Grows each pool with newly-activated entities and evicts anyone
    churning today. Pointer-based: O(new activations + churns today), not a
    full rescan, so this stays cheap across a multi-year run."""
    for kind in ("users", "merchants", "dashers"):
        st = zone_state[kind]
        sorted_list = st["sorted"]
        ptr = st["ptr"]
        pool = st["pool"]
        while ptr < len(sorted_list) and sorted_list[ptr][0] <= current:
            entity = sorted_list[ptr]
            activation_date, entity_id, weight, churn_date = entity[0], entity[1], entity[2], entity[3]
            pool.add(entity_id, weight)
            if kind == "merchants":
                st["meta"][entity_id] = {"cuisine_type": entity[4], "baseline_prep_time_min": entity[5]}
            elif kind == "users":
                st["meta"][entity_id] = activation_date
            if churn_date is not None:
                st["churn_by_date"].setdefault(churn_date, []).append(entity_id)
            ptr += 1
        st["ptr"] = ptr

        for entity_id in st["churn_by_date"].pop(current, []):
            pool.remove(entity_id)


def generate_fact_deliveries(conn, rng, zone_name_to_id, zone_users, zone_merchants, zone_dashers,
                              promos_by_merchant, platform_promos, weather_lut, events_df,
                              start_date, end_date):
    buffer = []
    total_inserted = 0

    total_user_weight = {z: sum(t[2] for t in zone_users[z]) for z in zone_users}

    zone_state = {}
    for zone in dist.ZONES:
        zone_name = zone["zone_name"]
        zone_state[zone_name] = {
            "users": {"sorted": zone_users[zone_name], "ptr": 0, "pool": ActivePool(), "churn_by_date": {}, "meta": {}, "history": {}},
            "merchants": {"sorted": zone_merchants[zone_name], "ptr": 0, "pool": ActivePool(), "churn_by_date": {}, "meta": {}},
            "dashers": {"sorted": zone_dashers[zone_name], "ptr": 0, "pool": ActivePool(), "churn_by_date": {}},
        }

    current = start_date
    while current <= end_date:
        for zone in dist.ZONES:
            zone_name = zone["zone_name"]
            zone_id = zone_name_to_id[zone_name]
            density = zone["density_tier"]
            tz = ZoneInfo(zone["timezone"])
            zstate = zone_state[zone_name]
            _advance_zone_pools(current, zstate)

            user_pool = zstate["users"]["pool"]
            merchant_pool = zstate["merchants"]["pool"]
            dasher_pool = zstate["dashers"]["pool"]
            merchant_meta = zstate["merchants"]["meta"]

            if len(user_pool) == 0 or len(merchant_pool) == 0 or len(dasher_pool) == 0:
                continue

            total_weight = total_user_weight[zone_name]
            active_fraction = (user_pool.weight_sum / total_weight) if total_weight > 0 else 0.0

            weather_row = weather_lut.get((zone_name, current))
            weather_demand_mult = 1.0
            travel_mean_mult, travel_sd_mult = 1.0, 1.0
            if weather_row is not None:
                weather_demand_mult = wx.demand_weather_multiplier(
                    weather_row["condition"], weather_row["is_flash_flood"], weather_row["is_heatwave"])
                travel_mean_mult, travel_sd_mult = wx.travel_time_weather_multipliers(
                    weather_row["condition"], weather_row["is_flash_flood"], weather_row["is_heatwave"])

            active_events = wx.active_events_for(events_df, zone["state"], current)
            event_demand_mult, event_value_mult = wx.combined_event_multipliers(active_events)

            hour_orders = []
            for hour in range(24):
                local_dt = dt.datetime(current.year, current.month, current.day, hour, tzinfo=tz)
                weight = dist.hourly_order_weight(local_dt.astimezone(dt.timezone.utc), zone["timezone"])
                expected = BASE_ORDER_RATE[density] * weight * weather_demand_mult * event_demand_mult * active_fraction
                hour_orders.append(int(rng.poisson(expected)))

            day_total = sum(hour_orders)
            if day_total == 0:
                continue

            user_ids_batch = user_pool.pick(rng, day_total)
            merchant_ids_batch = merchant_pool.pick(rng, day_total)
            dasher_ids_batch = dasher_pool.pick(rng, day_total)

            offset = 0
            for hour, n_orders in enumerate(hour_orders):
                if n_orders == 0:
                    continue
                local_dt = dt.datetime(current.year, current.month, current.day, hour, tzinfo=tz)
                daypart = dist.daypart_for_hour(hour)

                for _ in range(n_orders):
                    user_id = user_ids_batch[offset]
                    merchant_id = merchant_ids_batch[offset]
                    dasher_id = dasher_ids_batch[offset]
                    offset += 1
                    merchant = merchant_meta[merchant_id]

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

                    user_history = zstate["users"]["history"].setdefault(user_id, UserHistory())
                    trailing_count, trailing_gmv = user_history.snapshot(current)
                    tenure_days = (current - zstate["users"]["meta"][user_id]).days
                    loyalty_score = dist.compute_loyalty_score(tenure_days, trailing_count, trailing_gmv)

                    weather_issue_mult = 1.0
                    if weather_row is not None:
                        weather_issue_mult = wx.issue_rate_multiplier(
                            weather_row["condition"], weather_row["is_flash_flood"], weather_row["is_heatwave"])
                    issue_type = dist.sample_order_issue(rng, weather_issue_mult)
                    refund_amount = dist.compute_refund_amount(issue_type, total_amount, loyalty_score) if issue_type else 0.0
                    user_history.record(current, total_amount)

                    buffer.append((
                        user_id, merchant_id, dasher_id, zone_id,
                        promo["promo_id"] if promo else None, date_key, order_ts,
                        promised_eta_min, actual_delivery_min, round(prep_time, 2), round(pickup_wait, 2),
                        round(travel_time, 2), round(subtotal, 2), discount_amount, delivery_fee, tip_amount,
                        total_amount, item_count, is_late, issue_type, refund_amount, "historical",
                    ))

                    if len(buffer) >= BATCH_SIZE:
                        with conn.cursor() as cur:
                            psycopg2.extras.execute_values(cur, FACT_DELIVERIES_INSERT, buffer)
                        conn.commit()
                        total_inserted += len(buffer)
                        buffer.clear()

        if current.day == 1:
            print(f"  ... processed through {current.isoformat()} ({total_inserted:,} deliveries so far)")
        current += dt.timedelta(days=1)

    if buffer:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(cur, FACT_DELIVERIES_INSERT, buffer)
        conn.commit()
        total_inserted += len(buffer)

    print(f"  inserted {total_inserted:,} fact_deliveries rows total")


def main():
    parser = argparse.ArgumentParser(description="Seed historical data for the food delivery marketplace.")
    parser.add_argument("--years", type=int, default=3, help="Historical window length in years (default 3)")
    parser.add_argument("--start-date", type=str, default=None, help="Override window start (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None, help="Override window end (YYYY-MM-DD), default today")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for reproducibility")
    args = parser.parse_args()

    end_date = dt.date.fromisoformat(args.end_date) if args.end_date else dt.date.today()
    start_date = dt.date.fromisoformat(args.start_date) if args.start_date else end_date - dt.timedelta(days=365 * args.years)

    rng = np.random.default_rng(args.seed)
    conn = get_connection()

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM dim_zones")
        if cur.fetchone()[0] > 0:
            print("dim_zones is already populated — refusing to re-seed and create duplicates.")
            print("Reset the database first (e.g. `docker-compose down -v && docker-compose up -d`) and retry.")
            conn.close()
            sys.exit(1)

    print(f"Seeding historical data from {start_date} to {end_date}...")

    print("Inserting dimensions...")
    zone_name_to_id = insert_zones(conn)
    zone_users = insert_users(conn, rng, zone_name_to_id, start_date, end_date)
    zone_merchants = insert_merchants(conn, rng, zone_name_to_id, start_date, end_date)
    zone_dashers = insert_dashers(conn, rng, zone_name_to_id, start_date, end_date)
    all_merchant_ids = [m[1] for merchants in zone_merchants.values() for m in merchants]
    promos_by_merchant, platform_promos = insert_promotions(conn, rng, all_merchant_ids, start_date, end_date)
    insert_dim_date(conn, start_date, end_date)

    print("Inserting weather + external events...")
    weather_lut, events_df = insert_weather_and_events(conn, rng, zone_name_to_id, start_date, end_date)

    print("Generating fact_deliveries (this is the long-running step)...")
    generate_fact_deliveries(
        conn, rng, zone_name_to_id, zone_users, zone_merchants, zone_dashers,
        promos_by_merchant, platform_promos, weather_lut, events_df, start_date, end_date,
    )

    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()

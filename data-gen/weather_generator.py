"""Simulated daily weather per zone, plus a hand-curated set of external
macro/geopolitical events. Both feed into demand and travel-time multipliers
that distributions.py-based order generation applies on top of the base
time-of-day/weekend/holiday weighting."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

# Rough seasonal rain-probability multiplier by month (wetter in
# spring/fall, drier in mid-summer/mid-winter extremes) — approximate, not
# region-specific, since this is illustrative synthetic data.
MONTH_RAIN_MULTIPLIER = {1: 1.2, 2: 1.15, 3: 1.1, 4: 1.05, 5: 0.95, 6: 0.85,
                         7: 0.8, 8: 0.85, 9: 0.95, 10: 1.0, 11: 1.15, 12: 1.2}

BASE_RAIN_PROB = 0.25


def _weather_for_zone_day(zone: dict, date: dt.date, rng: np.random.Generator) -> dict:
    lat = zone["latitude"]
    day_of_year = date.timetuple().tm_yday

    # Annual temperature sinusoid: warmer/less seasonal swing near the
    # equator (lower latitude), colder/bigger swing further north. Peaks
    # around day 200 (mid-summer).
    base_mean = 28 - (lat * 0.55)
    amplitude = 8 + (lat * 0.35)
    temp_mid = float(base_mean + amplitude * np.cos(2 * np.pi * (day_of_year - 200) / 365.0) + rng.normal(0, 3))
    temp_high = float(temp_mid + abs(rng.normal(4, 1.5)))
    temp_low = float(temp_mid - abs(rng.normal(4, 1.5)))

    rain_prob = BASE_RAIN_PROB * MONTH_RAIN_MULTIPLIER[date.month]
    rains = rng.random() < rain_prob
    precipitation_mm = float(rng.gamma(2.0, 6.0)) if rains else 0.0
    is_flash_flood = bool(precipitation_mm > 45)
    wind_speed_kmh = float(max(0.0, rng.gamma(2.5, 6.0)))

    if temp_low < 0 and precipitation_mm > 0:
        condition = "snow"
    elif is_flash_flood or precipitation_mm > 25:
        condition = "storm"
    elif precipitation_mm > 0:
        condition = "rain"
    else:
        condition = "clear"

    is_heatwave = bool(date.month in (6, 7, 8, 9) and temp_high > (base_mean + amplitude * 1.15))

    return {
        "zone_name": zone["zone_name"],
        "date": date,
        "precipitation_mm": round(precipitation_mm, 2),
        "temp_high_c": round(temp_high, 2),
        "temp_low_c": round(temp_low, 2),
        "wind_speed_kmh": round(wind_speed_kmh, 2),
        "condition": condition,
        "is_flash_flood": is_flash_flood,
        "is_heatwave": is_heatwave,
    }


def generate_weather_series(zones: list[dict], start_date: dt.date, end_date: dt.date,
                             rng: np.random.Generator) -> pd.DataFrame:
    """Full daily weather time series for every zone across [start_date, end_date]."""
    rows = []
    current = start_date
    while current <= end_date:
        for zone in zones:
            rows.append(_weather_for_zone_day(zone, current, rng))
        current += dt.timedelta(days=1)
    return pd.DataFrame(rows)


def generate_weather_for_date(zones: list[dict], date: dt.date, rng: np.random.Generator) -> pd.DataFrame:
    """Single-day variant used by the realtime generator to extend the
    weather series forward as the accelerated clock advances."""
    return pd.DataFrame([_weather_for_zone_day(zone, date, rng) for zone in zones])


def weather_lookup(weather_df: pd.DataFrame) -> dict[tuple[str, dt.date], dict]:
    """O(1) lookup by (zone_name, date) for use during row-by-row order generation."""
    return {(r["zone_name"], r["date"]): r for r in weather_df.to_dict("records")}


def build_external_events(window_start: dt.date, window_end: dt.date) -> pd.DataFrame:
    """Hand-curated macro/geopolitical events spread across the historical
    window (offsets expressed as fractions of the window, so this stays
    correct regardless of the caller's actual start/end dates)."""
    total_days = (window_end - window_start).days

    def _window(offset_ratio: float, duration_days: int) -> tuple[dt.date, dt.date]:
        start = window_start + dt.timedelta(days=int(total_days * offset_ratio))
        return start, start + dt.timedelta(days=duration_days)

    specs = [
        # (event_type, offset_ratio, duration_days, affected_state, demand_mult, order_value_mult, description)
        ("economic_downturn", 0.05, 90, None, 1.12, 0.90, "Regional economic downturn increases delivery frequency for cheaper at-home meals while lowering average order value."),
        ("fuel_price_spike", 0.12, 45, None, 1.08, 1.05, "National fuel price spike pushes more consumers toward delivery over driving."),
        ("transit_strike", 0.18, 10, "NY", 1.35, 1.02, "New York City transit strike sharply increases delivery demand as commuting options shrink."),
        ("winter_storm_emergency", 0.22, 5, "MA", 1.25, 1.10, "Major winter storm emergency in the Northeast keeps residents home."),
        ("regional_flooding_emergency", 0.27, 7, "TX", 1.30, 1.08, "Flash flooding emergency in Texas Gulf Coast metros disrupts normal errands."),
        ("major_sporting_event", 0.31, 3, "CA", 1.40, 1.15, "Championship game weekend in California drives a large surge in group orders."),
        ("heatwave_advisory", 0.36, 12, "AZ", 1.20, 1.05, "Extended heatwave advisory in the Southwest keeps residents indoors."),
        ("gas_shortage", 0.41, 20, "GA", 1.15, 1.03, "Regional fuel supply disruption in the Southeast increases delivery reliance."),
        ("geopolitical_unrest_oil_shock", 0.46, 60, None, 1.10, 1.12, "Geopolitical unrest drives a global oil price shock, raising delivery fees and order values nationwide."),
        ("public_health_advisory", 0.50, 30, None, 1.18, 0.95, "Regional public health advisory encourages residents to stay home."),
        ("transit_strike_2", 0.55, 8, "IL", 1.30, 1.02, "Chicago public transit strike increases delivery demand."),
        ("hurricane_landfall", 0.60, 6, "FL", 1.28, 1.10, "Hurricane landfall in Florida disrupts normal commerce, spiking delivery demand around the storm."),
        ("economic_recovery_boom", 0.65, 75, None, 0.92, 1.10, "Broad economic recovery raises average order values as discretionary spending rebounds."),
        ("wildfire_smoke_advisory", 0.70, 14, "CA", 1.22, 1.04, "Wildfire smoke air-quality advisory in California keeps residents indoors."),
        ("major_snowstorm", 0.75, 4, "IL", 1.32, 1.08, "Major snowstorm blankets the Midwest, spiking delivery demand."),
        ("fuel_price_spike_2", 0.80, 40, None, 1.09, 1.06, "A second national fuel price spike increases delivery-app reliance."),
        ("transit_strike_3", 0.85, 9, "PA", 1.27, 1.02, "Philadelphia transit strike increases delivery demand."),
        ("regional_flooding_emergency_2", 0.90, 6, "NC", 1.28, 1.07, "Flash flooding emergency in the Carolinas disrupts normal errands."),
        ("holiday_shopping_surge", 0.95, 20, None, 1.15, 1.12, "End-of-year holiday season broadly increases both order frequency and order values."),
    ]

    events = []
    for event_type, offset_ratio, duration, state, dmult, omult, desc in specs:
        start, end = _window(offset_ratio, duration)
        events.append({
            "event_type": event_type, "start_date": start, "end_date": end,
            "affected_state": state, "demand_multiplier": dmult,
            "order_value_multiplier": omult, "description": desc,
        })
    return pd.DataFrame(events)


def active_events_for(events_df: pd.DataFrame, state: str | None, date: dt.date) -> list[dict]:
    mask = (
        (events_df["start_date"] <= date)
        & (events_df["end_date"] >= date)
        & (events_df["affected_state"].isna() | (events_df["affected_state"] == state))
    )
    return events_df[mask].to_dict("records")


def combined_event_multipliers(events: list[dict]) -> tuple[float, float]:
    """Compose active events into a single (demand_multiplier, order_value_multiplier) pair."""
    demand_mult, value_mult = 1.0, 1.0
    for event in events:
        demand_mult *= event["demand_multiplier"]
        value_mult *= event["order_value_multiplier"]
    return demand_mult, value_mult


def demand_weather_multiplier(condition: str, is_flash_flood: bool, is_heatwave: bool) -> float:
    """Bad weather increases delivery demand (people avoid going out)."""
    mult = {"clear": 1.0, "rain": 1.15, "storm": 1.25, "snow": 1.20}[condition]
    if is_flash_flood:
        mult *= 1.15
    if is_heatwave:
        mult *= 1.12
    return mult


def issue_rate_multiplier(condition: str, is_flash_flood: bool, is_heatwave: bool) -> float:
    """Bad weather increases the chance of misplaced/lost/wrong deliveries
    (more chaotic conditions for dashers navigating and matching orders)."""
    mult = {"clear": 1.0, "rain": 1.2, "storm": 1.5, "snow": 1.6}[condition]
    if is_flash_flood:
        mult *= 1.4
    if is_heatwave:
        mult *= 1.05
    return mult


def travel_time_weather_multipliers(condition: str, is_flash_flood: bool, is_heatwave: bool) -> tuple[float, float]:
    """(mean_multiplier, stddev_multiplier) for travel time under adverse weather."""
    mean_mult, sd_mult = {
        "clear": (1.0, 1.0), "rain": (1.15, 1.3), "storm": (1.35, 1.6), "snow": (1.5, 1.8),
    }[condition]
    if is_flash_flood:
        mean_mult *= 1.25
        sd_mult *= 1.4
    if is_heatwave:
        mean_mult *= 1.08  # dasher fatigue rather than traffic-driven
        sd_mult *= 1.1
    return mean_mult, sd_mult

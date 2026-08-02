"""Shared demand/seasonality distributions used by both the historical and
realtime generators, so the two produce statistically consistent data."""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np

# ~40 real US metro zones spanning ~25+ states across all 9 Census regions,
# each tagged with state/region/timezone so demand curves peak at each
# zone's *local* time rather than a shared UTC hour.
ZONES = [
    # New England
    {"zone_name": "Boston",          "city": "Boston",          "state": "MA", "region": "new_england",         "timezone": "America/New_York",           "density_tier": "high",   "latitude": 42.3601, "longitude": -71.0589},
    {"zone_name": "Providence",      "city": "Providence",      "state": "RI", "region": "new_england",         "timezone": "America/New_York",           "density_tier": "medium", "latitude": 41.8240, "longitude": -71.4128},
    {"zone_name": "Hartford",        "city": "Hartford",        "state": "CT", "region": "new_england",         "timezone": "America/New_York",           "density_tier": "low",    "latitude": 41.7658, "longitude": -72.6734},
    # Mid-Atlantic
    {"zone_name": "New York",        "city": "New York",        "state": "NY", "region": "mid_atlantic",        "timezone": "America/New_York",           "density_tier": "high",   "latitude": 40.7128, "longitude": -74.0060},
    {"zone_name": "Philadelphia",    "city": "Philadelphia",    "state": "PA", "region": "mid_atlantic",        "timezone": "America/New_York",           "density_tier": "high",   "latitude": 39.9526, "longitude": -75.1652},
    {"zone_name": "Newark",          "city": "Newark",          "state": "NJ", "region": "mid_atlantic",        "timezone": "America/New_York",           "density_tier": "medium", "latitude": 40.7357, "longitude": -74.1724},
    {"zone_name": "Pittsburgh",      "city": "Pittsburgh",      "state": "PA", "region": "mid_atlantic",        "timezone": "America/New_York",           "density_tier": "medium", "latitude": 40.4406, "longitude": -79.9959},
    # East North Central
    {"zone_name": "Chicago",         "city": "Chicago",         "state": "IL", "region": "east_north_central",  "timezone": "America/Chicago",             "density_tier": "high",   "latitude": 41.8781, "longitude": -87.6298},
    {"zone_name": "Detroit",         "city": "Detroit",         "state": "MI", "region": "east_north_central",  "timezone": "America/Detroit",             "density_tier": "medium", "latitude": 42.3314, "longitude": -83.0458},
    {"zone_name": "Columbus",        "city": "Columbus",        "state": "OH", "region": "east_north_central",  "timezone": "America/New_York",           "density_tier": "medium", "latitude": 39.9612, "longitude": -82.9988},
    {"zone_name": "Indianapolis",    "city": "Indianapolis",    "state": "IN", "region": "east_north_central",  "timezone": "America/Indiana/Indianapolis", "density_tier": "medium", "latitude": 39.7684, "longitude": -86.1581},
    {"zone_name": "Milwaukee",       "city": "Milwaukee",       "state": "WI", "region": "east_north_central",  "timezone": "America/Chicago",             "density_tier": "medium", "latitude": 43.0389, "longitude": -87.9065},
    # West North Central
    {"zone_name": "Minneapolis",     "city": "Minneapolis",     "state": "MN", "region": "west_north_central",  "timezone": "America/Chicago",             "density_tier": "medium", "latitude": 44.9778, "longitude": -93.2650},
    {"zone_name": "Kansas City",     "city": "Kansas City",     "state": "MO", "region": "west_north_central",  "timezone": "America/Chicago",             "density_tier": "medium", "latitude": 39.0997, "longitude": -94.5786},
    {"zone_name": "Omaha",           "city": "Omaha",           "state": "NE", "region": "west_north_central",  "timezone": "America/Chicago",             "density_tier": "low",    "latitude": 41.2565, "longitude": -95.9345},
    {"zone_name": "Des Moines",      "city": "Des Moines",      "state": "IA", "region": "west_north_central",  "timezone": "America/Chicago",             "density_tier": "low",    "latitude": 41.5868, "longitude": -93.6250},
    # South Atlantic
    {"zone_name": "Atlanta",         "city": "Atlanta",         "state": "GA", "region": "south_atlantic",       "timezone": "America/New_York",           "density_tier": "high",   "latitude": 33.7490, "longitude": -84.3880},
    {"zone_name": "Miami",           "city": "Miami",           "state": "FL", "region": "south_atlantic",       "timezone": "America/New_York",           "density_tier": "high",   "latitude": 25.7617, "longitude": -80.1918},
    {"zone_name": "Charlotte",       "city": "Charlotte",       "state": "NC", "region": "south_atlantic",       "timezone": "America/New_York",           "density_tier": "medium", "latitude": 35.2271, "longitude": -80.8431},
    {"zone_name": "Washington",      "city": "Washington",      "state": "DC", "region": "south_atlantic",       "timezone": "America/New_York",           "density_tier": "high",   "latitude": 38.9072, "longitude": -77.0369},
    {"zone_name": "Baltimore",       "city": "Baltimore",       "state": "MD", "region": "south_atlantic",       "timezone": "America/New_York",           "density_tier": "medium", "latitude": 39.2904, "longitude": -76.6122},
    {"zone_name": "Virginia Beach",  "city": "Virginia Beach",  "state": "VA", "region": "south_atlantic",       "timezone": "America/New_York",           "density_tier": "low",    "latitude": 36.8529, "longitude": -75.9780},
    {"zone_name": "Charleston",      "city": "Charleston",      "state": "SC", "region": "south_atlantic",       "timezone": "America/New_York",           "density_tier": "low",    "latitude": 32.7765, "longitude": -79.9311},
    # East South Central
    {"zone_name": "Nashville",       "city": "Nashville",       "state": "TN", "region": "east_south_central",  "timezone": "America/Chicago",             "density_tier": "medium", "latitude": 36.1627, "longitude": -86.7816},
    {"zone_name": "Memphis",         "city": "Memphis",         "state": "TN", "region": "east_south_central",  "timezone": "America/Chicago",             "density_tier": "low",    "latitude": 35.1495, "longitude": -90.0490},
    {"zone_name": "Louisville",      "city": "Louisville",      "state": "KY", "region": "east_south_central",  "timezone": "America/New_York",           "density_tier": "low",    "latitude": 38.2527, "longitude": -85.7585},
    {"zone_name": "Birmingham",      "city": "Birmingham",      "state": "AL", "region": "east_south_central",  "timezone": "America/Chicago",             "density_tier": "low",    "latitude": 33.5186, "longitude": -86.8104},
    # West South Central
    {"zone_name": "Dallas",          "city": "Dallas",          "state": "TX", "region": "west_south_central",  "timezone": "America/Chicago",             "density_tier": "high",   "latitude": 32.7767, "longitude": -96.7970},
    {"zone_name": "Houston",         "city": "Houston",         "state": "TX", "region": "west_south_central",  "timezone": "America/Chicago",             "density_tier": "high",   "latitude": 29.7604, "longitude": -95.3698},
    {"zone_name": "Austin",          "city": "Austin",          "state": "TX", "region": "west_south_central",  "timezone": "America/Chicago",             "density_tier": "medium", "latitude": 30.2672, "longitude": -97.7431},
    {"zone_name": "Oklahoma City",   "city": "Oklahoma City",   "state": "OK", "region": "west_south_central",  "timezone": "America/Chicago",             "density_tier": "low",    "latitude": 35.4676, "longitude": -97.5164},
    {"zone_name": "New Orleans",     "city": "New Orleans",     "state": "LA", "region": "west_south_central",  "timezone": "America/Chicago",             "density_tier": "medium", "latitude": 29.9511, "longitude": -90.0715},
    # Mountain
    {"zone_name": "Denver",          "city": "Denver",          "state": "CO", "region": "mountain",             "timezone": "America/Denver",              "density_tier": "high",   "latitude": 39.7392, "longitude": -104.9903},
    {"zone_name": "Phoenix",         "city": "Phoenix",         "state": "AZ", "region": "mountain",             "timezone": "America/Phoenix",             "density_tier": "high",   "latitude": 33.4484, "longitude": -112.0740},
    {"zone_name": "Salt Lake City",  "city": "Salt Lake City",  "state": "UT", "region": "mountain",             "timezone": "America/Denver",              "density_tier": "medium", "latitude": 40.7608, "longitude": -111.8910},
    {"zone_name": "Albuquerque",     "city": "Albuquerque",     "state": "NM", "region": "mountain",             "timezone": "America/Denver",              "density_tier": "low",    "latitude": 35.0844, "longitude": -106.6504},
    {"zone_name": "Boise",           "city": "Boise",           "state": "ID", "region": "mountain",             "timezone": "America/Boise",               "density_tier": "low",    "latitude": 43.6150, "longitude": -116.2023},
    # Pacific
    {"zone_name": "Los Angeles",     "city": "Los Angeles",     "state": "CA", "region": "pacific",              "timezone": "America/Los_Angeles",         "density_tier": "high",   "latitude": 34.0522, "longitude": -118.2437},
    {"zone_name": "San Francisco",   "city": "San Francisco",   "state": "CA", "region": "pacific",              "timezone": "America/Los_Angeles",         "density_tier": "high",   "latitude": 37.7749, "longitude": -122.4194},
    {"zone_name": "Seattle",         "city": "Seattle",         "state": "WA", "region": "pacific",              "timezone": "America/Los_Angeles",         "density_tier": "high",   "latitude": 47.6062, "longitude": -122.3321},
    {"zone_name": "Portland",        "city": "Portland",        "state": "OR", "region": "pacific",              "timezone": "America/Los_Angeles",         "density_tier": "medium", "latitude": 45.5152, "longitude": -122.6784},
    {"zone_name": "San Diego",       "city": "San Diego",       "state": "CA", "region": "pacific",              "timezone": "America/Los_Angeles",         "density_tier": "medium", "latitude": 32.7157, "longitude": -117.1611},
]

CUISINES = ["american", "italian", "chinese", "mexican", "indian", "japanese", "thai", "pizza", "burgers", "cafe"]

VEHICLE_TYPES = ["bike", "scooter", "car"]

ACQUISITION_CHANNELS = ["organic", "paid_search", "referral", "social", "push_notification"]

# Illustrative fixed-date US holidays (not exhaustive) used to bump is_holiday / demand.
HOLIDAYS = {(1, 1), (7, 4), (11, 11), (12, 25), (12, 31)}

# Relative order-volume weight by *local* hour of day (index 0 = midnight),
# tuned for breakfast/lunch/dinner/late-night peaks typical of food delivery.
HOURLY_WEIGHTS = np.array([
    0.2, 0.1, 0.1, 0.1, 0.1, 0.2, 0.4, 0.6,   # 0-7
    0.8, 0.6, 0.5, 0.9, 1.6, 1.4, 0.8, 0.6,   # 8-15
    0.7, 1.0, 1.7, 2.0, 1.6, 1.1, 0.8, 0.5,   # 16-23
])

WEEKEND_MULTIPLIER = 1.35


def local_datetime(utc_dt: dt.datetime, timezone: str) -> dt.datetime:
    """Convert a UTC-aware timestamp to the zone's local time, since demand
    curves must peak at each zone's local lunch/dinner hour, not a shared UTC hour."""
    return utc_dt.astimezone(ZoneInfo(timezone))


def daypart_for_hour(hour: int) -> str:
    if 6 <= hour < 11:
        return "breakfast"
    if 11 <= hour < 15:
        return "lunch"
    if 15 <= hour < 21:
        return "dinner"
    return "late_night"


def is_holiday(date: dt.date) -> bool:
    return (date.month, date.day) in HOLIDAYS


def hourly_order_weight(utc_dt: dt.datetime, timezone: str) -> float:
    """Relative expected order volume for a given UTC timestamp in a given
    zone, combining local time-of-day, weekend, and holiday effects."""
    local_dt = local_datetime(utc_dt, timezone)
    weight = HOURLY_WEIGHTS[local_dt.hour]
    if local_dt.weekday() >= 5:  # Saturday/Sunday
        weight *= WEEKEND_MULTIPLIER
    if is_holiday(local_dt.date()):
        weight *= 1.5
    return weight


def zone_density_multiplier(density_tier: str) -> float:
    return {"high": 1.8, "medium": 1.0, "low": 0.5}[density_tier]


def sample_prep_time_min(rng: np.random.Generator, baseline_prep_time_min: float, daypart: str) -> float:
    """Gamma distribution: prep time is always positive and right-skewed
    (occasional slow kitchens), with a peak-hours penalty."""
    peak_multiplier = 1.25 if daypart in ("lunch", "dinner") else 1.0
    shape, scale = 9.0, (baseline_prep_time_min * peak_multiplier) / 9.0
    return float(rng.gamma(shape, scale))


def sample_travel_time_min(rng: np.random.Generator, density_tier: str,
                            weather_mean_multiplier: float = 1.0,
                            weather_sd_multiplier: float = 1.0) -> float:
    """Denser zones: shorter distances but more traffic variance.
    Sparser zones: longer distances, more consistent travel time.
    Weather multipliers (from weather_generator.travel_time_weather_multipliers)
    stretch both the mean and variance under adverse conditions."""
    base = {"high": 12.0, "medium": 18.0, "low": 26.0}[density_tier] * weather_mean_multiplier
    noise_sd = {"high": 6.0, "medium": 4.0, "low": 3.0}[density_tier] * weather_sd_multiplier
    return float(max(2.0, rng.normal(base, noise_sd)))


def sample_pickup_wait_min(rng: np.random.Generator) -> float:
    return float(max(0.0, rng.normal(4.0, 2.5)))


def sample_order_value(rng: np.random.Generator, cuisine_type: str) -> float:
    """Lognormal: order subtotals are right-skewed (most orders cluster
    low-to-mid, occasional large group orders)."""
    cuisine_mean = {
        "pizza": 24.0, "burgers": 18.0, "cafe": 14.0, "chinese": 26.0,
        "mexican": 20.0, "italian": 30.0, "indian": 28.0, "japanese": 32.0,
        "thai": 24.0, "american": 22.0,
    }.get(cuisine_type, 22.0)
    mu = np.log(cuisine_mean) - 0.5 * (0.35 ** 2)
    return float(rng.lognormal(mu, 0.35))


def sample_tip_amount(rng: np.random.Generator, subtotal: float) -> float:
    tip_pct = float(max(0.0, rng.normal(0.15, 0.07)))
    return round(subtotal * tip_pct, 2)


def sample_item_count(rng: np.random.Generator) -> int:
    return int(rng.poisson(2.2)) + 1


def sample_rating(rng: np.random.Generator) -> float:
    return float(np.clip(rng.normal(4.5, 0.35), 1.0, 5.0))


def sample_scurve_date(rng: np.random.Generator, start_date: dt.date, span_days: int,
                        seed_fraction: float = 0.3, steepness: float = 6.0,
                        lag_days: float = 0.0) -> dt.date:
    """Inverse-samples an acquisition date from a logistic growth curve:
    seed_fraction of the eventual population already exists at day 0,
    saturating by ~day span_days. lag_days shifts the whole curve later —
    used so user acquisition trails merchant/dasher acquisition per zone."""
    k = steepness / span_days
    t0 = np.log(1.0 / seed_fraction - 1.0) / k + lag_days
    u = rng.uniform(1e-6, 1.0 - 1e-6)
    t = t0 - np.log(1.0 / u - 1.0) / k
    t = float(np.clip(t, 0.0, span_days))
    return start_date + dt.timedelta(days=int(t))


def sample_acquisition_date(rng: np.random.Generator, start_date: dt.date, span_days: int,
                             seed_fraction: float = 0.3, steepness: float = 6.0,
                             lag_days: float = 0.0, trickle_fraction: float = 0.25) -> dt.date:
    """Acquisition timing as a mixture: most entities join via the initial
    S-curve adoption wave (sample_scurve_date), but a steady trickle keeps
    joining uniformly across the whole window — representing ongoing
    marketing/organic growth that doesn't stop once the initial wave
    saturates. Without this, churn has nothing to replenish it and the
    active population terminally declines after the S-curve saturates."""
    if rng.random() < trickle_fraction:
        return start_date + dt.timedelta(days=int(rng.integers(0, span_days + 1)))
    return sample_scurve_date(rng, start_date, span_days, seed_fraction, steepness, lag_days)


def sample_churn_date(rng: np.random.Generator, signup_date: dt.date, end_date: dt.date,
                       shape: float = 0.75, scale_days: float = 600.0) -> dt.date | None:
    """Weibull hazard tenure sampling: shape < 1 gives a decreasing hazard
    rate (highest churn risk early, tapering off for long-tenured entities) —
    the standard shape for customer retention curves. Returns None if the
    sampled tenure runs past end_date (still active as of the window's end)."""
    tenure_days = scale_days * float(rng.weibull(shape))
    churn_date = signup_date + dt.timedelta(days=int(tenure_days))
    return churn_date if churn_date <= end_date else None


def sample_activity_weight(rng: np.random.Generator, sigma: float = 0.6) -> float:
    """Fixed per-entity heterogeneity multiplier: lognormal with median 1.0,
    right-skewed so a few entities are much more active/popular than most."""
    return float(rng.lognormal(mean=0.0, sigma=sigma))


# Order issues: misplaced/lost, never delivered, or the wrong order arriving.
# Refund policy is base-percentage-by-severity (not_delivered highest, since
# the customer got nothing; wrong_order lowest, since they at least got food)
# plus a modest loyalty-based bonus, floored at $5 and capped at order value.
ISSUE_BASE_PROBABILITIES = {"misplaced": 0.01, "not_delivered": 0.01, "wrong_order": 0.02}
ISSUE_REFUND_BASE_PCT = {"misplaced": 0.90, "not_delivered": 1.00, "wrong_order": 0.50}
LOYALTY_REFUND_BONUS = 0.10


def sample_order_issue(rng: np.random.Generator, weather_issue_multiplier: float = 1.0) -> str | None:
    """Returns an issue type with small probability (bumped under adverse
    weather via weather_issue_multiplier), or None if the order had no issue."""
    u = rng.random()
    cumulative = 0.0
    for issue_type, base_prob in ISSUE_BASE_PROBABILITIES.items():
        cumulative += base_prob * weather_issue_multiplier
        if u < cumulative:
            return issue_type
    return None


def compute_loyalty_score(tenure_days: float, trailing_order_count: int, trailing_gmv: float) -> float:
    """Blends account tenure, trailing order frequency, and trailing spend
    into a 0-1 loyalty score used to nudge refund generosity."""
    tenure_component = min(1.0, tenure_days / 365.0)
    frequency_component = min(1.0, trailing_order_count / 20.0)
    gmv_component = min(1.0, trailing_gmv / 500.0)
    return (tenure_component + frequency_component + gmv_component) / 3.0


def compute_refund_amount(issue_type: str, order_total: float, loyalty_score: float) -> float:
    base_pct = ISSUE_REFUND_BASE_PCT[issue_type]
    refund_pct = min(1.0, base_pct + LOYALTY_REFUND_BONUS * loyalty_score)
    return round(min(order_total, max(5.0, order_total * refund_pct)), 2)

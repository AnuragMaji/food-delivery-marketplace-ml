-- ============================================================
-- Dimension tables
-- ============================================================

CREATE TABLE dim_zones (
    zone_id      SERIAL PRIMARY KEY,
    zone_name    TEXT NOT NULL,
    city         TEXT NOT NULL,
    state        TEXT NOT NULL,
    region       TEXT NOT NULL CHECK (region IN (
                     'new_england', 'mid_atlantic', 'east_north_central', 'west_north_central',
                     'south_atlantic', 'east_south_central', 'west_south_central', 'mountain', 'pacific'
                 )),
    timezone     TEXT NOT NULL,  -- IANA name, e.g. 'America/New_York'; local hour drives demand curves
    density_tier TEXT NOT NULL CHECK (density_tier IN ('low', 'medium', 'high')),
    latitude     NUMERIC(9, 6),
    longitude    NUMERIC(9, 6)
);

CREATE TABLE dim_users (
    user_id             SERIAL PRIMARY KEY,
    signup_date         DATE NOT NULL,
    activation_date     DATE NOT NULL,  -- short lag after signup before the user starts actually ordering
    churn_date          DATE,           -- NULL = still active as of the observation window's end
    activity_weight     NUMERIC(6, 3) NOT NULL DEFAULT 1.0,  -- fixed per-user heterogeneity multiplier (order frequency)
    home_zone_id        INT NOT NULL REFERENCES dim_zones(zone_id),
    acquisition_channel TEXT,
    is_subscriber       BOOLEAN NOT NULL DEFAULT FALSE,
    status              TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'inactive'))
);

CREATE TABLE dim_merchants (
    merchant_id            SERIAL PRIMARY KEY,
    name                    TEXT NOT NULL,
    cuisine_type            TEXT NOT NULL,
    zone_id                 INT NOT NULL REFERENCES dim_zones(zone_id),
    baseline_prep_time_min  NUMERIC(5, 2) NOT NULL,
    rating                  NUMERIC(2, 1) CHECK (rating BETWEEN 1.0 AND 5.0),
    is_active               BOOLEAN NOT NULL DEFAULT TRUE,
    onboarded_date          DATE NOT NULL,
    activation_date         DATE NOT NULL,  -- short vetting lag after onboarding before the merchant goes live
    churn_date              DATE,           -- NULL = still active as of the observation window's end
    activity_weight         NUMERIC(6, 3) NOT NULL DEFAULT 1.0  -- fixed per-merchant popularity/selection-weight multiplier
);

CREATE TABLE dim_dashers (
    dasher_id       SERIAL PRIMARY KEY,
    signup_date     DATE NOT NULL,
    activation_date DATE NOT NULL,  -- short vetting lag after signup before the dasher goes live
    churn_date      DATE,           -- NULL = still active as of the observation window's end
    activity_weight NUMERIC(6, 3) NOT NULL DEFAULT 1.0,  -- fixed per-dasher capacity/selection-weight multiplier
    vehicle_type    TEXT NOT NULL CHECK (vehicle_type IN ('bike', 'scooter', 'car')),
    home_zone_id    INT NOT NULL REFERENCES dim_zones(zone_id),
    rating          NUMERIC(2, 1) CHECK (rating BETWEEN 1.0 AND 5.0),
    status          TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'inactive'))
);

-- merchant_id NULL means the promotion is platform-wide rather than merchant-specific
CREATE TABLE dim_promotions (
    promo_id      SERIAL PRIMARY KEY,
    promo_type    TEXT NOT NULL CHECK (promo_type IN ('discount_pct', 'flat_discount', 'free_delivery', 'bogo')),
    discount_pct  NUMERIC(4, 2),
    flat_amount   NUMERIC(6, 2),
    merchant_id   INT REFERENCES dim_merchants(merchant_id),
    start_date    DATE NOT NULL,
    end_date      DATE NOT NULL
);

-- Hourly grain (date_key format YYYYMMDDHH) so daypart/hour can live on the same row
CREATE TABLE dim_date (
    date_key     BIGINT PRIMARY KEY,
    full_date    DATE NOT NULL,
    day_of_week  SMALLINT NOT NULL,
    is_weekend   BOOLEAN NOT NULL,
    is_holiday   BOOLEAN NOT NULL DEFAULT FALSE,
    month        SMALLINT NOT NULL,
    year         SMALLINT NOT NULL,
    hour         SMALLINT NOT NULL,
    daypart      TEXT NOT NULL CHECK (daypart IN ('breakfast', 'lunch', 'dinner', 'late_night'))
);

-- ============================================================
-- External context: weather + macro/geopolitical events
-- ============================================================

-- Grain: one row per zone per day. Joined to fact_deliveries via zone_id +
-- date(order_ts) rather than a stored FK, since it's looked up at generation
-- time, not a transactional relationship.
CREATE TABLE fact_weather_daily (
    weather_id       BIGSERIAL PRIMARY KEY,
    zone_id          INT NOT NULL REFERENCES dim_zones(zone_id),
    date             DATE NOT NULL,
    precipitation_mm NUMERIC(6, 2) NOT NULL DEFAULT 0,
    temp_high_c      NUMERIC(5, 2) NOT NULL,
    temp_low_c       NUMERIC(5, 2) NOT NULL,
    wind_speed_kmh   NUMERIC(5, 2) NOT NULL DEFAULT 0,
    condition        TEXT NOT NULL CHECK (condition IN ('clear', 'rain', 'storm', 'snow')),
    is_flash_flood   BOOLEAN NOT NULL DEFAULT FALSE,
    is_heatwave      BOOLEAN NOT NULL DEFAULT FALSE,
    UNIQUE (zone_id, date)
);

CREATE INDEX idx_fact_weather_daily_zone_date ON fact_weather_daily(zone_id, date);

-- Grain: one row per macro/geopolitical event. Sparse and hand-curated rather
-- than randomly generated, so events read as plausible (fuel price spikes,
-- transit strikes, regional flooding emergencies, economic downturns, etc.).
CREATE TABLE dim_external_events (
    event_id               SERIAL PRIMARY KEY,
    event_type             TEXT NOT NULL,
    start_date             DATE NOT NULL,
    end_date               DATE NOT NULL,
    affected_state         TEXT,  -- NULL = platform-wide (national) effect
    demand_multiplier      NUMERIC(4, 2) NOT NULL DEFAULT 1.0,
    order_value_multiplier NUMERIC(4, 2) NOT NULL DEFAULT 1.0,
    description             TEXT
);

CREATE INDEX idx_dim_external_events_dates ON dim_external_events(start_date, end_date);

-- ============================================================
-- Fact tables
-- ============================================================

-- Grain: one row per delivery ("dash")
CREATE TABLE fact_deliveries (
    delivery_id          BIGSERIAL PRIMARY KEY,
    user_id              INT NOT NULL REFERENCES dim_users(user_id),
    merchant_id          INT NOT NULL REFERENCES dim_merchants(merchant_id),
    dasher_id            INT NOT NULL REFERENCES dim_dashers(dasher_id),
    zone_id              INT NOT NULL REFERENCES dim_zones(zone_id),
    promo_id             INT REFERENCES dim_promotions(promo_id),
    date_key             BIGINT NOT NULL REFERENCES dim_date(date_key),
    order_ts             TIMESTAMPTZ NOT NULL,
    promised_eta_min     NUMERIC(5, 2) NOT NULL,
    actual_delivery_min  NUMERIC(5, 2),
    prep_time_min        NUMERIC(5, 2) NOT NULL,
    pickup_wait_min      NUMERIC(5, 2),
    travel_time_min      NUMERIC(5, 2),
    subtotal             NUMERIC(8, 2) NOT NULL,
    discount_amount      NUMERIC(8, 2) NOT NULL DEFAULT 0,
    delivery_fee         NUMERIC(6, 2) NOT NULL DEFAULT 0,
    tip_amount           NUMERIC(6, 2) NOT NULL DEFAULT 0,
    total_amount         NUMERIC(8, 2) NOT NULL,
    item_count           SMALLINT NOT NULL,
    is_late              BOOLEAN,
    issue_type           TEXT CHECK (issue_type IN ('misplaced', 'not_delivered', 'wrong_order')),  -- NULL = no issue
    refund_amount        NUMERIC(8, 2) NOT NULL DEFAULT 0,
    source               TEXT NOT NULL CHECK (source IN ('historical', 'realtime')),
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_fact_deliveries_order_ts ON fact_deliveries(order_ts);
CREATE INDEX idx_fact_deliveries_zone     ON fact_deliveries(zone_id);
CREATE INDEX idx_fact_deliveries_dasher   ON fact_deliveries(dasher_id);
CREATE INDEX idx_fact_deliveries_merchant ON fact_deliveries(merchant_id);

-- Grain: one row per model prediction for a delivery
CREATE TABLE fact_predictions (
    prediction_id      BIGSERIAL PRIMARY KEY,
    delivery_id        BIGINT NOT NULL REFERENCES fact_deliveries(delivery_id),
    model_name         TEXT NOT NULL,
    model_version      TEXT NOT NULL,
    predicted_eta_min  NUMERIC(5, 2) NOT NULL,
    features_snapshot  JSONB NOT NULL,
    scored_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_fact_predictions_delivery ON fact_predictions(delivery_id);

-- ============================================================
-- Feature store (offline) + data contract quarantine
-- ============================================================

-- Offline mirror of whatever is written to the Redis online store, keyed by entity
CREATE TABLE feature_snapshots (
    snapshot_id    BIGSERIAL PRIMARY KEY,
    entity_type    TEXT NOT NULL CHECK (entity_type IN ('dasher', 'merchant', 'zone', 'user')),
    entity_id      INT NOT NULL,
    feature_name   TEXT NOT NULL,
    feature_value  NUMERIC,
    computed_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (entity_type, entity_id, feature_name, computed_at)
);

CREATE INDEX idx_feature_snapshots_lookup ON feature_snapshots(entity_type, entity_id, feature_name);

-- Rows that failed the Great Expectations contract gate before reaching fact_deliveries
CREATE TABLE deliveries_quarantine (
    quarantine_id       BIGSERIAL PRIMARY KEY,
    raw_record          JSONB NOT NULL,
    failed_expectations JSONB NOT NULL,
    batch_id            TEXT,
    quarantined_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

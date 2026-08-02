"""Trains the ETA prediction model: given what's knowable *before* a delivery
happens (who's involved, when, the weather, and each entity's recent track
record), predict actual_delivery_min.

Point-in-time correctness: training features use each dasher/merchant/zone's
PRIOR calendar month's stats (via SQL window functions on Postgres), never
the same month a delivery happened in — otherwise the model would be
peeking at information that didn't exist yet when the order was placed.
This mirrors what features/batch_features.py computes for live serving,
just at monthly granularity (computed once directly against fact_deliveries
for the whole historical training set, rather than requiring a backfill of
feature_snapshots — see README's Data Dictionary/architecture notes).

Deliberately excluded from the feature set (would leak the answer):
prep_time_min, travel_time_min, pickup_wait_min, promised_eta_min, is_late —
these either compose actual_delivery_min directly or are derived from it.

Usage:
    python ml/train_eta_model.py
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

load_dotenv()

MODEL_DIR = Path(__file__).parent / "model_api"
MODEL_PATH = MODEL_DIR / "model.joblib"
SAMPLE_PCT = 10  # TABLESAMPLE percentage of fact_deliveries used for training
MODEL_VERSION = "eta-v1"

# Feature columns, split by type since HistGradientBoostingRegressor needs to
# know which ones are categorical (it handles them natively — no one-hot
# encoding needed).
NUMERIC_FEATURES = [
    "hour", "day_of_week", "item_count", "subtotal",
    "baseline_prep_time_min", "precipitation_mm", "temp_high_c",
    "dasher_avg_delivery_30d", "dasher_count_30d",
    "merchant_avg_prep_30d", "merchant_late_rate_30d",
    "zone_avg_travel_30d", "zone_order_volume_30d",
]
CATEGORICAL_FEATURES = ["density_tier", "weather_condition", "is_weekend", "is_flash_flood", "is_heatwave"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "actual_delivery_min"

TRAINING_QUERY = f"""
WITH dasher_monthly AS (
    SELECT dasher_id, date_trunc('month', order_ts) AS month,
           AVG(actual_delivery_min) AS avg_val, COUNT(*) AS cnt
    FROM fact_deliveries GROUP BY dasher_id, date_trunc('month', order_ts)
),
dasher_monthly_lag AS (
    SELECT dasher_id, month,
           LAG(avg_val) OVER (PARTITION BY dasher_id ORDER BY month) AS prior_avg_delivery,
           LAG(cnt) OVER (PARTITION BY dasher_id ORDER BY month) AS prior_count
    FROM dasher_monthly
),
merchant_monthly AS (
    SELECT merchant_id, date_trunc('month', order_ts) AS month,
           AVG(prep_time_min) AS avg_prep,
           100.0 * SUM(CASE WHEN is_late THEN 1 ELSE 0 END) / COUNT(*) AS late_rate
    FROM fact_deliveries GROUP BY merchant_id, date_trunc('month', order_ts)
),
merchant_monthly_lag AS (
    SELECT merchant_id, month,
           LAG(avg_prep) OVER (PARTITION BY merchant_id ORDER BY month) AS prior_avg_prep,
           LAG(late_rate) OVER (PARTITION BY merchant_id ORDER BY month) AS prior_late_rate
    FROM merchant_monthly
),
zone_monthly AS (
    SELECT zone_id, date_trunc('month', order_ts) AS month,
           AVG(travel_time_min) AS avg_travel, COUNT(*) AS order_volume
    FROM fact_deliveries GROUP BY zone_id, date_trunc('month', order_ts)
),
zone_monthly_lag AS (
    SELECT zone_id, month,
           LAG(avg_travel) OVER (PARTITION BY zone_id ORDER BY month) AS prior_avg_travel,
           LAG(order_volume) OVER (PARTITION BY zone_id ORDER BY month) AS prior_order_volume
    FROM zone_monthly
),
sampled AS (
    SELECT * FROM fact_deliveries TABLESAMPLE SYSTEM ({SAMPLE_PCT})
)
SELECT
    s.delivery_id, s.order_ts, s.actual_delivery_min, s.item_count, s.subtotal,
    z.density_tier, z.timezone, m.baseline_prep_time_min,
    w.condition AS weather_condition, w.is_flash_flood, w.is_heatwave,
    w.precipitation_mm, w.temp_high_c,
    dml.prior_avg_delivery AS dasher_avg_delivery_30d, dml.prior_count AS dasher_count_30d,
    mml.prior_avg_prep AS merchant_avg_prep_30d, mml.prior_late_rate AS merchant_late_rate_30d,
    zml.prior_avg_travel AS zone_avg_travel_30d, zml.prior_order_volume AS zone_order_volume_30d
FROM sampled s
JOIN dim_zones z ON z.zone_id = s.zone_id
JOIN dim_merchants m ON m.merchant_id = s.merchant_id
LEFT JOIN fact_weather_daily w ON w.zone_id = s.zone_id AND w.date = s.order_ts::date
LEFT JOIN dasher_monthly_lag dml ON dml.dasher_id = s.dasher_id AND dml.month = date_trunc('month', s.order_ts)
LEFT JOIN merchant_monthly_lag mml ON mml.merchant_id = s.merchant_id AND mml.month = date_trunc('month', s.order_ts)
LEFT JOIN zone_monthly_lag zml ON zml.zone_id = s.zone_id AND zml.month = date_trunc('month', s.order_ts)
WHERE dml.prior_avg_delivery IS NOT NULL
  AND mml.prior_avg_prep IS NOT NULL
  AND zml.prior_avg_travel IS NOT NULL
"""


def load_training_data() -> pd.DataFrame:
    conn = psycopg2.connect(os.environ["POSTGRES_URL"])
    df = pd.read_sql(TRAINING_QUERY, conn)
    conn.close()
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """hour/day_of_week/is_weekend must reflect each zone's LOCAL time, not
    UTC — the entire demand model (distributions.py) varies by local
    daypart (lunch/dinner peaks), so a UTC hour would blur that signal
    differently depending on each zone's timezone offset. This same
    per-timezone conversion has to happen identically in model_api/main.py
    at serving time, or training and serving would disagree — exactly the
    training/serving skew failure mode discussed for the feature store."""
    df = df.copy()
    df["order_ts"] = pd.to_datetime(df["order_ts"], utc=True)
    # groupby/transform result is object dtype (mixed per-row timezones can't
    # share one datetime64 column), so .dt accessor won't work — extract via
    # .apply() on the individual tz-aware Timestamp objects instead.
    local_ts = df.groupby("timezone")["order_ts"].transform(lambda s: s.dt.tz_convert(s.name))
    df["hour"] = local_ts.apply(lambda ts: ts.hour)
    df["day_of_week"] = local_ts.apply(lambda ts: ts.weekday())
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(str)
    df["is_flash_flood"] = df["is_flash_flood"].fillna(False).infer_objects(copy=False).astype(str)
    df["is_heatwave"] = df["is_heatwave"].fillna(False).infer_objects(copy=False).astype(str)
    df["weather_condition"] = df["weather_condition"].fillna("clear")
    df["density_tier"] = df["density_tier"].astype(str)
    return df


def time_based_split(df: pd.DataFrame, test_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits chronologically (train on the past, test on the most recent
    slice) rather than randomly — matches how the model is actually used:
    predicting forward, never backward."""
    df_sorted = df.sort_values("order_ts")
    split_idx = int(len(df_sorted) * (1 - test_frac))
    return df_sorted.iloc[:split_idx], df_sorted.iloc[split_idx:]


def main():
    print("Loading training data (point-in-time correct features)...")
    df = load_training_data()
    print(f"  loaded {len(df):,} rows")

    df = engineer_features(df)
    train_df, test_df = time_based_split(df)
    print(f"  train: {len(train_df):,} rows, test: {len(test_df):,} rows (time-based split)")

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df[TARGET_COLUMN]

    categorical_mask = [col in CATEGORICAL_FEATURES for col in FEATURE_COLUMNS]

    mlflow.set_experiment("eta-prediction")
    with mlflow.start_run():
        params = {"max_iter": 200, "max_depth": 6, "learning_rate": 0.1, "random_state": 42}
        mlflow.log_params(params)
        mlflow.log_param("training_rows", len(train_df))
        mlflow.log_param("test_rows", len(test_df))
        mlflow.log_param("sample_pct", SAMPLE_PCT)

        print("Training HistGradientBoostingRegressor...")
        model = HistGradientBoostingRegressor(categorical_features=categorical_mask, **params)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        print(f"  MAE:  {mae:.2f} min")
        print(f"  RMSE: {rmse:.2f} min")
        print(f"  R2:   {r2:.3f}")

        mlflow.log_metric("mae_minutes", mae)
        mlflow.log_metric("rmse_minutes", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(model, "model")

        run_id = mlflow.active_run().info.run_id
        print(f"  MLflow run: {run_id}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "categorical_features": CATEGORICAL_FEATURES,
        "model_version": MODEL_VERSION,
        "mlflow_run_id": run_id,
    }, MODEL_PATH)
    print(f"Saved model artifact to {MODEL_PATH}")


if __name__ == "__main__":
    main()

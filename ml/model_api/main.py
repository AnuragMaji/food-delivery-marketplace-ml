"""FastAPI service that predicts a delivery's ETA using the trained model
plus the online feature store. Given a delivery_id already in
fact_deliveries, looks up what was knowable before the delivery happened
(dasher/merchant/zone identity, weather, live rolling features from Redis),
scores it, and persists the prediction to fact_predictions.

Run locally:
    uvicorn ml.model_api.main:app --reload --port 8000
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import joblib
import pandas as pd
import psycopg2.extras
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "features"))
from feature_store import get_pg_connection, get_redis_client, read_online

MODEL_PATH = Path(__file__).parent / "model.joblib"

app = FastAPI(title="ETA Prediction API")

_artifact = joblib.load(MODEL_PATH)
MODEL = _artifact["model"]
FEATURE_COLUMNS = _artifact["feature_columns"]
MODEL_VERSION = _artifact["model_version"]


class PredictRequest(BaseModel):
    delivery_id: int


class PredictResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    delivery_id: int
    predicted_eta_min: float
    actual_delivery_min: Optional[float]
    model_version: str


@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    conn = get_pg_connection()
    redis_client = get_redis_client()

    with conn.cursor() as cur:
        cur.execute(
            """SELECT dasher_id, merchant_id, zone_id, item_count, subtotal, order_ts, actual_delivery_min
               FROM fact_deliveries WHERE delivery_id = %s""",
            (req.delivery_id,),
        )
        row = cur.fetchone()
    if row is None:
        conn.close()
        raise HTTPException(status_code=404, detail=f"delivery_id {req.delivery_id} not found")

    dasher_id, merchant_id, zone_id, item_count, subtotal, order_ts, actual_delivery_min = row

    with conn.cursor() as cur:
        cur.execute("SELECT density_tier, timezone FROM dim_zones WHERE zone_id = %s", (zone_id,))
        density_tier, timezone = cur.fetchone()
        cur.execute("SELECT baseline_prep_time_min FROM dim_merchants WHERE merchant_id = %s", (merchant_id,))
        baseline_prep_time_min = float(cur.fetchone()[0])
        cur.execute(
            """SELECT condition, is_flash_flood, is_heatwave, precipitation_mm, temp_high_c
               FROM fact_weather_daily WHERE zone_id = %s AND date = %s""",
            (zone_id, order_ts.date()),
        )
        weather_row = cur.fetchone()

    if weather_row is not None:
        weather_condition, is_flash_flood, is_heatwave, precipitation_mm, temp_high_c = weather_row
    else:
        weather_condition, is_flash_flood, is_heatwave, precipitation_mm, temp_high_c = "clear", False, False, 0.0, 20.0

    # Must match train_eta_model.py's local-time conversion exactly, or
    # training and serving disagree about what "hour"/"day_of_week" mean —
    # the same training/serving skew risk discussed for the feature store.
    local_ts = order_ts.astimezone(ZoneInfo(timezone))

    dasher_avg_delivery_30d = read_online(redis_client, "dasher", dasher_id, "avg_delivery_time_30d")
    dasher_count_30d = read_online(redis_client, "dasher", dasher_id, "delivery_count_30d")
    merchant_avg_prep_30d = read_online(redis_client, "merchant", merchant_id, "avg_prep_time_30d")
    merchant_late_rate_30d = read_online(redis_client, "merchant", merchant_id, "late_rate_30d")
    zone_avg_travel_30d = read_online(redis_client, "zone", zone_id, "avg_travel_time_30d")
    zone_order_volume_30d = read_online(redis_client, "zone", zone_id, "order_volume_30d")

    feature_row = {
        "hour": local_ts.hour,
        "day_of_week": local_ts.weekday(),
        "item_count": item_count,
        "subtotal": float(subtotal),
        "baseline_prep_time_min": baseline_prep_time_min,
        "precipitation_mm": float(precipitation_mm),
        "temp_high_c": float(temp_high_c),
        "dasher_avg_delivery_30d": dasher_avg_delivery_30d,
        "dasher_count_30d": dasher_count_30d,
        "merchant_avg_prep_30d": merchant_avg_prep_30d,
        "merchant_late_rate_30d": merchant_late_rate_30d,
        "zone_avg_travel_30d": zone_avg_travel_30d,
        "zone_order_volume_30d": zone_order_volume_30d,
        "density_tier": str(density_tier),
        "weather_condition": weather_condition,
        "is_weekend": str(local_ts.weekday() >= 5),
        "is_flash_flood": str(is_flash_flood),
        "is_heatwave": str(is_heatwave),
    }

    X = pd.DataFrame([feature_row])[FEATURE_COLUMNS]
    predicted_eta_min = round(float(MODEL.predict(X)[0]), 2)

    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO fact_predictions (delivery_id, model_name, model_version, predicted_eta_min, features_snapshot)
               VALUES (%s, %s, %s, %s, %s)""",
            (req.delivery_id, "eta_regressor", MODEL_VERSION, predicted_eta_min, psycopg2.extras.Json(feature_row)),
        )
    conn.commit()
    conn.close()

    return PredictResponse(
        delivery_id=req.delivery_id,
        predicted_eta_min=predicted_eta_min,
        actual_delivery_min=float(actual_delivery_min) if actual_delivery_min is not None else None,
        model_version=MODEL_VERSION,
    )

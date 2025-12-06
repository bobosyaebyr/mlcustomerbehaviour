# src/api/main.py

from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow import keras
import json


BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models"

# Загружаем артефакты при старте приложения
scaler = joblib.load(MODELS_DIR / "scaler.joblib")
pca = joblib.load(MODELS_DIR / "pca.joblib")
model = keras.models.load_model(MODELS_DIR / "tf_model.keras")

with open(MODELS_DIR / "feature_columns.json", "r", encoding="utf-8") as f:
    FEATURE_COLUMNS: List[str] = json.load(f)


class CustomerFeatures(BaseModel):
    age: int
    income: float
    tenure_months: int
    transactions_last_30d: int
    amount_last_30d: float
    days_since_last_purchase: int
    is_active_app_user: int
    support_tickets_last_90d: int
    discount_rate_mean: float
    segment_code: int


app = FastAPI(
    title="Customer Behavior Prediction API",
    description="API для прогнозирования вероятности покупки клиента в ближайшие 30 дней",
    version="1.0.0",
)


@app.get("/")
def read_root():
    return {
        "message": "Customer Behavior Prediction API",
        "docs_url": "/docs",
    }


@app.post("/predict")
def predict_purchase(features: CustomerFeatures):
    # Преобразуем вход в DataFrame
    data = pd.DataFrame([features.dict()], columns=FEATURE_COLUMNS)

    # Препроцессинг
    scaled = scaler.transform(data.values)
    transformed = pca.transform(scaled)

    # Предсказание
    proba = float(model.predict(transformed)[0][0])

    # Округление до 3 знаков после запятой
    proba_rounded = round(proba, 3)

    return {
        "purchase_probability_next_30d": proba_rounded,
        "will_buy_next_30d": int(proba_rounded >= 0.5),
    }

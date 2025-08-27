# TODO: Implement FastAPI app for churn inference.
# Endpoints: GET /health, POST /predict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np

# Cargar pipeline y modelo
feature_pipeline = joblib.load("artifacts/feature_pipeline.pkl")
model = joblib.load("artifacts/model.pkl")

app = FastAPI(title="Churn Prediction API")

# === Pydantic schema ===
class CustomerFeatures(BaseModel):
    plan_type: str
    contract_type: str
    autopay: str
    is_promo_user: str
    add_on_count: float
    tenure_months: float
    monthly_usage_gb: float
    avg_latency_ms: float
    support_tickets_30d: float
    discount_pct: float
    payment_failures_90d: float
    downtime_hours_30d: float

# === Endpoints ===
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(rows: List[CustomerFeatures]):
    try:
        import pandas as pd
        df = pd.DataFrame([r.dict() for r in rows])

        # Ordenar columnas para que coincidan con el pipeline
        expected_cols = [
            "plan_type",
            "contract_type",
            "autopay",
            "is_promo_user",
            "add_on_count",
            "tenure_months",
            "monthly_usage_gb",
            "avg_latency_ms",
            "support_tickets_30d",
            "discount_pct",
            "payment_failures_90d",
            "downtime_hours_30d"
        ]
        df = df[expected_cols]

        # Transformar features
        X = feature_pipeline.transform(df)

        probs = model.predict_proba(X)[:, 1]
        preds = (probs >= 0.5).astype(int)

        return [
            {"probability": float(p), "class": int(c)}
            for p, c in zip(probs, preds)
        ]

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

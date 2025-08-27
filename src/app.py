# TODO: Implement FastAPI app for churn inference.
# Endpoints: GET /health, POST /predict
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional


# Cargar el pipeline completo (preprocesador + modelo)
pipe = joblib.load("artifacts/model.pkl")

app = FastAPI(title="Churn Prediction API")

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
    downtime_hours_30d: Optional[float] = 0.0 
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(rows: List[CustomerFeatures]):
    try:
        df = pd.DataFrame([r.dict() for r in rows])
        # aseguramos columna downtime_hours_30d
        if "downtime_hours_30d" not in df.columns:
            df["downtime_hours_30d"] = 0.0

        # El pipeline ya se encarga de todo (preprocesar + modelar)
        probs = pipe.predict_proba(df)[:, 1]
        preds = (probs >= 0.5).astype(int)

        return [
            {"probability": float(p), "class": int(c)}
            for p, c in zip(probs, preds)
        ]

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))



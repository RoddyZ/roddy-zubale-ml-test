# TODO: Pydantic schemas for /predict
# src/io_schemas.py
from pydantic import BaseModel, Field
from typing import List, Literal

class PredictRequest(BaseModel):
    plan_type: Literal["Basic", "Standard", "Pro"]
    contract_type: Literal["Monthly", "Annual"]
    autopay: Literal["Yes", "No"]
    is_promo_user: Literal["Yes", "No"]
    add_on_count: int = Field(..., ge=0)
    tenure_months: int = Field(..., ge=0)
    monthly_usage_gb: float = Field(..., ge=0)
    avg_latency_ms: float = Field(..., ge=0)
    support_tickets_30d: int = Field(..., ge=0)
    discount_pct: float = Field(..., ge=0)
    payment_failures_90d: int = Field(..., ge=0)
    downtime_hours_30d: float = Field(..., ge=0)

class PredictResponse(BaseModel):
    probability: float
    prediction: int

class BatchPredictRequest(BaseModel):
    __root__: List[PredictRequest]

class BatchPredictResponse(BaseModel):
    __root__: List[PredictResponse]

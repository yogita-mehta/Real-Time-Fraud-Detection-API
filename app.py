"""
app.py  –  FastAPI real-time fraud-detection inference service.
Start with:  uvicorn app:app --reload
"""

from __future__ import annotations
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from typing import List
import os

# ── Pydantic schemas ─────────────────────────────────────────────────────────

class TransactionFeatures(BaseModel):
    """
    One credit-card transaction.  All 20 features are required.
    The first three are human-interpretable; v1-v17 are PCA components.
    """
    amount:       float = Field(..., example=129.99,  description="Transaction amount (USD)")
    hour_of_day:  float = Field(..., example=14.0,    description="Hour 0-23 when txn occurred")
    day_of_week:  float = Field(..., example=2.0,     description="Day 0=Mon … 6=Sun")
    v1:  float = Field(..., example=-1.36)
    v2:  float = Field(..., example=0.24)
    v3:  float = Field(..., example=-1.54)
    v4:  float = Field(..., example=0.09)
    v5:  float = Field(..., example=-0.37)
    v6:  float = Field(..., example=0.45)
    v7:  float = Field(..., example=-0.61)
    v8:  float = Field(..., example=0.19)
    v9:  float = Field(..., example=-0.23)
    v10: float = Field(..., example=0.82)
    v11: float = Field(..., example=-0.09)
    v12: float = Field(..., example=0.12)
    v13: float = Field(..., example=-0.54)
    v14: float = Field(..., example=0.31)
    v15: float = Field(..., example=-0.78)
    v16: float = Field(..., example=0.65)
    v17: float = Field(..., example=-0.42)


class PredictionResponse(BaseModel):
    is_fraud:          bool
    fraud_probability: float
    confidence:        str   # "High" / "Medium" / "Low"


class BatchRequest(BaseModel):
    transactions: List[TransactionFeatures]


# ── App lifecycle ─────────────────────────────────────────────────────────────

MODEL_PATH = "fraud_model.pkl"
_artefact: dict = {}   # populated at startup

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once at startup; release on shutdown."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model artefact '{MODEL_PATH}' not found. "
            "Run `python model.py` first to generate it."
        )
    _artefact.update(joblib.load(MODEL_PATH))
    print(f"[startup] Loaded model from {MODEL_PATH}")
    print(f"[startup] Features expected: {_artefact['feature_names']}")
    yield
    _artefact.clear()
    print("[shutdown] Model released from memory.")


app = FastAPI(
    title="Real-Time Fraud Detection API",
    description=(
        "Serves a Random Forest classifier trained on synthetic "
        "credit-card transactions. POST a transaction to /predict "
        "and receive a fraud probability instantly."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _confidence(prob: float) -> str:
    if prob >= 0.75 or prob <= 0.25:
        return "High"
    if prob >= 0.60 or prob <= 0.40:
        return "Medium"
    return "Low"

def _predict_one(txn: TransactionFeatures) -> PredictionResponse:
    features = _artefact["feature_names"]
    row = np.array([[getattr(txn, f) for f in features]])
    prob = float(_artefact["model"].predict_proba(row)[0, 1])
    return PredictionResponse(
        is_fraud=prob >= 0.50,
        fraud_probability=round(prob, 6),
        confidence=_confidence(prob),
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "Fraud Detection API is running."}


@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy", "model_loaded": bool(_artefact)}


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(transaction: TransactionFeatures):
    """
    Classify a single credit-card transaction as legitimate or fraudulent.

    - **is_fraud**: boolean verdict  
    - **fraud_probability**: raw model probability (0 – 1)  
    - **confidence**: High / Medium / Low  
    """
    try:
        return _predict_one(transaction)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict/batch", response_model=List[PredictionResponse], tags=["Inference"])
def predict_batch(payload: BatchRequest):
    """Classify up to 1 000 transactions in a single request."""
    if len(payload.transactions) > 1_000:
        raise HTTPException(status_code=400, detail="Batch size exceeds limit of 1 000.")
    try:
        return [_predict_one(t) for t in payload.transactions]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

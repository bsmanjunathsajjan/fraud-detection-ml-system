"""Fraud risk prediction API. Run python train.py before starting."""
import os, joblib, numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
app = FastAPI(title="Fraud Detection ML System", version="1.0.0")
MODEL_PATH = os.getenv("MODEL_PATH","fraud_model.joblib")
class Transaction(BaseModel):
    amount: float = Field(ge=0)
    hour: int = Field(ge=0,le=23)
    distance_km: float = Field(ge=0)
    failed_attempts: int = Field(ge=0)
@app.get("/health")
def health():
    return {"status":"ok","model_available":os.path.exists(MODEL_PATH)}
@app.post("/predict")
def predict(t: Transaction):
    if not os.path.exists(MODEL_PATH): raise HTTPException(503,"Model missing. Run python train.py first.")
    model=joblib.load(MODEL_PATH)
    x=np.array([[t.amount,t.hour,t.distance_km,t.failed_attempts]])
    p=float(model.predict_proba(x)[0,1])
    return {"fraud_probability":round(p,4),"risk_level":"high" if p>=.7 else "medium" if p>=.35 else "low","requires_human_review":p>=.35}

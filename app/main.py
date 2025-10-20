from __future__ import annotations
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import os
from ml_service.model_io import load_model

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.joblib")
_model, _meta = load_model(MODEL_PATH)
FEATURES = _meta.get("features", ["age","sex","bmi","bp","s1","s2","s3","s4","s5","s6"])

class PredictRequest(BaseModel):
    age: float; sex: float; bmi: float; bp: float; s1: float; s2: float; s3: float; s4: float; s5: float; s6: float

class PredictResponse(BaseModel):
    prediction: float = Field(..., description="Predicted short-term progression index")
    model_version: str

app = FastAPI(title="Virtual Diabetes Clinic – ML Service")

@app.get("/health")
def health():
    return {"status": "ok", "model_version": _meta.get("model_version", "unknown")}

@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    try:
        X = [[getattr(payload, f) for f in FEATURES]]
        yhat = float(_model.predict(X)[0])
        return {"prediction": yhat, "model_version": _meta.get("model_version", "unknown")}
    except Exception as e:
        raise HTTPException(status_code=400, detail={"error": str(e)})

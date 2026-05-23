from __future__ import annotations

import sys
from pathlib import Path
from typing import List

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from inference import get_device, load_labels, load_model
from utils.config import model_artifact_path


DEVICE = get_device()
ID_TO_LABEL = load_labels()
MODEL_PATH = model_artifact_path()
model, ckpt = load_model(path=MODEL_PATH, device=DEVICE)

WINDOW_SIZE = ckpt["window_size"]
INPUT_SIZE = ckpt["input_size"]
NUM_CLASSES = ckpt["num_classes"]
FEATURE_COLS = ckpt.get("feature_cols", [])


class PredictRequest(BaseModel):
    sequence: List[List[float]] = Field(
        ...,
        description="Input sensor window. Expected shape is (window_size, input_size).",
    )


class PredictResponse(BaseModel):
    predicted_class_id: int
    predicted_label: str
    probability: float
    probabilities: dict[str, float]


app = FastAPI(
    title="Behavior Prediction API",
    description="LSTM behavior prediction API based on a sensor time window.",
    version="0.1.0",
)


@app.get("/")
def root():
    return {"message": "Behavior Prediction API is running", "docs": "/docs"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "model_loaded": True,
        "model_path": str(MODEL_PATH),
    }


@app.get("/model-info")
def model_info():
    return {
        "model_type": "LSTM",
        "window_size": WINDOW_SIZE,
        "input_size": INPUT_SIZE,
        "num_classes": NUM_CLASSES,
        "feature_cols": FEATURE_COLS,
        "id_to_label": ID_TO_LABEL,
        "model_path": str(MODEL_PATH),
    }


@app.post("/predict", response_model=PredictResponse)
@torch.no_grad()
def predict(request: PredictRequest):
    arr = np.array(request.sequence, dtype=np.float32)
    if arr.ndim != 2:
        raise HTTPException(status_code=400, detail=f"sequence must be 2D. current shape={arr.shape}")
    if arr.shape[0] != WINDOW_SIZE:
        raise HTTPException(status_code=400, detail=f"window length must be {WINDOW_SIZE}. current={arr.shape[0]}")
    if arr.shape[1] != INPUT_SIZE:
        raise HTTPException(status_code=400, detail=f"feature count must be {INPUT_SIZE}. current={arr.shape[1]}")

    x = torch.from_numpy(arr).unsqueeze(0).to(DEVICE)
    probs = torch.softmax(model(x), dim=1).cpu().numpy()[0]
    pred_id = int(np.argmax(probs))
    return {
        "predicted_class_id": pred_id,
        "predicted_label": ID_TO_LABEL.get(pred_id, str(pred_id)),
        "probability": float(probs[pred_id]),
        "probabilities": {
            ID_TO_LABEL.get(i, str(i)): float(probs[i])
            for i in range(len(probs))
        },
    }

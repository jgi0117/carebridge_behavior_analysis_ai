# =========================================================
# app_api.py
# LSTM 행동 예측 FastAPI 서버
# =========================================================

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# =========================================================
# 1. Config
# =========================================================
PROJECT_ROOT = Path(__file__).parent

MODEL_PATH = PROJECT_ROOT / "output" / "lstm_window_1h_baseline" / "best_lstm_window_1h.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 학습 시 target_behavior_id mapping 기준에 맞게 수정 필요
ID_TO_LABEL = {
    0: "기타",
    1: "수면",
    2: "외출",
    3: "식사",
}


# =========================================================
# 2. Model
# =========================================================
class LSTMBaseline(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=1, dropout=0.0):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        logits = self.classifier(last_hidden)
        return logits


# =========================================================
# 3. Request / Response Schema
# =========================================================
class PredictRequest(BaseModel):
    sequence: List[List[float]] = Field(
        ...,
        description="1시간 window 입력. shape = (6, 15)",
        example=[
            [23.1, 45.2, 120.0, 0.0, 550.0, 0.03, 0.0, 6.31, 0.02, 4.79, 14, 2, 78, 0, 1],
            [23.0, 45.1, 118.0, 0.0, 552.0, 0.03, 0.0, 6.31, 0.02, 4.78, 14, 2, 78, 0, 1],
            [23.0, 45.0, 117.0, 0.0, 553.0, 0.03, 0.0, 6.32, 0.02, 4.77, 14, 2, 78, 0, 1],
            [22.9, 44.9, 116.0, 0.0, 554.0, 0.03, 0.0, 6.32, 0.02, 4.76, 14, 2, 78, 0, 1],
            [22.9, 44.8, 115.0, 0.0, 555.0, 0.03, 0.0, 6.32, 0.02, 4.75, 14, 2, 78, 0, 1],
            [22.8, 44.7, 114.0, 0.0, 556.0, 0.03, 0.0, 6.32, 0.02, 4.74, 14, 2, 78, 0, 1],
        ],
    )


class PredictResponse(BaseModel):
    predicted_class_id: int
    predicted_label: str
    probability: float
    probabilities: dict


# =========================================================
# 4. Load Model
# =========================================================
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"모델 파일이 없습니다: {MODEL_PATH}")

    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)

    model = LSTMBaseline(
        input_size=ckpt["input_size"],
        hidden_size=ckpt["hidden_size"],
        num_classes=ckpt["num_classes"],
        num_layers=ckpt.get("num_layers", 1),
        dropout=ckpt.get("dropout", 0.0),
    )

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model, ckpt


model, ckpt = load_model()

WINDOW_SIZE = ckpt["window_size"]
INPUT_SIZE = ckpt["input_size"]
NUM_CLASSES = ckpt["num_classes"]
FEATURE_COLS = ckpt.get("feature_cols", [])


# =========================================================
# 5. FastAPI App
# =========================================================
app = FastAPI(
    title="Behavior Prediction API",
    description="1시간 센서 window 기반 행동 예측 LSTM API",
    version="0.1.0",
)


@app.get("/")
def root():
    return {
        "message": "Behavior Prediction API is running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "model_loaded": True,
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
    }


@app.post("/predict", response_model=PredictResponse)
@torch.no_grad()
def predict(request: PredictRequest):
    arr = np.array(request.sequence, dtype=np.float32)

    if arr.ndim != 2:
        raise HTTPException(
            status_code=400,
            detail=f"sequence는 2차원 배열이어야 합니다. 현재 shape={arr.shape}",
        )

    if arr.shape[0] != WINDOW_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"window 길이는 {WINDOW_SIZE}이어야 합니다. 현재={arr.shape[0]}",
        )

    if arr.shape[1] != INPUT_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"feature 개수는 {INPUT_SIZE}이어야 합니다. 현재={arr.shape[1]}",
        )

    x = torch.from_numpy(arr).unsqueeze(0).to(DEVICE)

    logits = model(x)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_id = int(np.argmax(probs))
    pred_prob = float(probs[pred_id])

    probabilities = {
        ID_TO_LABEL.get(i, str(i)): float(probs[i])
        for i in range(len(probs))
    }

    return {
        "predicted_class_id": pred_id,
        "predicted_label": ID_TO_LABEL.get(pred_id, str(pred_id)),
        "probability": pred_prob,
        "probabilities": probabilities,
    }
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


# ---------------------------------------------------------
# 모델 로딩
# ---------------------------------------------------------
# API 프로세스가 시작될 때 checkpoint를 한 번만 로드합니다.
# 요청마다 모델을 다시 읽지 않기 때문에 예측 응답이 빨라집니다.
DEVICE = get_device()
ID_TO_LABEL = load_labels()
MODEL_PATH = model_artifact_path()
model, ckpt = load_model(path=MODEL_PATH, device=DEVICE)

WINDOW_SIZE = ckpt["window_size"]
INPUT_SIZE = ckpt["input_size"]
NUM_CLASSES = ckpt["num_classes"]
FEATURE_COLS = ckpt.get("feature_cols", [])


# ---------------------------------------------------------
# 요청/응답 스키마
# ---------------------------------------------------------
# sequence는 학습 때 사용한 window shape와 같아야 합니다.
# 현재 checkpoint 기준으로는 보통 (6, 15) 형태입니다.
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


# ---------------------------------------------------------
# FastAPI 앱 정의
# ---------------------------------------------------------
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
    # 입력 검증: 모델은 고정된 window 길이와 feature 개수만 받을 수 있습니다.
    arr = np.array(request.sequence, dtype=np.float32)
    if arr.ndim != 2:
        raise HTTPException(status_code=400, detail=f"sequence must be 2D. current shape={arr.shape}")
    if arr.shape[0] != WINDOW_SIZE:
        raise HTTPException(status_code=400, detail=f"window length must be {WINDOW_SIZE}. current={arr.shape[0]}")
    if arr.shape[1] != INPUT_SIZE:
        raise HTTPException(status_code=400, detail=f"feature count must be {INPUT_SIZE}. current={arr.shape[1]}")

    # 추론: softmax 확률 중 가장 큰 클래스를 최종 행동으로 선택합니다.
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

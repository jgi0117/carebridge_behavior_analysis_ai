from __future__ import annotations

from pathlib import Path

import torch

from models import LSTMBaseline
from utils.config import load_config, model_artifact_path


# ---------------------------------------------------------
# 추론 환경과 모델 로딩
# ---------------------------------------------------------
# API와 Gradio가 같은 로딩 코드를 공유하도록 분리했습니다.
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(path: Path | None = None, device: torch.device | None = None) -> dict:
    # checkpoint 경로를 직접 넘기지 않으면 configs/paths.yaml의 경로를 사용합니다.
    device = device or get_device()
    model_path = path or model_artifact_path()
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    return torch.load(model_path, map_location=device)


def load_model(path: Path | None = None, device: torch.device | None = None):
    # checkpoint에 저장된 모델 hyperparameter로 동일한 LSTM 구조를 복원합니다.
    device = device or get_device()
    ckpt = load_checkpoint(path, device)
    model = LSTMBaseline(
        input_size=ckpt["input_size"],
        hidden_size=ckpt["hidden_size"],
        num_classes=ckpt["num_classes"],
        num_layers=ckpt.get("num_layers", 1),
        dropout=ckpt.get("dropout", 0.0),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt


def load_labels() -> dict[int, str]:
    # API 응답에서 class id 대신 사람이 읽을 수 있는 한글 라벨을 보여주기 위한 mapping입니다.
    config = load_config("paths.yaml")
    labels = config.get("app", {}).get("labels", {})
    return {int(key): str(value) for key, value in labels.items()}

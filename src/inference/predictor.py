from __future__ import annotations

from pathlib import Path

import torch

from models import LSTMBaseline
from utils.config import load_config, model_artifact_path


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(path: Path | None = None, device: torch.device | None = None) -> dict:
    device = device or get_device()
    model_path = path or model_artifact_path()
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    return torch.load(model_path, map_location=device)


def load_model(path: Path | None = None, device: torch.device | None = None):
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
    config = load_config("paths.yaml")
    labels = config.get("app", {}).get("labels", {})
    return {int(key): str(value) for key, value in labels.items()}

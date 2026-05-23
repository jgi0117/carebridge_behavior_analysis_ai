from __future__ import annotations

import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import LSTMBaseline
from train.dataset import WindowNPZDataset
from train.engine import compute_class_weight, evaluate, seed_everything, train_one_epoch
from utils.config import configured_path, load_config
from utils.io import save_json


def main() -> None:
    config = load_config("paths.yaml", "training.yaml")
    training_cfg = config["training"]
    seed_everything(int(training_cfg["seed"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = configured_path("window_data", config)
    save_dir = configured_path("model_output", config)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = WindowNPZDataset(data_root / "train_parts")
    valid_dataset = WindowNPZDataset(data_root / "valid_parts")

    window_size = train_dataset.X.shape[1]
    input_size = train_dataset.X.shape[2]
    num_classes = int(max(train_dataset.y.max(), valid_dataset.y.max()) + 1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training_cfg["batch_size"]),
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=int(training_cfg["batch_size"]),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    model = LSTMBaseline(
        input_size=input_size,
        hidden_size=int(training_cfg["hidden_size"]),
        num_classes=num_classes,
        num_layers=int(training_cfg["num_layers"]),
        dropout=float(training_cfg["dropout"]),
    ).to(device)

    class_weight = compute_class_weight(train_dataset.y, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training_cfg["lr"]))

    best_valid_acc = -1.0
    history = []
    print("[INFO] device:", device)
    print("[INFO] window_size:", window_size)
    print("[INFO] input_size:", input_size)
    print("[INFO] num_classes:", num_classes)

    for epoch in range(1, int(training_cfg["epochs"]) + 1):
        print(f"\n========== Epoch {epoch}/{training_cfg['epochs']} ==========")
        train_result = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        valid_result = evaluate(model, valid_loader, criterion, device, num_classes, epoch)
        row = {
            "epoch": epoch,
            "train_loss": train_result["loss"],
            "train_acc": train_result["acc"],
            "valid_loss": valid_result["loss"],
            "valid_acc": valid_result["acc"],
            "valid_class_acc": valid_result["class_acc"],
            "valid_class_total": valid_result["class_total"],
        }
        history.append(row)
        print(
            f"[Epoch {epoch:02d}] "
            f"train_loss={row['train_loss']:.4f} train_acc={row['train_acc']:.4f} "
            f"valid_loss={row['valid_loss']:.4f} valid_acc={row['valid_acc']:.4f}"
        )

        if valid_result["acc"] > best_valid_acc:
            best_valid_acc = valid_result["acc"]
            feature_cols = np.load(train_dataset.files[0])["feature_cols"].tolist()
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_size": input_size,
                    "hidden_size": int(training_cfg["hidden_size"]),
                    "num_classes": num_classes,
                    "num_layers": int(training_cfg["num_layers"]),
                    "dropout": float(training_cfg["dropout"]),
                    "window_size": window_size,
                    "feature_cols": feature_cols,
                    "best_valid_acc": best_valid_acc,
                },
                save_dir / "best_lstm_window_1h.pt",
            )
            print("  [SAVE] best model saved")

    save_json(history, save_dir / "train_history.json")
    save_json(
        {
            "data_root": str(data_root),
            "save_dir": str(save_dir),
            **training_cfg,
            "device": str(device),
            "best_valid_acc": best_valid_acc,
        },
        save_dir / "train_config.json",
    )
    print("[FINISH] save dir:", save_dir)


if __name__ == "__main__":
    main()

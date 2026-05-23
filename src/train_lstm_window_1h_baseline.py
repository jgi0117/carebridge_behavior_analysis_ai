# =========================================================
# train_lstm_window_1h_baseline.py
# 1시간 window 데이터 기반 LSTM baseline 학습 + tqdm 진행률
# =========================================================

import json
import random
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# =========================================================
# 1. Config
# =========================================================
PROJECT_ROOT = Path(__file__).parent

DATA_ROOT = PROJECT_ROOT / "output" / "behavior_window_1h"
TRAIN_DIR = DATA_ROOT / "train_parts"
VALID_DIR = DATA_ROOT / "valid_parts"

SAVE_DIR = PROJECT_ROOT / "output" / "lstm_window_1h_baseline"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 512
EPOCHS = 10
LR = 1e-3

HIDDEN_SIZE = 64
NUM_LAYERS = 1
DROPOUT = 0.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# 2. Seed
# =========================================================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything(SEED)


# =========================================================
# 3. Dataset
# =========================================================
class WindowNPZDataset(Dataset):
    def __init__(self, npz_dir):
        self.files = sorted(Path(npz_dir).glob("part_*.npz"))

        if len(self.files) == 0:
            raise FileNotFoundError(f"npz 파일이 없습니다: {npz_dir}")

        self.X_list = []
        self.y_list = []

        print(f"[LOAD] {npz_dir}")

        for f in tqdm(self.files, desc=f"Loading {Path(npz_dir).name}"):
            data = np.load(f)
            X = data["X"].astype(np.float32)
            y = data["y"].astype(np.int64)

            self.X_list.append(X)
            self.y_list.append(y)

            print(f"  - {f.name} | X={X.shape}, y={y.shape}")

        self.X = np.concatenate(self.X_list, axis=0)
        self.y = np.concatenate(self.y_list, axis=0)

        print(f"[DONE] total X={self.X.shape}, y={self.y.shape}")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X[idx]),
            torch.tensor(self.y[idx], dtype=torch.long),
        )


# =========================================================
# 4. Model
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
# 5. Class weight
# =========================================================
def compute_class_weight(y, num_classes):
    counter = Counter(y.tolist())
    total = len(y)

    weights = []
    for cls in range(num_classes):
        count = counter.get(cls, 0)
        if count == 0:
            weights.append(0.0)
        else:
            weights.append(total / (num_classes * count))

    return torch.tensor(weights, dtype=torch.float32)


# =========================================================
# 6. Train / Eval
# =========================================================
def train_one_epoch(model, loader, criterion, optimizer, epoch):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    pbar = tqdm(loader, desc=f"[Train Epoch {epoch}]", leave=True)

    for X, y in pbar:
        X = X.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)

        total_loss += loss.item() * len(y)
        total_correct += (preds == y).sum().item()
        total_count += len(y)

        current_loss = total_loss / total_count
        current_acc = total_correct / total_count

        pbar.set_postfix({
            "loss": f"{current_loss:.4f}",
            "acc": f"{current_acc:.4f}",
        })

    return {
        "loss": total_loss / total_count,
        "acc": total_correct / total_count,
    }


@torch.no_grad()
def evaluate(model, loader, criterion, num_classes, epoch):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    class_correct = np.zeros(num_classes, dtype=np.int64)
    class_total = np.zeros(num_classes, dtype=np.int64)

    pbar = tqdm(loader, desc=f"[Valid Epoch {epoch}]", leave=True)

    for X, y in pbar:
        X = X.to(DEVICE)
        y = y.to(DEVICE)

        logits = model(X)
        loss = criterion(logits, y)
        preds = logits.argmax(dim=1)

        total_loss += loss.item() * len(y)
        total_correct += (preds == y).sum().item()
        total_count += len(y)

        for cls in range(num_classes):
            mask = y == cls
            class_total[cls] += mask.sum().item()
            class_correct[cls] += ((preds == cls) & mask).sum().item()

        current_loss = total_loss / total_count
        current_acc = total_correct / total_count

        pbar.set_postfix({
            "loss": f"{current_loss:.4f}",
            "acc": f"{current_acc:.4f}",
        })

    class_acc = {}
    for cls in range(num_classes):
        if class_total[cls] == 0:
            class_acc[str(cls)] = None
        else:
            class_acc[str(cls)] = float(class_correct[cls] / class_total[cls])

    return {
        "loss": total_loss / total_count,
        "acc": total_correct / total_count,
        "class_acc": class_acc,
        "class_total": {str(i): int(v) for i, v in enumerate(class_total)},
    }


# =========================================================
# 7. Main
# =========================================================
def main():
    print("[INFO] device:", DEVICE)

    train_dataset = WindowNPZDataset(TRAIN_DIR)
    valid_dataset = WindowNPZDataset(VALID_DIR)

    window_size = train_dataset.X.shape[1]
    input_size = train_dataset.X.shape[2]
    num_classes = int(max(train_dataset.y.max(), valid_dataset.y.max()) + 1)

    print("[INFO] window_size:", window_size)
    print("[INFO] input_size :", input_size)
    print("[INFO] num_classes:", num_classes)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    class_weight = compute_class_weight(train_dataset.y, num_classes).to(DEVICE)
    print("[INFO] class_weight:", class_weight.cpu().numpy())

    model = LSTMBaseline(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_classes=num_classes,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_valid_acc = -1
    history = []

    for epoch in range(1, EPOCHS + 1):
        print(f"\n========== Epoch {epoch}/{EPOCHS} ==========")

        train_result = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
        )

        valid_result = evaluate(
            model=model,
            loader=valid_loader,
            criterion=criterion,
            num_classes=num_classes,
            epoch=epoch,
        )

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
            f"[Epoch {epoch:02d}/{EPOCHS}] "
            f"train_loss={train_result['loss']:.4f} "
            f"train_acc={train_result['acc']:.4f} "
            f"valid_loss={valid_result['loss']:.4f} "
            f"valid_acc={valid_result['acc']:.4f}"
        )

        print("  valid_class_acc:", valid_result["class_acc"])

        if valid_result["acc"] > best_valid_acc:
            best_valid_acc = valid_result["acc"]

            feature_cols = np.load(train_dataset.files[0])["feature_cols"].tolist()

            ckpt = {
                "model_state_dict": model.state_dict(),
                "input_size": input_size,
                "hidden_size": HIDDEN_SIZE,
                "num_classes": num_classes,
                "num_layers": NUM_LAYERS,
                "dropout": DROPOUT,
                "window_size": window_size,
                "feature_cols": feature_cols,
                "best_valid_acc": best_valid_acc,
            }

            torch.save(ckpt, SAVE_DIR / "best_lstm_window_1h.pt")
            print("  [SAVE] best model saved")

    with open(SAVE_DIR / "train_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    config = {
        "data_root": str(DATA_ROOT),
        "train_dir": str(TRAIN_DIR),
        "valid_dir": str(VALID_DIR),
        "save_dir": str(SAVE_DIR),
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "lr": LR,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "device": str(DEVICE),
        "best_valid_acc": best_valid_acc,
    }

    with open(SAVE_DIR / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("\n[FINISH]")
    print("best_valid_acc:", best_valid_acc)
    print("save_dir:", SAVE_DIR)


if __name__ == "__main__":
    main()
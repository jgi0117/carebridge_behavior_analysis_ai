from __future__ import annotations

from collections import Counter

import numpy as np
import torch
from tqdm import tqdm


def seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_class_weight(y: np.ndarray, num_classes: int) -> torch.Tensor:
    counter = Counter(y.tolist())
    total = len(y)
    weights = []
    for cls in range(num_classes):
        count = counter.get(cls, 0)
        weights.append(0.0 if count == 0 else total / (num_classes * count))
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(model, loader, criterion, optimizer, device, epoch: int) -> dict:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    pbar = tqdm(loader, desc=f"[Train Epoch {epoch}]", leave=True)
    for X, y in pbar:
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        total_loss += loss.item() * len(y)
        total_correct += (preds == y).sum().item()
        total_count += len(y)
        pbar.set_postfix({
            "loss": f"{total_loss / total_count:.4f}",
            "acc": f"{total_correct / total_count:.4f}",
        })

    return {"loss": total_loss / total_count, "acc": total_correct / total_count}


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes: int, epoch: int) -> dict:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    class_correct = np.zeros(num_classes, dtype=np.int64)
    class_total = np.zeros(num_classes, dtype=np.int64)

    pbar = tqdm(loader, desc=f"[Valid Epoch {epoch}]", leave=True)
    for X, y in pbar:
        X = X.to(device)
        y = y.to(device)

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

        pbar.set_postfix({
            "loss": f"{total_loss / total_count:.4f}",
            "acc": f"{total_correct / total_count:.4f}",
        })

    class_acc = {
        str(cls): None if class_total[cls] == 0 else float(class_correct[cls] / class_total[cls])
        for cls in range(num_classes)
    }
    return {
        "loss": total_loss / total_count,
        "acc": total_correct / total_count,
        "class_acc": class_acc,
        "class_total": {str(i): int(v) for i, v in enumerate(class_total)},
    }

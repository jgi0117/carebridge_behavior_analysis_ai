from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class WindowNPZDataset(Dataset):
    def __init__(self, npz_dir: Path):
        self.files = sorted(Path(npz_dir).glob("part_*.npz"))
        if not self.files:
            raise FileNotFoundError(f"npz files not found: {npz_dir}")

        x_list = []
        y_list = []
        print(f"[LOAD] {npz_dir}")
        for path in tqdm(self.files, desc=f"Loading {Path(npz_dir).name}"):
            data = np.load(path)
            x_list.append(data["X"].astype(np.float32))
            y_list.append(data["y"].astype(np.int64))
            print(f"  - {path.name} | X={x_list[-1].shape}, y={y_list[-1].shape}")

        self.X = np.concatenate(x_list, axis=0)
        self.y = np.concatenate(y_list, axis=0)
        print(f"[DONE] total X={self.X.shape}, y={self.y.shape}")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X[idx]),
            torch.tensor(self.y[idx], dtype=torch.long),
        )

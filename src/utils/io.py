from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def read_csv_safe(path: Path, **kwargs) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig", **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8", **kwargs)


def list_part_files(part_dir: Path, suffix: str = ".csv") -> list[Path]:
    return sorted(part_dir.glob(f"part_*{suffix}"))


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

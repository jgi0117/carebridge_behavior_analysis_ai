from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


# ---------------------------------------------------------
# 공통 파일 입출력 헬퍼
# ---------------------------------------------------------
# 여러 단계에서 반복되는 디렉토리 초기화, CSV 읽기, JSON 저장을 모아둔 파일입니다.
def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def read_csv_safe(path: Path, **kwargs) -> pd.DataFrame:
    # utf-8-sig를 먼저 시도하고, 실패하면 utf-8로 다시 읽습니다.
    try:
        return pd.read_csv(path, encoding="utf-8-sig", **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8", **kwargs)


def list_part_files(part_dir: Path, suffix: str = ".csv") -> list[Path]:
    # part_0000.csv 또는 part_0000.npz 형태의 분할 파일을 정렬해서 반환합니다.
    return sorted(part_dir.glob(f"part_*{suffix}"))


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

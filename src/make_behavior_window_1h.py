# =========================================================
# make_behavior_window_1h.py
# 행동 분석용 1시간 window 학습 데이터 생성
#
# 입력:
#   output/behavior_preprocessed/{train_parts, valid_parts}/part_*.csv
#
# 출력:
#   output/behavior_window_1h/{train_parts, valid_parts}/part_*.npz
#   output/behavior_window_1h/window_summary.json
#   output/behavior_window_1h/window_config.json
#
# 핵심 구조:
#   - 10분 간격 데이터 기준 1시간 window = 6 rows
#   - sample_key 기준 정렬
#   - time gap > 1시간이면 segment 분리
#   - 각 window의 마지막 row target_behavior_id를 y로 사용
# =========================================================

import json
import shutil
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd


# =========================================================
# 1. Config
# =========================================================
PROJECT_ROOT = Path(__file__).parent

INPUT_ROOT = PROJECT_ROOT / "output" / "behavior_preprocessed"
INPUT_TRAIN_DIR = INPUT_ROOT / "train_parts"
INPUT_VALID_DIR = INPUT_ROOT / "valid_parts"

SAVE_ROOT = PROJECT_ROOT / "output" / "behavior_window_1h"
SAVE_TRAIN_DIR = SAVE_ROOT / "train_parts"
SAVE_VALID_DIR = SAVE_ROOT / "valid_parts"

WINDOW_SIZE = 6          # 10분 단위 x 6 = 1시간
STRIDE = 1               # 10분씩 이동
GAP_SEC = 3600           # 1시간 초과 gap이면 segment 분리
MIN_SEGMENT_LEN = WINDOW_SIZE

TARGET_COL = "target_behavior_id"

FEATURE_COLS = [
    "em_temperature",
    "em_humidity",
    "em_illuminance",
    "em_activity_ir",
    "em_co2",
    "em_tvoc",
    "em_activity_ir_log1p",
    "em_co2_log1p",
    "em_tvoc_log1p",
    "em_illuminance_log1p",
    "hour",
    "weekday",
    "age",
    "gender_id",
    "environment_id",
]

DTYPE_X = np.float32
DTYPE_Y = np.int64


# =========================================================
# 2. Helper
# =========================================================
def reset_output_dirs():
    if SAVE_ROOT.exists():
        shutil.rmtree(SAVE_ROOT)
    SAVE_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_VALID_DIR.mkdir(parents=True, exist_ok=True)


def list_part_files(part_dir: Path) -> List[Path]:
    return sorted(part_dir.glob("part_*.csv"))


def read_csv_safe(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    except Exception:
        return pd.read_csv(path, encoding="utf-8", low_memory=False)


def save_json(obj: dict, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def check_required_columns(df: pd.DataFrame, path: Path):
    required = ["sample_key", "timestamp", TARGET_COL] + FEATURE_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")


def add_segment_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    sample_key 내부에서 time_diff가 GAP_SEC를 초과하면 새로운 segment로 분리
    """
    sort_cols = ["sample_key", "timestamp"]
    if "seq_num" in df.columns:
        sort_cols.append("seq_num")

    df = df.sort_values(sort_cols).copy()

    df["time_diff_sec"] = df.groupby("sample_key")["timestamp"].diff().dt.total_seconds()
    df["new_segment"] = (df["time_diff_sec"] > GAP_SEC).fillna(False).astype("int32")

    df["segment_no"] = df.groupby("sample_key")["new_segment"].cumsum().astype("int32")
    df["segment_key"] = df["sample_key"].astype(str) + "_seg" + df["segment_no"].astype(str)

    return df


def make_windows_from_segment(g: pd.DataFrame):
    """
    하나의 segment에서 window 생성
    X shape: (num_windows, WINDOW_SIZE, num_features)
    y shape: (num_windows,)
    meta: window 마지막 시점 기준 sample_key, segment_key, timestamp
    """
    if len(g) < WINDOW_SIZE:
        return None

    x_arr = g[FEATURE_COLS].to_numpy(dtype=DTYPE_X)
    y_arr = g[TARGET_COL].to_numpy(dtype=DTYPE_Y)

    windows = []
    labels = []
    end_sample_keys = []
    end_segment_keys = []
    end_timestamps = []

    for start in range(0, len(g) - WINDOW_SIZE + 1, STRIDE):
        end = start + WINDOW_SIZE
        windows.append(x_arr[start:end])
        labels.append(y_arr[end - 1])

        end_row = g.iloc[end - 1]
        end_sample_keys.append(str(end_row["sample_key"]))
        end_segment_keys.append(str(end_row["segment_key"]))
        end_timestamps.append(str(end_row["timestamp"]))

    if len(windows) == 0:
        return None

    return {
        "X": np.stack(windows).astype(DTYPE_X),
        "y": np.array(labels, dtype=DTYPE_Y),
        "sample_key": np.array(end_sample_keys),
        "segment_key": np.array(end_segment_keys),
        "timestamp": np.array(end_timestamps),
    }


def preprocess_one_file(path: Path, save_path: Path, split_name: str) -> Dict:
    print(f"[WINDOW {split_name.upper()}] {path.name}")

    df = read_csv_safe(path)
    input_rows = len(df)

    check_required_columns(df, path)

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df[df["timestamp"].notna()].copy()
    df = df[df[TARGET_COL].notna()].copy()
    df = df[df[TARGET_COL] != -1].copy()

    # feature numeric 변환
    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # feature 결측 처리
    # 이미 이전 단계에서 대부분 정리되어 있어야 하지만 window 생성 전 안전하게 처리
    df[FEATURE_COLS] = df[FEATURE_COLS].ffill().bfill()
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL]).copy()

    if len(df) == 0:
        return {
            "file": path.name,
            "input_rows": input_rows,
            "usable_rows": 0,
            "num_segments": 0,
            "num_windows": 0,
            "saved": False,
        }

    df = add_segment_id(df)

    windows_all = []
    labels_all = []
    sample_keys_all = []
    segment_keys_all = []
    timestamps_all = []

    segment_lengths = []
    skipped_short_segments = 0

    for segment_key, g in df.groupby("segment_key", sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True)
        segment_lengths.append(len(g))

        if len(g) < MIN_SEGMENT_LEN:
            skipped_short_segments += 1
            continue

        result = make_windows_from_segment(g)
        if result is None:
            skipped_short_segments += 1
            continue

        windows_all.append(result["X"])
        labels_all.append(result["y"])
        sample_keys_all.append(result["sample_key"])
        segment_keys_all.append(result["segment_key"])
        timestamps_all.append(result["timestamp"])

    if len(windows_all) == 0:
        return {
            "file": path.name,
            "input_rows": input_rows,
            "usable_rows": len(df),
            "num_segments": int(df["segment_key"].nunique()),
            "num_windows": 0,
            "skipped_short_segments": skipped_short_segments,
            "saved": False,
        }

    X = np.concatenate(windows_all, axis=0)
    y = np.concatenate(labels_all, axis=0)
    sample_key = np.concatenate(sample_keys_all, axis=0)
    segment_key = np.concatenate(segment_keys_all, axis=0)
    timestamp = np.concatenate(timestamps_all, axis=0)

    np.savez_compressed(
        save_path,
        X=X,
        y=y,
        sample_key=sample_key,
        segment_key=segment_key,
        timestamp=timestamp,
        feature_cols=np.array(FEATURE_COLS),
        window_size=np.array([WINDOW_SIZE]),
        stride=np.array([STRIDE]),
        gap_sec=np.array([GAP_SEC]),
    )

    y_unique, y_counts = np.unique(y, return_counts=True)
    label_dist = {str(int(k)): int(v) for k, v in zip(y_unique, y_counts)}

    summary = {
        "file": path.name,
        "input_rows": input_rows,
        "usable_rows": len(df),
        "num_segments": int(df["segment_key"].nunique()),
        "min_segment_len": int(np.min(segment_lengths)) if segment_lengths else 0,
        "max_segment_len": int(np.max(segment_lengths)) if segment_lengths else 0,
        "mean_segment_len": float(np.mean(segment_lengths)) if segment_lengths else 0,
        "skipped_short_segments": int(skipped_short_segments),
        "num_windows": int(len(y)),
        "X_shape": list(X.shape),
        "y_shape": list(y.shape),
        "label_distribution": label_dist,
        "saved": True,
        "save_path": str(save_path),
    }

    print(f"  -> saved: {save_path.name} | X={X.shape} | y={y.shape}")
    return summary


def preprocess_split(files: List[Path], save_dir: Path, split_name: str) -> Dict:
    split_summary = {
        "split": split_name,
        "num_input_parts": len(files),
        "num_output_parts": 0,
        "input_rows": 0,
        "usable_rows": 0,
        "num_segments": 0,
        "num_windows": 0,
        "files": [],
        "label_distribution": {},
    }

    for path in files:
        save_path = save_dir / path.with_suffix(".npz").name

        file_summary = preprocess_one_file(
            path=path,
            save_path=save_path,
            split_name=split_name,
        )

        split_summary["files"].append(file_summary)
        split_summary["input_rows"] += file_summary.get("input_rows", 0)
        split_summary["usable_rows"] += file_summary.get("usable_rows", 0)
        split_summary["num_segments"] += file_summary.get("num_segments", 0)
        split_summary["num_windows"] += file_summary.get("num_windows", 0)

        if file_summary.get("saved", False):
            split_summary["num_output_parts"] += 1

        for k, v in file_summary.get("label_distribution", {}).items():
            split_summary["label_distribution"][k] = split_summary["label_distribution"].get(k, 0) + v

    return split_summary


# =========================================================
# 3. Main
# =========================================================
def main():
    reset_output_dirs()

    train_files = list_part_files(INPUT_TRAIN_DIR)
    valid_files = list_part_files(INPUT_VALID_DIR)

    print("[INFO] train files:", len(train_files))
    print("[INFO] valid files:", len(valid_files))
    print("[INFO] window_size:", WINDOW_SIZE)
    print("[INFO] stride:", STRIDE)
    print("[INFO] gap_sec:", GAP_SEC)

    if len(train_files) == 0:
        raise FileNotFoundError(f"train part 파일이 없습니다: {INPUT_TRAIN_DIR}")
    if len(valid_files) == 0:
        print(f"[WARN] valid part 파일이 없습니다: {INPUT_VALID_DIR}")

    window_config = {
        "task": "behavior_sequence_classification",
        "input_root": str(INPUT_ROOT),
        "save_root": str(SAVE_ROOT),
        "window_name": "1h",
        "window_size": WINDOW_SIZE,
        "window_description": "10분 간격 데이터 기준 6 rows = 1 hour",
        "stride": STRIDE,
        "gap_sec": GAP_SEC,
        "gap_rule": "time_diff_sec > gap_sec이면 새로운 segment로 분리",
        "target_col": TARGET_COL,
        "feature_cols": FEATURE_COLS,
        "output_format": "npz",
        "X_shape": "(num_windows, window_size, num_features)",
        "y_shape": "(num_windows,)",
    }

    train_summary = preprocess_split(
        files=train_files,
        save_dir=SAVE_TRAIN_DIR,
        split_name="train",
    )

    valid_summary = preprocess_split(
        files=valid_files,
        save_dir=SAVE_VALID_DIR,
        split_name="valid",
    )

    final_summary = {
        "train_summary": train_summary,
        "valid_summary": valid_summary,
    }

    save_json(window_config, SAVE_ROOT / "window_config.json")
    save_json(final_summary, SAVE_ROOT / "window_summary.json")

    print("\n[FINISH]")
    print("save root:", SAVE_ROOT)
    print(json.dumps(final_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

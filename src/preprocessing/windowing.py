from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from utils.io import read_csv_safe


# ---------------------------------------------------------
# 입력 컬럼 검증
# ---------------------------------------------------------
# window 생성에 필요한 시간, 라벨, feature 컬럼이 모두 있는지 먼저 확인합니다.
def check_required_columns(df: pd.DataFrame, path: Path, target_col: str, feature_cols: list[str]) -> None:
    missing = [col for col in ["sample_key", "timestamp", target_col] + feature_cols if col not in df.columns]
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")


def add_segment_id(df: pd.DataFrame, gap_sec: int) -> pd.DataFrame:
    # 같은 sample 안에서도 시간 간격이 너무 크게 벌어지면 연속 시계열로 보기 어렵습니다.
    # gap_sec보다 큰 시간 차이가 나오면 새로운 segment로 분리합니다.
    sort_cols = ["sample_key", "timestamp"]
    if "seq_num" in df.columns:
        sort_cols.append("seq_num")

    df = df.sort_values(sort_cols).copy()
    df["time_diff_sec"] = df.groupby("sample_key")["timestamp"].diff().dt.total_seconds()
    df["new_segment"] = (df["time_diff_sec"] > gap_sec).fillna(False).astype("int32")
    df["segment_no"] = df.groupby("sample_key")["new_segment"].cumsum().astype("int32")
    df["segment_key"] = df["sample_key"].astype(str) + "_seg" + df["segment_no"].astype(str)
    return df


def make_windows_from_segment(
    segment: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_col: str,
    window_size: int,
    stride: int,
):
    # 하나의 연속 segment에서 sliding window를 만듭니다.
    # 각 window의 라벨은 window 마지막 시점의 target 값을 사용합니다.
    if len(segment) < window_size:
        return None

    x_arr = segment[feature_cols].to_numpy(dtype=np.float32)
    y_arr = segment[target_col].to_numpy(dtype=np.int64)
    windows, labels, sample_keys, segment_keys, timestamps = [], [], [], [], []

    for start in range(0, len(segment) - window_size + 1, stride):
        end = start + window_size
        end_row = segment.iloc[end - 1]
        windows.append(x_arr[start:end])
        labels.append(y_arr[end - 1])
        sample_keys.append(str(end_row["sample_key"]))
        segment_keys.append(str(end_row["segment_key"]))
        timestamps.append(str(end_row["timestamp"]))

    if not windows:
        return None

    return {
        "X": np.stack(windows).astype(np.float32),
        "y": np.array(labels, dtype=np.int64),
        "sample_key": np.array(sample_keys),
        "segment_key": np.array(segment_keys),
        "timestamp": np.array(timestamps),
    }


def make_window_file(path: Path, save_path: Path, split_name: str, config: dict) -> dict:
    # CSV part 하나를 읽어 여러 개의 LSTM 학습용 window로 변환합니다.
    target_col = config["window"]["target_col"]
    feature_cols = config["window"]["feature_cols"]
    window_size = int(config["window"]["size"])
    stride = int(config["window"]["stride"])
    gap_sec = int(config["window"]["gap_sec"])

    print(f"[WINDOW {split_name.upper()}] {path.name}")
    df = read_csv_safe(path, low_memory=False)
    input_rows = len(df)
    check_required_columns(df, path, target_col, feature_cols)

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df[df["timestamp"].notna()].copy()
    df = df[df[target_col].notna()].copy()
    df = df[df[target_col] != -1].copy()

    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # window 생성 직전 남은 결측 feature는 앞뒤 값으로 보정합니다.
    df[feature_cols] = df[feature_cols].ffill().bfill()
    df = df.dropna(subset=feature_cols + [target_col]).copy()

    if len(df) == 0:
        return {"file": path.name, "input_rows": input_rows, "num_windows": 0, "saved": False}

    df = add_segment_id(df, gap_sec)
    outputs = []
    segment_lengths = []
    skipped_short_segments = 0

    for _, segment in df.groupby("segment_key", sort=False):
        segment = segment.sort_values("timestamp").reset_index(drop=True)
        segment_lengths.append(len(segment))
        result = make_windows_from_segment(
            segment,
            feature_cols=feature_cols,
            target_col=target_col,
            window_size=window_size,
            stride=stride,
        )
        if result is None:
            skipped_short_segments += 1
            continue
        outputs.append(result)

    if not outputs:
        return {
            "file": path.name,
            "input_rows": input_rows,
            "usable_rows": len(df),
            "num_segments": int(df["segment_key"].nunique()),
            "num_windows": 0,
            "skipped_short_segments": skipped_short_segments,
            "saved": False,
        }

    X = np.concatenate([item["X"] for item in outputs], axis=0)
    y = np.concatenate([item["y"] for item in outputs], axis=0)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        # X: (num_windows, window_size, num_features)
        # y: (num_windows,)
        save_path,
        X=X,
        y=y,
        sample_key=np.concatenate([item["sample_key"] for item in outputs], axis=0),
        segment_key=np.concatenate([item["segment_key"] for item in outputs], axis=0),
        timestamp=np.concatenate([item["timestamp"] for item in outputs], axis=0),
        feature_cols=np.array(feature_cols),
        window_size=np.array([window_size]),
        stride=np.array([stride]),
        gap_sec=np.array([gap_sec]),
    )

    y_unique, y_counts = np.unique(y, return_counts=True)
    return {
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
        "label_distribution": {str(int(k)): int(v) for k, v in zip(y_unique, y_counts)},
        "saved": True,
        "save_path": str(save_path),
    }

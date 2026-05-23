# =========================================================
# preprocessing.py
# 행동 분석용 전처리 복사본 생성
# - 입력: output/risk_detection_master/{train_parts, valid_parts}/*.csv
# - 출력: output/behavior_preprocessed/{train_parts, valid_parts}/*.csv
# - 타깃: em_label (sample 내부 ffill/bfill 적용)
# - 사용 데이터: B 데이터만
# =========================================================

import json
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


# =========================================================
# 1. Config
# =========================================================
PROJECT_ROOT = Path(__file__).parent

RAW_ROOT = PROJECT_ROOT / "output" / "risk_detection_master"
TRAIN_PART_DIR = RAW_ROOT / "train_parts"
VALID_PART_DIR = RAW_ROOT / "valid_parts"

SAVE_ROOT = PROJECT_ROOT / "output" / "behavior_preprocessed"
SAVE_TRAIN_DIR = SAVE_ROOT / "train_parts"
SAVE_VALID_DIR = SAVE_ROOT / "valid_parts"

USE_ONLY_B = True
CHUNKSIZE = 200_000

BASE_KEEP_COLS = [
    "sample_key",
    "split",
    "person_id",
    "sex",
    "seq",
    "seq_num",
    "timestamp",
    "age",
    "gender",
    "environment",
    "em_temperature",
    "em_humidity",
    "em_illuminance",
    "em_activity_ir",
    "em_co2",
    "em_tvoc",
    "em_label",
]

FINAL_FEATURE_COLS = [
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

TARGET_COL = "target_behavior"
TARGET_ID_COL = "target_behavior_id"


# =========================================================
# 2. Helper
# =========================================================
def reset_output_dirs():
    if SAVE_ROOT.exists():
        shutil.rmtree(SAVE_ROOT)
    SAVE_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_VALID_DIR.mkdir(parents=True, exist_ok=True)


def read_csv_safe(path: Path, **kwargs) -> pd.DataFrame:
    base_kwargs = {
        "engine": "python",
        "on_bad_lines": "warn",
    }
    base_kwargs.update(kwargs)

    try:
        return pd.read_csv(path, encoding="utf-8-sig", **base_kwargs)
    except Exception:
        return pd.read_csv(path, encoding="utf-8", **base_kwargs)


def list_part_files(part_dir: Path) -> List[Path]:
    return sorted(part_dir.glob("part_*.csv"))


def save_json(obj: dict, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def build_mapping_from_values(values: List[str]) -> Dict[str, int]:
    values = sorted({str(v) for v in values if pd.notna(v) and str(v) != ""})
    return {v: i for i, v in enumerate(values)}


def encode_with_mapping(series: pd.Series, mapping: Dict[str, int], unknown_value: int = -1) -> pd.Series:
    s = series.fillna("").astype(str)
    return s.map(lambda x: mapping.get(x, unknown_value)).astype("int32")


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def fill_behavior_label_within_sample(df: pd.DataFrame) -> pd.DataFrame:
    """
    sample 내에서 em_label forward fill / backward fill
    """
    df = df.sort_values(["sample_key", "timestamp", "seq_num"]).copy()

    df["em_label"] = df["em_label"].replace("", np.nan)
    df["em_label"] = df.groupby("sample_key")["em_label"].ffill().bfill()

    # sample 전체가 NaN인 경우만 제거
    df = df[df["em_label"].notna()].copy()
    return df


# =========================================================
# 3. Mapping build (train only)
# =========================================================
def collect_train_mappings(train_files: List[Path]) -> Dict[str, Dict[str, int]]:
    gender_values = []
    environment_values = []
    behavior_values = []

    usecols = ["person_id", "gender", "environment", "em_label", "sample_key", "timestamp", "seq_num"]

    for f in train_files:
        print(f"[SCAN MAPPING] {f.name}")
        df = read_csv_safe(f, usecols=usecols)

        if USE_ONLY_B:
            df = df[df["person_id"].astype(str).str.startswith("B")].copy()

        if len(df) == 0:
            continue

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["em_label"] = df["em_label"].fillna("").astype(str).str.strip()
        df = fill_behavior_label_within_sample(df)

        if len(df) == 0:
            continue

        gender_values.extend(df["gender"].dropna().astype(str).tolist())
        environment_values.extend(df["environment"].dropna().astype(str).tolist())
        behavior_values.extend(df["em_label"].dropna().astype(str).tolist())

    return {
        "gender_mapping": build_mapping_from_values(gender_values),
        "environment_mapping": build_mapping_from_values(environment_values),
        "behavior_mapping": build_mapping_from_values(behavior_values),
    }


# =========================================================
# 4. Chunk preprocess
# =========================================================
def preprocess_chunk(
    df: pd.DataFrame,
    gender_mapping: Dict[str, int],
    environment_mapping: Dict[str, int],
    behavior_mapping: Dict[str, int],
    split_name: str,
) -> pd.DataFrame:
    existing_cols = [c for c in BASE_KEEP_COLS if c in df.columns]
    df = df[existing_cols].copy()

    if USE_ONLY_B and "person_id" in df.columns:
        df = df[df["person_id"].astype(str).str.startswith("B")].copy()

    if len(df) == 0:
        return df

    # timestamp 처리
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # em_label 정리 + sample 내부 채우기
    df["em_label"] = df["em_label"].fillna("").astype(str).str.strip()
    df = fill_behavior_label_within_sample(df)

    if len(df) == 0:
        return df

    # 시간 feature
    df["hour"] = df["timestamp"].dt.hour.astype("float32")
    df["weekday"] = df["timestamp"].dt.weekday.astype("float32")

    # 수치형 변환
    numeric_cols = [
        "age",
        "em_temperature",
        "em_humidity",
        "em_illuminance",
        "em_activity_ir",
        "em_co2",
        "em_tvoc",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = safe_numeric(df[col])

    # 기본 범위 정리
    if "age" in df.columns:
        df.loc[(df["age"] < 0) | (df["age"] > 120), "age"] = np.nan

    if "em_humidity" in df.columns:
        df.loc[(df["em_humidity"] < 0) | (df["em_humidity"] > 100), "em_humidity"] = np.nan

    for col in ["em_illuminance", "em_activity_ir", "em_co2", "em_tvoc"]:
        if col in df.columns:
            df.loc[df[col] < 0, col] = np.nan

    # 로그 변환
    df["em_activity_ir_log1p"] = np.log1p(df["em_activity_ir"].clip(lower=0))
    df["em_co2_log1p"] = np.log1p(df["em_co2"].clip(lower=0))
    df["em_tvoc_log1p"] = np.log1p(df["em_tvoc"].clip(lower=0))
    df["em_illuminance_log1p"] = np.log1p(df["em_illuminance"].clip(lower=0))

    # 범주형 인코딩
    df["gender_id"] = encode_with_mapping(df["gender"], gender_mapping)
    df["environment_id"] = encode_with_mapping(df["environment"], environment_mapping)

    # 타깃
    df[TARGET_COL] = df["em_label"].astype(str)
    df[TARGET_ID_COL] = encode_with_mapping(df[TARGET_COL], behavior_mapping)

    df["split"] = split_name

    keep_cols = [
        "sample_key",
        "split",
        "person_id",
        "seq",
        "seq_num",
        "timestamp",
        TARGET_COL,
        TARGET_ID_COL,
    ] + FINAL_FEATURE_COLS

    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    return df


# =========================================================
# 5. Split preprocess
# =========================================================
def preprocess_split(
    files: List[Path],
    save_dir: Path,
    split_name: str,
    gender_mapping: Dict[str, int],
    environment_mapping: Dict[str, int],
    behavior_mapping: Dict[str, int],
) -> dict:
    summary = {
        "split": split_name,
        "num_input_parts": len(files),
        "num_output_parts": 0,
        "input_rows": 0,
        "output_rows": 0,
        "b_rows_kept": 0,
        "target_non_null_rows": 0,
    }

    for f in files:
        print(f"[PREPROCESS {split_name.upper()}] {f.name}")

        out_path = save_dir / f.name
        input_rows_this_file = 0
        output_rows_this_file = 0
        out_chunks = []

        for chunk in read_csv_safe(f, chunksize=CHUNKSIZE):
            input_rows_this_file += len(chunk)

            processed = preprocess_chunk(
                df=chunk,
                gender_mapping=gender_mapping,
                environment_mapping=environment_mapping,
                behavior_mapping=behavior_mapping,
                split_name=split_name,
            )

            if len(processed) > 0:
                out_chunks.append(processed)
                output_rows_this_file += len(processed)

        summary["input_rows"] += input_rows_this_file
        summary["output_rows"] += output_rows_this_file
        summary["b_rows_kept"] += output_rows_this_file
        summary["target_non_null_rows"] += output_rows_this_file

        if output_rows_this_file > 0:
            out_df = pd.concat(out_chunks, ignore_index=True)
            out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
            summary["num_output_parts"] += 1
            print(
                f"  -> saved: {out_path.name} | "
                f"input_rows={input_rows_this_file:,} | output_rows={output_rows_this_file:,}"
            )
        else:
            print(
                f"  -> skipped: {f.name} | "
                f"input_rows={input_rows_this_file:,} | output_rows=0"
            )

    return summary


# =========================================================
# 6. Main
# =========================================================
def main():
    reset_output_dirs()

    train_files = list_part_files(TRAIN_PART_DIR)
    valid_files = list_part_files(VALID_PART_DIR)

    print("[INFO] train files:", len(train_files))
    print("[INFO] valid files:", len(valid_files))
    print("[INFO] USE_ONLY_B:", USE_ONLY_B)

    if len(train_files) == 0:
        raise FileNotFoundError(f"train part 파일이 없습니다: {TRAIN_PART_DIR}")

    mappings = collect_train_mappings(train_files)
    gender_mapping = mappings["gender_mapping"]
    environment_mapping = mappings["environment_mapping"]
    behavior_mapping = mappings["behavior_mapping"]

    print("[INFO] gender_mapping:", gender_mapping)
    print("[INFO] environment_mapping:", environment_mapping)
    print("[INFO] behavior_mapping:", behavior_mapping)

    train_summary = preprocess_split(
        files=train_files,
        save_dir=SAVE_TRAIN_DIR,
        split_name="train",
        gender_mapping=gender_mapping,
        environment_mapping=environment_mapping,
        behavior_mapping=behavior_mapping,
    )

    valid_summary = preprocess_split(
        files=valid_files,
        save_dir=SAVE_VALID_DIR,
        split_name="valid",
        gender_mapping=gender_mapping,
        environment_mapping=environment_mapping,
        behavior_mapping=behavior_mapping,
    )

    feature_info = {
        "target_col": TARGET_COL,
        "target_id_col": TARGET_ID_COL,
        "feature_cols": FINAL_FEATURE_COLS,
        "use_only_b": USE_ONLY_B,
        "label_fill_strategy": "groupby(sample_key).ffill().bfill()",
        "task": "behavior_classification",
        "task_description": "환경 + 시간 + 메타로 em_label(행동 상태) 예측",
    }

    preprocess_summary = {
        "train_summary": train_summary,
        "valid_summary": valid_summary,
    }

    save_json(feature_info, SAVE_ROOT / "feature_columns.json")
    save_json(mappings, SAVE_ROOT / "category_mappings.json")
    save_json(preprocess_summary, SAVE_ROOT / "preprocess_summary.json")

    print("\n[FINISH]")
    print("save root:", SAVE_ROOT)
    print(json.dumps(preprocess_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
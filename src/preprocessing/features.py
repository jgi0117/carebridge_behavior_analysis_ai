from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from utils.io import read_csv_safe


def build_mapping_from_values(values: list[str]) -> dict[str, int]:
    cleaned = sorted({str(v) for v in values if pd.notna(v) and str(v) != ""})
    return {value: idx for idx, value in enumerate(cleaned)}


def encode_with_mapping(
    series: pd.Series,
    mapping: dict[str, int],
    unknown_value: int = -1,
) -> pd.Series:
    return series.fillna("").astype(str).map(lambda x: mapping.get(x, unknown_value)).astype("int32")


def fill_behavior_label_within_sample(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["sample_key", "timestamp", "seq_num"]).copy()
    df["em_label"] = df["em_label"].replace("", np.nan)
    df["em_label"] = df.groupby("sample_key")["em_label"].ffill().bfill()
    return df[df["em_label"].notna()].copy()


def collect_train_mappings(
    train_files: list[Path],
    *,
    use_only_b: bool,
) -> Dict[str, Dict[str, int]]:
    gender_values: list[str] = []
    environment_values: list[str] = []
    behavior_values: list[str] = []
    usecols = ["person_id", "gender", "environment", "em_label", "sample_key", "timestamp", "seq_num"]

    for path in train_files:
        print(f"[SCAN MAPPING] {path.name}")
        df = read_csv_safe(path, usecols=usecols, engine="python", on_bad_lines="warn")

        if use_only_b:
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


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["hour"] = df["timestamp"].dt.hour.astype("float32")
    df["weekday"] = df["timestamp"].dt.weekday.astype("float32")
    return df


def clean_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
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
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "age" in df.columns:
        df.loc[(df["age"] < 0) | (df["age"] > 120), "age"] = np.nan
    if "em_humidity" in df.columns:
        df.loc[(df["em_humidity"] < 0) | (df["em_humidity"] > 100), "em_humidity"] = np.nan

    for col in ["em_illuminance", "em_activity_ir", "em_co2", "em_tvoc"]:
        if col in df.columns:
            df.loc[df[col] < 0, col] = np.nan
    return df


def add_log_features(df: pd.DataFrame) -> pd.DataFrame:
    df["em_activity_ir_log1p"] = np.log1p(df["em_activity_ir"].clip(lower=0))
    df["em_co2_log1p"] = np.log1p(df["em_co2"].clip(lower=0))
    df["em_tvoc_log1p"] = np.log1p(df["em_tvoc"].clip(lower=0))
    df["em_illuminance_log1p"] = np.log1p(df["em_illuminance"].clip(lower=0))
    return df


def preprocess_chunk(
    df: pd.DataFrame,
    *,
    base_keep_cols: list[str],
    feature_cols: list[str],
    target_col: str,
    target_id_col: str,
    use_only_b: bool,
    gender_mapping: dict[str, int],
    environment_mapping: dict[str, int],
    behavior_mapping: dict[str, int],
    split_name: str,
) -> pd.DataFrame:
    existing_cols = [col for col in base_keep_cols if col in df.columns]
    df = df[existing_cols].copy()

    if use_only_b and "person_id" in df.columns:
        df = df[df["person_id"].astype(str).str.startswith("B")].copy()
    if len(df) == 0:
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["em_label"] = df["em_label"].fillna("").astype(str).str.strip()
    df = fill_behavior_label_within_sample(df)
    if len(df) == 0:
        return df

    df = add_time_features(df)
    df = clean_numeric_features(df)
    df = add_log_features(df)
    df["gender_id"] = encode_with_mapping(df["gender"], gender_mapping)
    df["environment_id"] = encode_with_mapping(df["environment"], environment_mapping)
    df[target_col] = df["em_label"].astype(str)
    df[target_id_col] = encode_with_mapping(df[target_col], behavior_mapping)
    df["split"] = split_name

    keep_cols = [
        "sample_key",
        "split",
        "person_id",
        "seq",
        "seq_num",
        "timestamp",
        target_col,
        target_id_col,
    ] + feature_cols
    return df[[col for col in keep_cols if col in df.columns]].copy()

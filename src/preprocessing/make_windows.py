from __future__ import annotations

import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preprocessing.windowing import make_window_file
from utils.config import configured_path, load_config
from utils.io import list_part_files, reset_dir, save_json


# ---------------------------------------------------------
# split 단위 window 생성
# ---------------------------------------------------------
# 전처리된 CSV part를 읽어 LSTM이 바로 학습할 수 있는 npz 파일로 저장합니다.
def make_split(files: list[Path], save_dir: Path, split_name: str, config: dict) -> dict:
    summary = {
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
        file_summary = make_window_file(
            path=path,
            save_path=save_dir / path.with_suffix(".npz").name,
            split_name=split_name,
            config=config,
        )
        summary["files"].append(file_summary)
        for key in ["input_rows", "usable_rows", "num_segments", "num_windows"]:
            summary[key] += file_summary.get(key, 0)
        if file_summary.get("saved"):
            summary["num_output_parts"] += 1
        for key, value in file_summary.get("label_distribution", {}).items():
            summary["label_distribution"][key] = summary["label_distribution"].get(key, 0) + value

    return summary


def main() -> None:
    # -----------------------------------------------------
    # 설정 및 입출력 경로 준비
    # -----------------------------------------------------
    config = load_config("paths.yaml", "preprocessing.yaml")
    input_root = configured_path("preprocessed", config)
    save_root = configured_path("window_data", config)
    save_train_dir = save_root / "train_parts"
    save_valid_dir = save_root / "valid_parts"

    reset_dir(save_train_dir)
    reset_dir(save_valid_dir)

    train_files = list_part_files(input_root / "train_parts")
    valid_files = list_part_files(input_root / "valid_parts")
    if not train_files:
        raise FileNotFoundError(f"train part files not found: {input_root / 'train_parts'}")

    train_summary = make_split(train_files, save_train_dir, "train", config)
    valid_summary = make_split(valid_files, save_valid_dir, "valid", config)

    save_json(
        {
            "task": "behavior_sequence_classification",
            "input_root": str(input_root),
            "save_root": str(save_root),
            "window_name": config["window"]["name"],
            "window_size": config["window"]["size"],
            "stride": config["window"]["stride"],
            "gap_sec": config["window"]["gap_sec"],
            "target_col": config["window"]["target_col"],
            "feature_cols": config["window"]["feature_cols"],
            "output_format": "npz",
        },
        save_root / "window_config.json",
    )
    save_json({"train_summary": train_summary, "valid_summary": valid_summary}, save_root / "window_summary.json")
    print("[FINISH] save root:", save_root)


if __name__ == "__main__":
    main()

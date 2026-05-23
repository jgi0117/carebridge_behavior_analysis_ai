from __future__ import annotations

import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preprocessing.features import collect_train_mappings, preprocess_chunk
from utils.config import configured_path, load_config
from utils.io import list_part_files, reset_dir, save_json, read_csv_safe


# ---------------------------------------------------------
# split 단위 전처리
# ---------------------------------------------------------
# train_parts, valid_parts에 있는 part_*.csv를 순회하면서 동일한 feature 정리를 적용합니다.
def preprocess_split(
    files: list[Path],
    save_dir: Path,
    split_name: str,
    config: dict,
    mappings: dict,
) -> dict:
    summary = {
        "split": split_name,
        "num_input_parts": len(files),
        "num_output_parts": 0,
        "input_rows": 0,
        "output_rows": 0,
    }

    preprocessing_cfg = config["preprocessing"]
    window_cfg = config["window"]

    for path in files:
        print(f"[PREPROCESS {split_name.upper()}] {path.name}")
        out_path = save_dir / path.name
        input_rows = 0
        output_rows = 0
        chunks = []

        for chunk in read_csv_safe(
            path,
            chunksize=int(preprocessing_cfg["chunksize"]),
            engine="python",
            on_bad_lines="warn",
        ):
            # 대용량 CSV를 고려해 chunksize 단위로 읽고 처리합니다.
            input_rows += len(chunk)
            processed = preprocess_chunk(
                chunk,
                base_keep_cols=preprocessing_cfg["base_keep_cols"],
                feature_cols=window_cfg["feature_cols"],
                target_col=preprocessing_cfg["target_col"],
                target_id_col=preprocessing_cfg["target_id_col"],
                use_only_b=bool(preprocessing_cfg["use_only_b"]),
                gender_mapping=mappings["gender_mapping"],
                environment_mapping=mappings["environment_mapping"],
                behavior_mapping=mappings["behavior_mapping"],
                split_name=split_name,
            )
            if len(processed) > 0:
                chunks.append(processed)
                output_rows += len(processed)

        summary["input_rows"] += input_rows
        summary["output_rows"] += output_rows

        if output_rows > 0:
            import pandas as pd

            pd.concat(chunks, ignore_index=True).to_csv(out_path, index=False, encoding="utf-8-sig")
            summary["num_output_parts"] += 1
            print(f"  -> saved: {out_path.name} | input_rows={input_rows:,} | output_rows={output_rows:,}")
        else:
            print(f"  -> skipped: {path.name} | input_rows={input_rows:,} | output_rows=0")

    return summary


def main() -> None:
    # -----------------------------------------------------
    # 설정 및 입출력 경로 준비
    # -----------------------------------------------------
    config = load_config("paths.yaml", "preprocessing.yaml")
    raw_root = configured_path("risk_master", config)
    save_root = configured_path("preprocessed", config)
    save_train_dir = save_root / "train_parts"
    save_valid_dir = save_root / "valid_parts"

    reset_dir(save_train_dir)
    reset_dir(save_valid_dir)

    train_files = list_part_files(raw_root / "train_parts")
    valid_files = list_part_files(raw_root / "valid_parts")
    print("[INFO] train files:", len(train_files))
    print("[INFO] valid files:", len(valid_files))

    if not train_files:
        raise FileNotFoundError(f"train part files not found: {raw_root / 'train_parts'}")

    # train split에서만 category mapping을 만든 뒤 train/valid에 동일하게 적용합니다.
    mappings = collect_train_mappings(
        train_files,
        use_only_b=bool(config["preprocessing"]["use_only_b"]),
    )

    train_summary = preprocess_split(train_files, save_train_dir, "train", config, mappings)
    valid_summary = preprocess_split(valid_files, save_valid_dir, "valid", config, mappings)

    save_json(
        {
            "target_col": config["preprocessing"]["target_col"],
            "target_id_col": config["preprocessing"]["target_id_col"],
            "feature_cols": config["window"]["feature_cols"],
            "use_only_b": config["preprocessing"]["use_only_b"],
            "label_fill_strategy": "groupby(sample_key).ffill().bfill()",
        },
        save_root / "feature_columns.json",
    )
    save_json(mappings, save_root / "category_mappings.json")
    save_json(
        {"train_summary": train_summary, "valid_summary": valid_summary},
        save_root / "preprocess_summary.json",
    )
    print("[FINISH] save root:", save_root)


if __name__ == "__main__":
    main()

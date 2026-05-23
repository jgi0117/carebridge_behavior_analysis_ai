# =========================================================
# 0. Import
# =========================================================
import json
import re
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import pandas as pd


# =========================================================
# 1. Config
# =========================================================
PROJECT_ROOT = Path(__file__).parent
BASE_DIR = PROJECT_ROOT / "1.데이터"

TRAIN_DIR = BASE_DIR / "Training"
VALID_DIR = BASE_DIR / "Validation"

# 깔끔한 저장 폴더명
SAVE_DIR = PROJECT_ROOT / "output" / "risk_detection_master"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# JSON 몇 개 처리할 때마다 csv part 1개로 저장할지
CHUNK_SIZE = 20


# =========================================================
# 2. Regex / Key Utils
# =========================================================
SRC_FILE_PATTERN = re.compile(r"^I_([AB]\d{4,5})_(\d{2})_([FM])_(\d+)\.csv$")
LBL_FILE_PATTERN = re.compile(r"^L_([AB]\d{4,5})_(\d{2})_([FM])_(\d+)\.json$")
IR_PNG_PATTERN = re.compile(r"^IR_([AB]\d{4,5})_(\d{8})_(\d{4})\.png$")


def sample_key(person_id: str, sex: str, seq: str) -> str:
    return f"{person_id}|{sex}|{seq}"


def parse_source_csv_name(path: Path) -> Optional[dict]:
    m = SRC_FILE_PATTERN.match(path.name)
    if not m:
        return None
    return {
        "person_id": m.group(1),
        "modality_code": m.group(2),
        "sex": m.group(3),
        "seq": m.group(4),
    }


def parse_label_json_name(path: Path) -> Optional[dict]:
    m = LBL_FILE_PATTERN.match(path.name)
    if not m:
        return None
    return {
        "person_id": m.group(1),
        "label_code": m.group(2),
        "sex": m.group(3),
        "seq": m.group(4),
    }


def parse_ir_png_name(path: Path) -> Optional[dict]:
    m = IR_PNG_PATTERN.match(path.name)
    if not m:
        return None
    dt = pd.to_datetime(m.group(2) + m.group(3), format="%Y%m%d%H%M", errors="coerce")
    return {
        "person_id": m.group(1),
        "ir_timestamp": dt,
    }


def load_json_safe(json_path: Path):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except UnicodeDecodeError:
        with open(json_path, "r", encoding="cp949") as f:
            return json.load(f)


def safe_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


# =========================================================
# 3. Directory Helpers
# =========================================================
def get_split_roots(split_dir: Path) -> Tuple[Path, Path]:
    src_root = split_dir / "01.원천데이터"
    lbl_root = split_dir / "02.라벨링데이터"
    return src_root, lbl_root


def count_csv_rows(csv_path: Optional[Path]) -> Optional[int]:
    if csv_path is None or not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
        return len(df)
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="cp949")
        return len(df)
    except Exception:
        return None


def read_csv_columns(csv_path: Optional[Path]) -> Optional[str]:
    if csv_path is None or not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path, nrows=3, encoding="utf-8")
        return "|".join(df.columns.tolist())
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, nrows=3, encoding="cp949")
        return "|".join(df.columns.tolist())
    except Exception:
        return None


# =========================================================
# 4. Build Sample Manifest
# =========================================================
def build_sample_manifest(split_dir: Path, split_name: str) -> pd.DataFrame:
    src_root, lbl_root = get_split_roots(split_dir)

    print("=" * 80)
    print(f"[START] {split_name.upper()} manifest 생성")
    print(f"[SRC ROOT] {src_root}")
    print(f"[LBL ROOT] {lbl_root}")

    buckets: Dict[str, dict] = {}

    src_csv_files = sorted(src_root.rglob("*.csv"))
    print(f"[INFO] source csv 수: {len(src_csv_files)}")

    code_to_col = {
        "01": "env_csv_path",
        "02": "bio_csv_path",
        "03": "ir_csv_path",
        "04": "emergency_csv_path",
        "05": "counsel_csv_path",
        "06": "meta_csv_path",
    }

    for p in src_csv_files:
        info = parse_source_csv_name(p)
        if info is None:
            continue

        key = sample_key(info["person_id"], info["sex"], info["seq"])
        if key not in buckets:
            buckets[key] = {
                "split": split_name,
                "person_id": info["person_id"],
                "sex": info["sex"],
                "seq": info["seq"],
                "env_csv_path": None,
                "bio_csv_path": None,
                "ir_csv_path": None,
                "emergency_csv_path": None,
                "counsel_csv_path": None,
                "meta_csv_path": None,
                "label_json_path": None,
                "ir_image_dir": None,
            }

        col = code_to_col.get(info["modality_code"])
        if col is not None:
            buckets[key][col] = str(p)

        if info["modality_code"] == "03":
            buckets[key]["ir_image_dir"] = str(p.parent)

    lbl_json_files = sorted(lbl_root.rglob("*.json"))
    print(f"[INFO] label json 수: {len(lbl_json_files)}")

    for p in lbl_json_files:
        info = parse_label_json_name(p)
        if info is None:
            continue

        key = sample_key(info["person_id"], info["sex"], info["seq"])
        if key not in buckets:
            buckets[key] = {
                "split": split_name,
                "person_id": info["person_id"],
                "sex": info["sex"],
                "seq": info["seq"],
                "env_csv_path": None,
                "bio_csv_path": None,
                "ir_csv_path": None,
                "emergency_csv_path": None,
                "counsel_csv_path": None,
                "meta_csv_path": None,
                "label_json_path": None,
                "ir_image_dir": None,
            }

        buckets[key]["label_json_path"] = str(p)

    manifest_rows = []
    for key, row in buckets.items():
        row["sample_key"] = key
        row["has_env_csv"] = row["env_csv_path"] is not None
        row["has_bio_csv"] = row["bio_csv_path"] is not None
        row["has_ir_csv"] = row["ir_csv_path"] is not None
        row["has_emergency_csv"] = row["emergency_csv_path"] is not None
        row["has_counsel_csv"] = row["counsel_csv_path"] is not None
        row["has_meta_csv"] = row["meta_csv_path"] is not None
        row["has_label_json"] = row["label_json_path"] is not None
        manifest_rows.append(row)

    manifest_df = pd.DataFrame(manifest_rows)
    if len(manifest_df) == 0:
        print(f"[WARN] {split_name} manifest가 비어 있음")
        return manifest_df

    for col in [
        "env_csv_path", "bio_csv_path", "ir_csv_path",
        "emergency_csv_path", "counsel_csv_path", "meta_csv_path"
    ]:
        manifest_df[f"{col}_rows"] = manifest_df[col].apply(
            lambda x: count_csv_rows(Path(x)) if pd.notna(x) else None
        )
        manifest_df[f"{col}_cols"] = manifest_df[col].apply(
            lambda x: read_csv_columns(Path(x)) if pd.notna(x) else None
        )

    manifest_df = manifest_df.sort_values(["person_id", "sex", "seq"]).reset_index(drop=True)

    print(f"[DONE] {split_name.upper()} manifest shape = {manifest_df.shape}")
    print("=" * 80)
    return manifest_df


# =========================================================
# 5. IR PNG Index
# =========================================================
def build_ir_png_index(src_root: Path) -> Dict[str, str]:
    print(f"[INFO] IR PNG 인덱싱 시작: {src_root}")
    png_map = {}

    png_files = sorted(src_root.rglob("*.png"))
    total = len(png_files)
    print(f"[INFO] PNG 파일 수: {total}")

    for idx, p in enumerate(png_files, start=1):
        png_map[p.name] = str(p)

        if idx % 500 == 0 or idx == total:
            print(f"[PNG INDEX] {idx:,}/{total:,} 완료")

    print("[INFO] IR PNG 인덱싱 완료")
    return png_map


# =========================================================
# 6. Progress / Resume Helpers
# =========================================================
def get_output_paths(split_name: str):
    manifest_path = SAVE_DIR / f"{split_name}_sample_manifest.csv"
    part_dir = SAVE_DIR / f"{split_name}_parts"
    progress_path = SAVE_DIR / f"{split_name}_progress.json"
    error_log_path = SAVE_DIR / f"{split_name}_error_log.txt"
    return manifest_path, part_dir, progress_path, error_log_path


def load_progress(progress_path: Path) -> dict:
    if progress_path.exists():
        try:
            with open(progress_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    return {
        "completed_sample_keys": [],
        "failed_files": [],
        "written_rows": 0,
        "part_idx": 0,
        "last_processed_sample_key": None,
    }


def save_progress(progress: dict, progress_path: Path):
    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def append_error_log(error_log_path: Path, message: str):
    with open(error_log_path, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def get_existing_written_sample_keys_from_parts(part_dir: Path) -> Set[str]:
    """
    progress 파일이 일부 깨졌더라도 csv part 파일을 읽어
    이미 저장된 sample_key를 복원할 수 있게 함
    """
    completed = set()

    if not part_dir.exists():
        return completed

    part_files = sorted(part_dir.glob("part_*.csv"))
    if not part_files:
        return completed

    print(f"[INFO] 기존 csv part 검사: {len(part_files)}개")

    for idx, p in enumerate(part_files, start=1):
        try:
            df = pd.read_csv(p, usecols=["sample_key"], encoding="utf-8-sig")
            completed.update(df["sample_key"].dropna().astype(str).unique().tolist())
        except Exception:
            try:
                df = pd.read_csv(p, usecols=["sample_key"], encoding="utf-8")
                completed.update(df["sample_key"].dropna().astype(str).unique().tolist())
            except Exception as e:
                print(f"[WARN] csv part 읽기 실패: {p} | {repr(e)}")

        if idx % 20 == 0 or idx == len(part_files):
            print(f"[PART SCAN] {idx}/{len(part_files)} 완료")

    return completed


def append_rows_to_csv(rows: List[dict], part_dir: Path, part_idx: int) -> Tuple[int, int]:
    """
    rows -> csv part 1개 저장
    return: (저장한 row 수, 다음 part_idx)
    """
    if not rows:
        return 0, part_idx

    df = pd.DataFrame(rows)
    part_path = part_dir / f"part_{part_idx:04d}.csv"
    df.to_csv(part_path, index=False, encoding="utf-8-sig")

    return len(df), part_idx + 1


# =========================================================
# 7. JSON -> Row-level Master
# =========================================================
def expand_label_json_with_manifest(
    json_path: Path,
    split_name: str,
    manifest_row: dict,
    png_map: Dict[str, str],
) -> List[dict]:
    data = load_json_safe(json_path)

    if not isinstance(data, dict):
        raise TypeError(f"JSON 최상위 구조가 dict가 아님: {type(data).__name__}")

    meta = data.get("MetaData", {})
    ts_list = data.get("TimeSeriesData", [])

    if ts_list is None:
        ts_list = []

    if not isinstance(ts_list, list):
        raise TypeError(f"TimeSeriesData가 list가 아님: {type(ts_list).__name__}")

    person_id = manifest_row["person_id"]
    sex = manifest_row["sex"]
    seq = manifest_row["seq"]

    age = meta.get("Age") if isinstance(meta, dict) else None
    gender = meta.get("Gender", sex) if isinstance(meta, dict) else sex
    region = meta.get("Region") if isinstance(meta, dict) else None
    disease_yn = meta.get("DiseaseYN") if isinstance(meta, dict) else None
    receipt_st = meta.get("ReceiptST") if isinstance(meta, dict) else None
    environment = meta.get("Environment") if isinstance(meta, dict) else None
    model_list = meta.get("ModelList") if isinstance(meta, dict) else None

    raw_unit_env = safe_get(meta, "RawUnit", "Environmental")
    raw_unit_vital = safe_get(meta, "RawUnit", "VitalRing")

    rows = []

    for item in ts_list:
        if not isinstance(item, dict):
            continue

        seq_num = item.get("SeqNum")
        timestamp_raw = item.get("TimeStamp")
        timestamp = pd.to_datetime(timestamp_raw, errors="coerce")

        em = item.get("EM_Sensor", {}) if isinstance(item.get("EM_Sensor", {}), dict) else {}
        sm = item.get("SM_Sensor", {}) if isinstance(item.get("SM_Sensor", {}), dict) else {}
        ir = item.get("IR_Sensor", {}) if isinstance(item.get("IR_Sensor", {}), dict) else {}
        er = item.get("ER_Sensor", {}) if isinstance(item.get("ER_Sensor", {}), dict) else {}
        cs = item.get("Counseling", {}) if isinstance(item.get("Counseling", {}), dict) else {}
        tl = item.get("Total_Labeling", {}) if isinstance(item.get("Total_Labeling", {}), dict) else {}

        image_ir_name = ir.get("Image_IR")
        image_ir_path = png_map.get(image_ir_name) if image_ir_name else None

        ir_meta = parse_ir_png_name(Path(image_ir_name)) if image_ir_name else None
        ir_timestamp = ir_meta["ir_timestamp"] if ir_meta else pd.NaT

        row = {
            "sample_id": f"{split_name}_{person_id}_{sex}_{seq}_{seq_num}",
            "sample_key": manifest_row["sample_key"],
            "split": split_name,
            "person_id": person_id,
            "sex": sex,
            "seq": seq,
            "seq_num": seq_num,
            "timestamp": timestamp,
            "timestamp_raw": timestamp_raw,

            "age": age,
            "gender": gender,
            "region": region,
            "disease_yn": disease_yn,
            "receipt_st": receipt_st,
            "environment": environment,
            "model_list": str(model_list),
            "raw_unit_environmental": raw_unit_env,
            "raw_unit_vitalring": raw_unit_vital,

            "env_csv_path": manifest_row["env_csv_path"],
            "bio_csv_path": manifest_row["bio_csv_path"],
            "ir_csv_path": manifest_row["ir_csv_path"],
            "emergency_csv_path": manifest_row["emergency_csv_path"],
            "counsel_csv_path": manifest_row["counsel_csv_path"],
            "meta_csv_path": manifest_row["meta_csv_path"],
            "label_json_path": manifest_row["label_json_path"],
            "ir_image_dir": manifest_row["ir_image_dir"],

            "image_ir_name": image_ir_name,
            "image_ir_path": image_ir_path,
            "ir_timestamp": ir_timestamp,

            "em_temperature": em.get("Temperature"),
            "em_humidity": em.get("Humidity"),
            "em_illuminance": em.get("Illuminance"),
            "em_activity_ir": em.get("Activity_IR"),
            "em_co2": em.get("CO2"),
            "em_tvoc": em.get("TVOC"),
            "em_label": em.get("Label"),

            "sm_heart_rate": sm.get("HeartRate"),
            "sm_breath_rate": sm.get("BreathRate"),
            "sm_spo2": sm.get("SPO2"),
            "sm_skin_temperature": sm.get("SkinTemperature"),
            "sm_sleep_phase": sm.get("SleepPhase"),
            "sm_sleep_score": sm.get("SleepScore"),
            "sm_walking_steps": sm.get("WalkingSteps"),
            "sm_stress_index": sm.get("StressIndex"),
            "sm_activity_intensity": sm.get("ActivityIntensity"),
            "sm_caloric_expenditure": sm.get("CaloricExpenditure"),
            "sm_label": sm.get("Label"),

            "ir_caption": ir.get("Caption"),
            "er_button": er.get("Button"),
            "er_shout": er.get("Shout"),
            "er_label": er.get("Label"),
            "counseling_text": cs.get("Counseling"),
            "counseling_memo": cs.get("Memo"),
            "counseling_information": cs.get("Information"),
            "target_estimation": tl.get("Estimation"),
            "target_reason": tl.get("Reason"),
        }

        rows.append(row)

    return rows


def build_timeseries_master_from_manifest_resume(
    split_dir: Path,
    split_name: str,
    manifest_df: pd.DataFrame,
    chunk_size: int = CHUNK_SIZE,
):
    src_root, _ = get_split_roots(split_dir)
    png_map = build_ir_png_index(src_root)

    manifest_path, part_dir, progress_path, error_log_path = get_output_paths(split_name)
    part_dir.mkdir(parents=True, exist_ok=True)

    work_df = manifest_df.copy()
    work_df = work_df[work_df["has_label_json"] == True].reset_index(drop=True)

    progress = load_progress(progress_path)

    completed_sample_keys = set(progress.get("completed_sample_keys", []))
    existing_sample_keys = get_existing_written_sample_keys_from_parts(part_dir)
    completed_sample_keys = completed_sample_keys.union(existing_sample_keys)

    existing_parts = sorted(part_dir.glob("part_*.csv"))
    part_idx = len(existing_parts)

    print("=" * 80)
    print(f"[START] {split_name.upper()} timeseries master 생성 (resume mode)")
    print(f"[INFO] label json 있는 sample 수: {len(work_df)}")
    print(f"[INFO] 이미 완료된 sample 수: {len(completed_sample_keys)}")
    print(f"[INFO] 저장 디렉토리: {part_dir}")
    print(f"[INFO] 시작 part_idx: {part_idx}")

    buffer_rows = []
    fail_count = 0
    done_count = 0
    skip_count = 0
    written_rows_total = int(progress.get("written_rows", 0))

    for idx, row in work_df.iterrows():
        row_dict = row.to_dict()
        s_key = row_dict["sample_key"]
        json_path = Path(row_dict["label_json_path"])

        if s_key in completed_sample_keys:
            skip_count += 1
            cur = idx + 1
            if cur % 50 == 0 or cur == len(work_df):
                print(
                    f"[{split_name.upper()} PROGRESS] "
                    f"sample {cur:,}/{len(work_df):,} | "
                    f"done {done_count:,} | skip {skip_count:,} | "
                    f"buffer_rows {len(buffer_rows):,} | fail {fail_count:,}"
                )
            continue

        try:
            rows = expand_label_json_with_manifest(
                json_path=json_path,
                split_name=split_name,
                manifest_row=row_dict,
                png_map=png_map,
            )
            buffer_rows.extend(rows)

            completed_sample_keys.add(s_key)
            progress["completed_sample_keys"] = sorted(completed_sample_keys)
            progress["last_processed_sample_key"] = s_key
            done_count += 1

        except Exception as e:
            fail_count += 1
            error_msg = (
                f"[ERROR] JSON 처리 실패: {json_path}\n"
                f"        sample_key: {s_key}\n"
                f"        예외 타입: {type(e).__name__}\n"
                f"        예외 repr: {repr(e)}\n"
                f"{traceback.format_exc()}\n"
                + ("-" * 80)
            )
            print(error_msg)
            append_error_log(error_log_path, error_msg)

            failed_files = progress.get("failed_files", [])
            failed_files.append({
                "sample_key": s_key,
                "json_path": str(json_path),
                "error_type": type(e).__name__,
                "error_repr": repr(e),
            })
            progress["failed_files"] = failed_files

        if done_count > 0 and (done_count % chunk_size == 0) and buffer_rows:
            written_now, part_idx = append_rows_to_csv(buffer_rows, part_dir, part_idx)
            written_rows_total += written_now

            progress["written_rows"] = written_rows_total
            progress["part_idx"] = part_idx
            save_progress(progress, progress_path)

            print(
                f"[SAVE CSV] {split_name} | "
                f"part_{part_idx-1:04d} | written_now {written_now:,} | "
                f"written_total {written_rows_total:,}"
            )

            buffer_rows = []

        cur = idx + 1
        if cur % 50 == 0 or cur == len(work_df):
            print(
                f"[{split_name.upper()} PROGRESS] "
                f"sample {cur:,}/{len(work_df):,} | "
                f"done {done_count:,} | skip {skip_count:,} | "
                f"buffer_rows {len(buffer_rows):,} | fail {fail_count:,}"
            )

    if buffer_rows:
        written_now, part_idx = append_rows_to_csv(buffer_rows, part_dir, part_idx)
        written_rows_total += written_now

        progress["written_rows"] = written_rows_total
        progress["part_idx"] = part_idx
        save_progress(progress, progress_path)

        print(
            f"[FINAL SAVE] {split_name} | "
            f"part_{part_idx-1:04d} | written_now {written_now:,} | "
            f"written_total {written_rows_total:,}"
        )

    print(f"[DONE] {split_name.upper()} csv 분할 저장 완료")
    print(f"[OUTPUT DIR] {part_dir}")
    print("=" * 80)


# =========================================================
# 8. Save
# =========================================================
def save_csv(df: pd.DataFrame, path: Path, name: str, overwrite: bool = True):
    if (not overwrite) and path.exists():
        print(f"[SKIP SAVE] 이미 존재: {path}")
        return

    print(f"[SAVE] {name} -> {path}")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[SAVE DONE] {name}")


# =========================================================
# 9. Main Split Process
# =========================================================
def process_one_split(split_dir: Path, split_name: str):
    manifest_path, part_dir, progress_path, error_log_path = get_output_paths(split_name)

    if manifest_path.exists():
        print(f"[INFO] 기존 manifest 재사용: {manifest_path}")
        try:
            manifest_df = pd.read_csv(manifest_path, encoding="utf-8-sig")
        except Exception:
            manifest_df = pd.read_csv(manifest_path, encoding="utf-8")
    else:
        manifest_df = build_sample_manifest(split_dir, split_name)
        save_csv(manifest_df, manifest_path, f"{split_name}_sample_manifest.csv", overwrite=True)

    build_timeseries_master_from_manifest_resume(
        split_dir=split_dir,
        split_name=split_name,
        manifest_df=manifest_df,
        chunk_size=CHUNK_SIZE,
    )

    return manifest_df


# =========================================================
# 10. Main
# =========================================================
def main():
    print("[INFO] PROJECT_ROOT:", PROJECT_ROOT)
    print("[INFO] BASE_DIR     :", BASE_DIR)
    print("[INFO] SAVE_DIR     :", SAVE_DIR)
    print("[INFO] CHUNK_SIZE   :", CHUNK_SIZE)

    if not BASE_DIR.exists():
        raise FileNotFoundError(f"데이터 폴더를 찾을 수 없습니다: {BASE_DIR}")

    train_manifest_df = process_one_split(TRAIN_DIR, "train")
    valid_manifest_df = process_one_split(VALID_DIR, "valid")

    print("\n[SUMMARY]")
    print("train_manifest_df:", train_manifest_df.shape)
    print("valid_manifest_df:", valid_manifest_df.shape)

    if len(train_manifest_df) > 0:
        print("\n[TRAIN MANIFEST HEAD]")
        print(train_manifest_df.head(3).to_string())

    print(f"\n[FINISH] 완료")
    print(f"[OUTPUT] {SAVE_DIR}")


if __name__ == "__main__":
    main()
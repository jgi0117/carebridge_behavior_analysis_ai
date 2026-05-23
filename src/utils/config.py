from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "configs" / "paths.yaml").exists():
            return parent
    raise FileNotFoundError("Could not find configs/paths.yaml")


PROJECT_ROOT = project_root()
CONFIG_DIR = PROJECT_ROOT / "configs"


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml(filename: str) -> dict[str, Any]:
    path = CONFIG_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(*filenames: str) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for filename in filenames:
        config = deep_merge(config, load_yaml(filename))
    return config


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def configured_path(key: str, config: dict[str, Any] | None = None) -> Path:
    cfg = config or load_config("paths.yaml")
    try:
        return resolve_path(cfg["paths"][key])
    except KeyError as exc:
        raise KeyError(f"paths.{key} is missing from configs/paths.yaml") from exc


def model_artifact_path(config: dict[str, Any] | None = None) -> Path:
    cfg = config or load_config("paths.yaml")
    primary = configured_path("model_artifact", cfg)
    if primary.exists():
        return primary

    fallback = cfg.get("paths", {}).get("model_artifact_fallback")
    if fallback:
        fallback_path = resolve_path(fallback)
        if fallback_path.exists():
            return fallback_path

    return primary

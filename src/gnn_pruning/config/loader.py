"""YAML loading and merge utilities for experiment configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

try:
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    yaml = None

from .schema import ExperimentConfig
from ..utils import simple_yaml

CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load a YAML file and validate the top-level type."""
    raw_text = Path(path).expanduser().read_text(encoding="utf-8")
    payload = _safe_load(raw_text) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return payload


def deep_merge(*parts: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge dictionaries from left to right."""
    merged: Dict[str, Any] = {}
    for part in parts:
        merged = _merge_pair(merged, part)
    return merged


def resolve_config(config_path: str | Path) -> ExperimentConfig:
    """Resolve layered config into a validated typed config."""
    user_config = load_yaml(config_path)

    base_ref = user_config.get("base", "base/default")
    dataset_ref = user_config.get("dataset")
    model_ref = user_config.get("model")
    preset_ref = user_config.get("preset")

    base_cfg = _load_ref(base_ref, folder="base") if base_ref else {}
    dataset_cfg = _load_ref(dataset_ref, folder="datasets") if dataset_ref else {}
    model_cfg = _load_ref(model_ref, folder="models") if model_ref else {}
    preset_cfg = _load_ref(preset_ref, folder="presets") if preset_ref else {}

    user_overrides = dict(user_config)
    for key in ("base", "dataset", "model", "preset"):
        user_overrides.pop(key, None)

    resolved = deep_merge(base_cfg, dataset_cfg, model_cfg, preset_cfg, user_overrides)
    return ExperimentConfig.from_dict(resolved)


def dump_yaml(payload: Dict[str, Any], path: str | Path) -> None:
    """Write a dictionary to YAML."""
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    text = _safe_dump(payload)
    target.write_text(text, encoding="utf-8")


def _merge_pair(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(left)
    for key, value in right.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_pair(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_ref(ref: str, folder: str) -> Dict[str, Any]:
    path = _resolve_ref_path(ref, folder)
    return load_yaml(path)


def _resolve_ref_path(ref: str, folder: str) -> Path:
    ref_path = Path(ref)
    if ref_path.suffix in {".yml", ".yaml"}:
        return ref_path

    if "/" in ref:
        candidate = CONFIG_DIR / f"{ref}.yaml"
    else:
        candidate = CONFIG_DIR / folder / f"{ref}.yaml"
    if not candidate.exists():
        raise FileNotFoundError(f"Config reference not found: {ref} ({candidate})")
    return candidate


def _safe_load(raw_text: str) -> Dict[str, Any]:
    if yaml is not None:
        return yaml.safe_load(raw_text)
    return simple_yaml.safe_load(raw_text)


def _safe_dump(payload: Dict[str, Any]) -> str:
    if yaml is not None:
        return yaml.safe_dump(payload, sort_keys=False)
    return simple_yaml.safe_dump(payload, sort_keys=False)

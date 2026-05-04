"""YAML loading and merge utilities for experiment configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Union

try:
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    yaml = None

from .schema import ExperimentConfig
from ..utils import simple_yaml

CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
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


def resolve_config(config_path: Union[str, Path]) -> ExperimentConfig:
    """Resolve layered config into a validated typed config."""
    user_config = load_yaml(config_path)

    base_ref = user_config.get("base", "base/default")
    dataset_ref = user_config.get("dataset")
    model_ref = user_config.get("model")
    preset_ref = user_config.get("preset")

    base_cfg = _resolve_reference_config(base_ref, folder="base", field_name="base")
    dataset_cfg = _resolve_component_config(dataset_ref, folder="datasets", field_name="dataset", section_key="data")
    model_cfg = _resolve_component_config(model_ref, folder="models", field_name="model", section_key="model")
    preset_cfg = _resolve_reference_config(preset_ref, folder="presets", field_name="preset")

    dataset_name = _resolve_dataset_name(dataset_cfg=dataset_cfg, dataset_ref=dataset_ref)
    preset_cfg = _apply_dataset_overrides(preset_cfg, dataset_name)

    user_overrides = dict(user_config)
    for key in ("base", "dataset", "model", "preset"):
        user_overrides.pop(key, None)

    resolved = deep_merge(base_cfg, dataset_cfg, model_cfg, preset_cfg, user_overrides)
    return ExperimentConfig.from_dict(resolved)


def dump_yaml(payload: Dict[str, Any], path: Union[str, Path]) -> None:
    """Write a dictionary to YAML."""
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    text = _safe_dump(payload)
    target.write_text(text, encoding="utf-8")


def _resolve_reference_config(ref: Any, folder: str, field_name: str) -> Dict[str, Any]:
    if ref is None:
        return {}
    if isinstance(ref, str):
        return _load_ref(ref, folder)
    raise ValueError(
        f"`{field_name}` must be a string reference or null. Got {type(ref).__name__}."
    )


def _resolve_component_config(
    value: Any,
    folder: str,
    field_name: str,
    section_key: str,
) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, str):
        return _load_ref(value, folder)
    if isinstance(value, Mapping):
        if section_key in value and isinstance(value[section_key], Mapping):
            return dict(value)
        return {section_key: dict(value)}
    raise ValueError(
        f"`{field_name}` must be either a string reference or a mapping. "
        f"Got {type(value).__name__}."
    )


def _resolve_dataset_name(dataset_cfg: Dict[str, Any], dataset_ref: Any) -> str:
    data_cfg = dataset_cfg.get("data", {}) if isinstance(dataset_cfg, dict) else {}
    if isinstance(data_cfg, dict) and "name" in data_cfg:
        return str(data_cfg["name"]).lower()
    if isinstance(dataset_ref, str):
        return dataset_ref.lower()
    if isinstance(dataset_ref, Mapping) and "name" in dataset_ref:
        return str(dataset_ref["name"]).lower()
    return ""


def _apply_dataset_overrides(preset_cfg: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
    if not isinstance(preset_cfg, dict):
        return preset_cfg

    overrides = preset_cfg.get("dataset_overrides")
    if not isinstance(overrides, dict):
        return preset_cfg

    selected = overrides.get(dataset_name, {})
    base_cfg = dict(preset_cfg)
    base_cfg.pop("dataset_overrides", None)

    if isinstance(selected, dict):
        return deep_merge(base_cfg, selected)
    return base_cfg


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

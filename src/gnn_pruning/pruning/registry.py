"""Pruner registry and helpers."""

from __future__ import annotations

from typing import Dict, List, Type

from .base import BasePruner

PRUNER_REGISTRY: Dict[str, Type[BasePruner]] = {}


def register_pruner(pruner_cls: Type[BasePruner]) -> Type[BasePruner]:
    """Register a pruner class by its declared name."""
    key = pruner_cls.name.strip().lower()
    if not key:
        raise ValueError("Pruner class must define a non-empty `name`.")
    if key in PRUNER_REGISTRY:
        raise ValueError(f"Pruner '{key}' already registered.")
    PRUNER_REGISTRY[key] = pruner_cls
    return pruner_cls


def get_pruner(name: str) -> Type[BasePruner]:
    """Fetch pruner class by name."""
    key = name.strip().lower()
    if key not in PRUNER_REGISTRY:
        raise KeyError(f"Unknown pruner '{name}'.")
    return PRUNER_REGISTRY[key]


def list_pruners() -> List[dict]:
    """List registered pruner metadata records."""
    rows: List[dict] = []
    for _, pruner_cls in sorted(PRUNER_REGISTRY.items()):
        instance = pruner_cls()
        rows.append(instance.metadata())
    return rows

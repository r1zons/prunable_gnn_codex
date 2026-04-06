"""Dataset registry and convenience wrappers."""

from __future__ import annotations

from typing import Dict

from .factory import get_dataset_loader, get_supported_datasets, load_dataset

DATASET_REGISTRY: Dict[str, str] = get_supported_datasets()

__all__ = ["DATASET_REGISTRY", "get_dataset_loader", "get_supported_datasets", "load_dataset"]

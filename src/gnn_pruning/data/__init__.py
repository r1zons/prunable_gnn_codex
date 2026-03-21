"""Dataset loading and split generation."""

from .factory import get_dataset_loader, get_supported_datasets, load_dataset
from .registry import DATASET_REGISTRY
from .splits import (
    SplitIndices,
    generate_exact_ratio_split,
    load_split_indices,
    save_split_indices,
    to_index_tensors,
)

__all__ = [
    "DATASET_REGISTRY",
    "SplitIndices",
    "generate_exact_ratio_split",
    "get_dataset_loader",
    "get_supported_datasets",
    "load_dataset",
    "load_split_indices",
    "save_split_indices",
    "to_index_tensors",
]

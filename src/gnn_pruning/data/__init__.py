"""Dataset loading and split generation."""

from .factory import get_dataset_loader, get_supported_datasets, load_dataset
from .registry import DATASET_REGISTRY
from .splits import SplitIndices, generate_exact_ratio_split, save_split_indices

__all__ = [
    "DATASET_REGISTRY",
    "SplitIndices",
    "generate_exact_ratio_split",
    "get_dataset_loader",
    "get_supported_datasets",
    "load_dataset",
    "save_split_indices",
]

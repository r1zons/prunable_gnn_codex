"""Split generation and artifact helpers for node classification datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from random import Random
from typing import Dict, List, Union

import torch

from gnn_pruning.config import dump_yaml, load_yaml


@dataclass(frozen=True)
class SplitIndices:
    """Integer node index splits for train/val/test."""

    train: List[int]
    val: List[int]
    test: List[int]

    def to_dict(self) -> Dict[str, List[int]]:
        """Serialize splits into plain dictionaries."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, List[int]]) -> "SplitIndices":
        """Create split object from dictionary payload."""
        return cls(train=list(payload["train"]), val=list(payload["val"]), test=list(payload["test"]))


def generate_exact_ratio_split(
    num_nodes: int,
    seed: int = 42,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
) -> SplitIndices:
    """Generate one random disjoint split that covers every node exactly once."""
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive.")

    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}.")

    indices = list(range(num_nodes))
    Random(seed).shuffle(indices)

    train_count = int(num_nodes * train_ratio)
    val_count = int(num_nodes * val_ratio)
    test_count = num_nodes - train_count - val_count

    train_idx = indices[:train_count]
    val_idx = indices[train_count : train_count + val_count]
    test_idx = indices[train_count + val_count : train_count + val_count + test_count]
    return SplitIndices(train=train_idx, val=val_idx, test=test_idx)


def save_split_indices(split: SplitIndices, output_dir: Union[str, Path]) -> Path:
    """Persist split indices as a YAML artifact under output directory."""
    target = Path(output_dir).expanduser() / "splits.yaml"
    dump_yaml(split.to_dict(), target)
    return target


def load_split_indices(path: Union[str, Path]) -> SplitIndices:
    """Load split artifact from disk."""
    payload = load_yaml(path)
    return SplitIndices.from_dict(payload)


def to_index_tensors(split: SplitIndices, device: Union[str, torch.device]) -> Dict[str, torch.Tensor]:
    """Convert split index lists into torch index tensors."""
    return {
        "train": torch.tensor(split.train, dtype=torch.long, device=device),
        "val": torch.tensor(split.val, dtype=torch.long, device=device),
        "test": torch.tensor(split.test, dtype=torch.long, device=device),
    }

"""Checkpoint utilities for dense model training."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

import torch


def save_checkpoint(payload: Dict[str, Any], path: Union[str, Path]) -> Path:
    """Save checkpoint payload to disk."""
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)
    return target


def load_checkpoint(path: Union[str, Path], map_location: Union[str, torch.device]) -> Dict[str, Any]:
    """Load checkpoint payload from disk."""
    return torch.load(Path(path).expanduser(), map_location=map_location)

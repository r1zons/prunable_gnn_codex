"""Reproducibility helpers."""

from __future__ import annotations

import random


def set_seed(seed: int = 42) -> None:
    """Set Python random seed; extend for NumPy/PyTorch later."""
    random.seed(seed)

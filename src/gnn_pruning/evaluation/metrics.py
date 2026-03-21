"""Evaluation metrics for node classification."""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import f1_score


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute node-classification metrics."""
    if y_true.size == 0:
        raise ValueError("Cannot compute metrics on empty target array.")

    accuracy = float((y_true == y_pred).mean())
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }

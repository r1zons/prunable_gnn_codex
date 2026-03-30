"""Base interfaces for node-classification models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import torch
from torch import Tensor, nn


class BaseNodeClassifier(nn.Module, ABC):
    """Common base interface for node classifiers."""

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = float(dropout)

    @abstractmethod
    def forward(self, data: Any) -> Tensor:
        """Compute logits for all nodes in the graph."""

    def predict(self, data: Any) -> Tensor:
        """Run model inference and return class predictions."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(data)
            return logits.argmax(dim=-1)

    @abstractmethod
    def export_architecture_config(self) -> Dict[str, Any]:
        """Export architecture parameters for serialization/rebuild."""

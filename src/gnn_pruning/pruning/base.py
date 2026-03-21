"""Base interfaces for pruning stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class BasePruner:
    """Abstract-like base class defining pruning phases."""

    def score(self, model: Any) -> Any:
        """Compute importance scores (placeholder)."""
        raise NotImplementedError

    def plan(self, scores: Any) -> Any:
        """Create pruning plan from scores (placeholder)."""
        raise NotImplementedError

    def apply(self, model: Any, plan: Any) -> Any:
        """Apply structural changes to model (placeholder)."""
        raise NotImplementedError

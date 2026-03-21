"""Pruning abstractions and shared dataclasses."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, MutableMapping, Optional


@dataclass
class PruningContext:
    """Runtime context available to pruners."""

    config: Mapping[str, Any]
    data: Any
    device: str
    seed: int = 42


@dataclass
class PruningPlan:
    """Serializable pruning decision output."""

    name: str
    category: str
    target_sparsity: float
    details: MutableMapping[str, Any] = field(default_factory=dict)
    achieved_sparsity: Optional[float] = None


class BasePruner(ABC):
    """Base interface for all pruning methods.

    Concrete pruners must separate scoring from application.
    """

    name: str = "base"
    category: str = "unknown"
    supports_unstructured: bool = False
    supports_structured: bool = False

    @abstractmethod
    def score(self, model: Any, context: PruningContext) -> Any:
        """Compute pruning scores without mutating model weights."""

    @abstractmethod
    def apply(self, model: Any, scores: Any, context: PruningContext) -> PruningPlan:
        """Apply pruning decisions and return a structural pruning plan."""

    def metadata(self) -> Dict[str, Any]:
        """Return standard metadata fields for this pruner."""
        return {
            "name": self.name,
            "category": self.category,
            "supports_unstructured": self.supports_unstructured,
            "supports_structured": self.supports_structured,
        }

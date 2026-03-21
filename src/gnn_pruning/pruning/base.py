"""Pruning abstractions and shared dataclasses."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence


@dataclass
class PruningContext:
    """Runtime context available to pruners."""

    seed: int = 42
    mode: str = "unstructured"
    structure_target: float = 0.0
    data: Any = None
    device: Optional[str] = None
    config: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_input(cls, context: Any) -> "PruningContext":
        """Normalize context from dataclass or dict."""
        if isinstance(context, cls):
            return context
        if isinstance(context, Mapping):
            return cls(
                seed=int(context.get("seed", 42)),
                mode=str(context.get("mode", "unstructured")),
                structure_target=float(context.get("structure_target", 0.0)),
                data=context.get("data"),
                device=context.get("device"),
                config=context.get("config", {}),
            )
        raise TypeError("context must be PruningContext or mapping/dict")


@dataclass
class PruningPlan:
    """Serializable pruning decision output."""

    name: str
    category: str
    requested_sparsity: float = 0.0
    mode: str = "unstructured"
    target_sparsity: Optional[float] = None
    layer_index: Optional[int] = None
    target_units: Sequence[int] = field(default_factory=list)
    details: MutableMapping[str, Any] = field(default_factory=dict)
    score_payload: Any = None
    achieved_sparsity: Optional[float] = None
    pruning_time_sec: Optional[float] = None

    def __post_init__(self) -> None:
        """Backwards-compatible sparsity naming."""
        if self.target_sparsity is not None:
            self.requested_sparsity = float(self.target_sparsity)
        self.target_sparsity = float(self.requested_sparsity)


class BasePruner(ABC):
    """Base interface for all pruning methods."""

    name: str = "base"
    category: str = "unknown"
    supports_unstructured: bool = False
    supports_structured: bool = False

    @abstractmethod
    def score(self, model: Any, context: Any, **kwargs: Any) -> PruningPlan:
        """Compute pruning scores without mutating model weights."""

    @abstractmethod
    def apply(self, model: Any, pruning_plan: PruningPlan, context: Any, **kwargs: Any) -> Any:
        """Apply pruning using a standardized pruning plan and return pruned model."""

    def metadata(self) -> Dict[str, Any]:
        """Return standard metadata fields for this pruner."""
        return {
            "name": self.name,
            "category": self.category,
            "supports_unstructured": self.supports_unstructured,
            "supports_structured": self.supports_structured,
        }

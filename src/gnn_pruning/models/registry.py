"""Model registry and model factory helpers."""

from __future__ import annotations

from typing import Any, Callable, Dict

from .gcn import GCNNodeClassifier
from .graphsage import GraphSAGENodeClassifier

ModelBuilder = Callable[..., Any]

MODEL_REGISTRY: Dict[str, ModelBuilder] = {
    "gcn": GCNNodeClassifier,
    "graphsage": GraphSAGENodeClassifier,
    # Extension points for future work:
    # "gat": GATNodeClassifier,
    # "gin": GINNodeClassifier,
}


def build_model(model_name: str, **kwargs: Any) -> Any:
    """Build model by registry key."""
    key = model_name.strip().lower()
    if key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{model_name}'. Available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[key](**kwargs)

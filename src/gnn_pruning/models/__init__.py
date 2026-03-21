"""Model abstractions and registry."""

from .base import BaseNodeClassifier
from .gcn import GCNNodeClassifier
from .graphsage import GraphSAGENodeClassifier
from .registry import MODEL_REGISTRY, build_model

__all__ = [
    "BaseNodeClassifier",
    "GCNNodeClassifier",
    "GraphSAGENodeClassifier",
    "MODEL_REGISTRY",
    "build_model",
]

"""State feature extraction and bucketing for tabular RL pruning."""

from __future__ import annotations

from typing import Any, Dict, Iterable


def bucketize(value: float, boundaries: Iterable[float]) -> int:
    """Return integer bucket index for value against ascending boundaries."""
    idx = 0
    for boundary in boundaries:
        if value <= float(boundary):
            return idx
        idx += 1
    return idx


def build_bucketed_state(
    *,
    data: Any,
    current_sparsity: float,
    target_sparsity: float,
    accuracy_drop: float,
    remaining_channels: int,
    total_channels: int,
    buckets: Dict[str, list[float]],
) -> Dict[str, int]:
    """Build discretized graph-aware state."""
    num_nodes = int(getattr(data, "num_nodes", 0) or 0)
    edge_index = getattr(data, "edge_index", None)
    num_edges = int(edge_index.size(1)) if edge_index is not None else 0
    num_features = int(getattr(data, "num_features", 0) or 0)
    y = getattr(data, "y", None)
    num_classes = int(y.max().item()) + 1 if y is not None and y.numel() > 0 else 0

    avg_degree = float((2.0 * num_edges) / max(1, num_nodes))
    density = float(num_edges / max(1, num_nodes * max(1, num_nodes - 1)))
    remaining_ratio = float(remaining_channels / max(1, total_channels))
    target_gap = max(0.0, float(target_sparsity) - float(current_sparsity))

    return {
        "num_nodes_bucket": bucketize(float(num_nodes), buckets["num_nodes"]),
        "num_edges_bucket": bucketize(float(num_edges), buckets["num_edges"]),
        "avg_degree_bucket": bucketize(avg_degree, buckets["avg_degree"]),
        "density_bucket": bucketize(density, buckets["density"]),
        "num_features_bucket": bucketize(float(num_features), buckets["num_features"]),
        "num_classes_bucket": bucketize(float(num_classes), buckets["num_classes"]),
        "current_sparsity_bucket": bucketize(float(current_sparsity), buckets["current_sparsity"]),
        "target_gap_bucket": bucketize(float(target_gap), buckets["target_gap"]),
        "accuracy_drop_bucket": bucketize(float(max(0.0, accuracy_drop)), buckets["accuracy_drop"]),
        "remaining_channels_bucket": bucketize(remaining_ratio, buckets["remaining_channels"]),
    }


def state_key(state: Dict[str, int]) -> str:
    """Serialize state dict to stable string key for Q-table."""
    keys = sorted(state.keys())
    return "|".join(f"{key}:{int(state[key])}" for key in keys)

"""Structural surgery utilities for hidden-channel pruning."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Sequence

import torch
from torch import nn
from torch_geometric.nn import GCNConv, SAGEConv

from gnn_pruning.models import GCNNodeClassifier, GraphSAGENodeClassifier


@dataclass
class FeasibilityResult:
    """Static feasibility check result for a structural prune action."""

    valid: bool
    reason: str
    hidden_widths_before: list[int]
    expected_hidden_widths_after: list[int]
    selected_layer_width: int
    keep_count: int
    affected_layer_shapes: Dict[str, Any]


def can_apply_structural_prune(
    model: Any,
    layer_index: int,
    keep_indices: Sequence[int],
    min_channels_per_layer: int = 1,
    mode: str = "local",
) -> FeasibilityResult:
    """Check whether structural pruning can be safely applied without modifying model."""
    hidden_widths_before = _hidden_widths(model)
    selected_layer_width = hidden_widths_before[layer_index] if 0 <= layer_index < len(hidden_widths_before) else 0
    normalized_keep = sorted(set(int(value) for value in keep_indices))
    keep_count = int(len(normalized_keep))
    affected = _affected_layer_shapes(model, layer_index=layer_index, keep_count=keep_count)

    if not hasattr(model, "convs"):
        return FeasibilityResult(
            valid=False,
            reason="missing_convs",
            hidden_widths_before=hidden_widths_before,
            expected_hidden_widths_after=list(hidden_widths_before),
            selected_layer_width=selected_layer_width,
            keep_count=keep_count,
            affected_layer_shapes=affected,
        )

    convs = model.convs
    if layer_index < 0 or layer_index >= len(convs) - 1:
        return FeasibilityResult(
            valid=False,
            reason="invalid_layer_index",
            hidden_widths_before=hidden_widths_before,
            expected_hidden_widths_after=list(hidden_widths_before),
            selected_layer_width=selected_layer_width,
            keep_count=keep_count,
            affected_layer_shapes=affected,
        )

    if mode not in {"local", "cascade"}:
        return FeasibilityResult(
            valid=False,
            reason="unsupported_mode",
            hidden_widths_before=hidden_widths_before,
            expected_hidden_widths_after=list(hidden_widths_before),
            selected_layer_width=selected_layer_width,
            keep_count=keep_count,
            affected_layer_shapes=affected,
        )

    if keep_count <= 0:
        return FeasibilityResult(
            valid=False,
            reason="empty_keep_indices",
            hidden_widths_before=hidden_widths_before,
            expected_hidden_widths_after=list(hidden_widths_before),
            selected_layer_width=selected_layer_width,
            keep_count=keep_count,
            affected_layer_shapes=affected,
        )
    if keep_count < int(min_channels_per_layer):
        return FeasibilityResult(
            valid=False,
            reason="min_channels_violation",
            hidden_widths_before=hidden_widths_before,
            expected_hidden_widths_after=list(hidden_widths_before),
            selected_layer_width=selected_layer_width,
            keep_count=keep_count,
            affected_layer_shapes=affected,
        )

    selected = convs[layer_index]
    downstream = convs[layer_index + 1]
    selected_out = _conv_out_channels(selected)
    next_in = _conv_in_channels(downstream)
    if selected_out != next_in:
        return FeasibilityResult(
            valid=False,
            reason="adjacent_shape_mismatch",
            hidden_widths_before=hidden_widths_before,
            expected_hidden_widths_after=list(hidden_widths_before),
            selected_layer_width=selected_layer_width,
            keep_count=keep_count,
            affected_layer_shapes=affected,
        )

    if min(normalized_keep) < 0 or max(normalized_keep) >= selected_out:
        return FeasibilityResult(
            valid=False,
            reason="keep_indices_out_of_bounds",
            hidden_widths_before=hidden_widths_before,
            expected_hidden_widths_after=list(hidden_widths_before),
            selected_layer_width=selected_layer_width,
            keep_count=keep_count,
            affected_layer_shapes=affected,
        )

    expected_hidden = list(hidden_widths_before)
    if 0 <= layer_index < len(expected_hidden):
        expected_hidden[layer_index] = int(keep_count)
    if mode == "cascade":
        for idx in range(layer_index + 1, len(expected_hidden)):
            expected_hidden[idx] = int(keep_count)

    return FeasibilityResult(
        valid=True,
        reason="ok",
        hidden_widths_before=hidden_widths_before,
        expected_hidden_widths_after=expected_hidden,
        selected_layer_width=selected_out,
        keep_count=keep_count,
        affected_layer_shapes=affected,
    )


def structurally_prune_hidden_channels(model: Any, layer_index: int, keep_indices: Sequence[int]) -> Any:
    """Return a structurally compressed model with pruned hidden channels removed.

    This operation physically rebuilds affected layers and copies surviving weights.
    """
    if not keep_indices:
        raise ValueError("keep_indices must not be empty.")

    kept = torch.tensor(sorted(set(int(i) for i in keep_indices)), dtype=torch.long)

    pruned_model = copy.deepcopy(model)
    if not hasattr(pruned_model, "convs"):
        raise TypeError("Model does not expose `convs` layers required for structural surgery.")

    convs = pruned_model.convs
    if layer_index < 0 or layer_index >= len(convs) - 1:
        raise ValueError("layer_index must target a hidden layer with a downstream layer.")

    if isinstance(pruned_model, GCNNodeClassifier):
        _cascade_hidden_prune_gcn(convs=convs, start_layer=layer_index, keep=kept)
    elif isinstance(pruned_model, GraphSAGENodeClassifier):
        _cascade_hidden_prune_sage(convs=convs, start_layer=layer_index, keep=kept)
    else:
        raise TypeError("Unsupported model type for structural hidden-channel pruning.")

    _sync_hidden_channel_metadata(pruned_model)
    _validate_all_internal_shapes(pruned_model)
    return pruned_model


def structurally_prune_hidden_channels_local(model: Any, layer_index: int, keep_indices: Sequence[int]) -> Any:
    """Structurally prune one hidden layer and immediate downstream input only."""
    if not keep_indices:
        raise ValueError("keep_indices must not be empty.")

    kept = torch.tensor(sorted(set(int(i) for i in keep_indices)), dtype=torch.long)

    pruned_model = copy.deepcopy(model)
    if not hasattr(pruned_model, "convs"):
        raise TypeError("Model does not expose `convs` layers required for structural surgery.")

    convs = pruned_model.convs
    if layer_index < 0 or layer_index >= len(convs) - 1:
        raise ValueError("layer_index must target a hidden layer with a downstream layer.")

    selected = convs[layer_index]
    downstream = convs[layer_index + 1]
    if _conv_out_channels(selected) != _conv_in_channels(downstream):
        raise ValueError("Selected layer output channels must match downstream input channels.")
    if kept.numel() <= 0:
        raise ValueError("keep_indices must not be empty.")
    if int(kept.min().item()) < 0 or int(kept.max().item()) >= _conv_out_channels(selected):
        raise ValueError("keep_indices out of bounds for selected layer output width.")

    if isinstance(pruned_model, GCNNodeClassifier):
        convs[layer_index] = _rebuild_gcn_out(selected, kept)
        convs[layer_index + 1] = _rebuild_gcn_in(downstream, kept)
    elif isinstance(pruned_model, GraphSAGENodeClassifier):
        convs[layer_index] = _rebuild_sage_out(selected, kept)
        convs[layer_index + 1] = _rebuild_sage_in(downstream, kept)
    else:
        raise TypeError("Unsupported model type for structural hidden-channel pruning.")

    _sync_hidden_channel_metadata(pruned_model)
    _validate_all_internal_shapes(pruned_model)
    return pruned_model


def _sync_hidden_channel_metadata(model: Any) -> None:
    widths = _hidden_widths(model)
    if not widths:
        return
    if hasattr(model, "hidden_channels"):
        model.hidden_channels = int(widths[0])
    if hasattr(model, "hidden_channel_dims"):
        model.hidden_channel_dims = [int(width) for width in widths]


def _cascade_hidden_prune_gcn(convs: Any, start_layer: int, keep: torch.Tensor) -> None:
    for idx in range(start_layer, len(convs) - 1):
        convs[idx] = _rebuild_gcn_out(convs[idx], keep)
        convs[idx + 1] = _rebuild_gcn_in(convs[idx + 1], keep)


def _cascade_hidden_prune_sage(convs: Any, start_layer: int, keep: torch.Tensor) -> None:
    for idx in range(start_layer, len(convs) - 1):
        convs[idx] = _rebuild_sage_out(convs[idx], keep)
        convs[idx + 1] = _rebuild_sage_in(convs[idx + 1], keep)


def validate_structural_compression(original_model: Any, compressed_model: Any, data: Any) -> None:
    """Validate core structural-compression guarantees."""
    original_params = _parameter_count(original_model)
    compressed_params = _parameter_count(compressed_model)
    if compressed_params >= original_params:
        raise ValueError("Compressed model must have fewer parameters than original model.")

    with torch.no_grad():
        out_original = original_model(data)
        out_compressed = compressed_model(data)

    if out_original.shape[-1] != out_compressed.shape[-1]:
        raise ValueError("Classifier output dimension must be preserved after surgery.")


def _rebuild_gcn_out(conv: GCNConv, keep: torch.Tensor) -> GCNConv:
    new_conv = GCNConv(
        in_channels=conv.in_channels,
        out_channels=int(keep.numel()),
        improved=conv.improved,
        cached=conv.cached,
        add_self_loops=conv.add_self_loops,
        normalize=conv.normalize,
        bias=conv.bias is not None,
    )
    new_conv.lin = _copy_linear_rows(conv.lin, keep)
    if conv.bias is not None:
        with torch.no_grad():
            new_conv.bias.copy_(conv.bias[keep])
    return new_conv


def _rebuild_gcn_in(conv: GCNConv, keep: torch.Tensor) -> GCNConv:
    new_conv = GCNConv(
        in_channels=int(keep.numel()),
        out_channels=conv.out_channels,
        improved=conv.improved,
        cached=conv.cached,
        add_self_loops=conv.add_self_loops,
        normalize=conv.normalize,
        bias=conv.bias is not None,
    )
    new_conv.lin = _copy_linear_cols(conv.lin, keep)
    if conv.bias is not None:
        with torch.no_grad():
            new_conv.bias.copy_(conv.bias)
    return new_conv


def _rebuild_sage_out(conv: SAGEConv, keep: torch.Tensor) -> SAGEConv:
    in_channels = conv.in_channels
    if isinstance(in_channels, tuple):
        in_channels = in_channels[0]

    new_conv = SAGEConv(
        in_channels=int(in_channels),
        out_channels=int(keep.numel()),
        aggr=conv.aggr,
        normalize=conv.normalize,
        root_weight=conv.root_weight,
        project=conv.project,
        bias=conv.lin_l.bias is not None,
    )
    new_conv.lin_l = _copy_linear_rows(conv.lin_l, keep)
    if conv.root_weight:
        new_conv.lin_r = _copy_linear_rows(conv.lin_r, keep)
    return new_conv


def _rebuild_sage_in(conv: SAGEConv, keep: torch.Tensor) -> SAGEConv:
    new_conv = SAGEConv(
        in_channels=int(keep.numel()),
        out_channels=conv.out_channels,
        aggr=conv.aggr,
        normalize=conv.normalize,
        root_weight=conv.root_weight,
        project=conv.project,
        bias=conv.lin_l.bias is not None,
    )
    new_conv.lin_l = _copy_linear_cols(conv.lin_l, keep)
    if conv.root_weight:
        new_conv.lin_r = _copy_linear_cols(conv.lin_r, keep)
    return new_conv


def _copy_linear_rows(linear: nn.Linear, keep: torch.Tensor) -> nn.Linear:
    new_linear = nn.Linear(int(linear.weight.size(1)), int(keep.numel()), bias=linear.bias is not None)
    with torch.no_grad():
        new_linear.weight.copy_(linear.weight[keep, :])
        if linear.bias is not None:
            new_linear.bias.copy_(linear.bias[keep])
    return new_linear


def _copy_linear_cols(linear: nn.Linear, keep: torch.Tensor) -> nn.Linear:
    new_linear = nn.Linear(int(keep.numel()), int(linear.weight.size(0)), bias=linear.bias is not None)
    with torch.no_grad():
        new_linear.weight.copy_(linear.weight[:, keep])
        if linear.bias is not None:
            new_linear.bias.copy_(linear.bias)
    return new_linear


def _validate_internal_shapes(model: Any, layer_index: int) -> None:
    conv = model.convs[layer_index]
    next_conv = model.convs[layer_index + 1]

    out_channels = _conv_out_channels(conv)
    in_channels = _conv_in_channels(next_conv)
    if out_channels != in_channels:
        raise ValueError("Surgery produced incompatible adjacent layer shapes.")


def _validate_all_internal_shapes(model: Any) -> None:
    if not hasattr(model, "convs"):
        return
    for idx in range(len(model.convs) - 1):
        _validate_internal_shapes(model, idx)


def _conv_out_channels(conv: Any) -> int:
    if isinstance(conv, GCNConv):
        return int(conv.out_channels)
    if isinstance(conv, SAGEConv):
        return int(conv.out_channels)
    raise TypeError("Unsupported conv type.")


def _conv_in_channels(conv: Any) -> int:
    in_channels = conv.in_channels
    if isinstance(in_channels, tuple):
        return int(in_channels[0])
    return int(in_channels)


def _parameter_count(model: Any) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _hidden_widths(model: Any) -> list[int]:
    if not hasattr(model, "convs"):
        return []
    widths: list[int] = []
    for conv in model.convs[:-1]:
        widths.append(_conv_out_channels(conv))
    return widths


def _affected_layer_shapes(model: Any, layer_index: int, keep_count: int) -> Dict[str, Any]:
    if not hasattr(model, "convs"):
        return {}
    convs = model.convs
    if layer_index < 0 or layer_index >= len(convs) - 1:
        return {}
    selected = convs[layer_index]
    downstream = convs[layer_index + 1]
    return {
        "selected_layer_index": int(layer_index),
        "selected_out_before": int(_conv_out_channels(selected)),
        "selected_out_after": int(keep_count),
        "downstream_layer_index": int(layer_index + 1),
        "downstream_in_before": int(_conv_in_channels(downstream)),
        "downstream_in_after": int(keep_count),
    }

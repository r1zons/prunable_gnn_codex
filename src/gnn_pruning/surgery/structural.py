"""Structural surgery utilities for hidden-channel pruning."""

from __future__ import annotations

import copy
from typing import Any, Iterable, Sequence

import torch
from torch import nn
from torch_geometric.nn import GCNConv, SAGEConv

from gnn_pruning.models import GCNNodeClassifier, GraphSAGENodeClassifier


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

    if hasattr(pruned_model, "hidden_channels"):
        pruned_model.hidden_channels = int(kept.numel())

    _validate_all_internal_shapes(pruned_model)
    return pruned_model


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

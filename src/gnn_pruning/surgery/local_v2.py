"""Physical local channel deletion for the explicitly supported V2 architectures.

Legacy surgery remains separate. This module neither scores nor chooses channels.
"""
from __future__ import annotations

import copy
from numbers import Integral
from typing import Sequence, Union

import torch
from torch import nn
from torch_geometric.nn import GCNConv, SAGEConv

from gnn_pruning.models import GCNNodeClassifier, GraphSAGENodeClassifier


Classifier = Union[GCNNodeClassifier, GraphSAGENodeClassifier]


def prune_hidden_channels_local_v2(
    model: Classifier,
    layer_index: int,
    keep_indices: Sequence[int],
    min_channels_per_layer: int = 4,
) -> Classifier:
    """Copy a classifier and physically delete one hidden layer's channels.

    Indices refer to the current layer and are stored in ascending order. Only
    selected outputs (including bias) and immediate downstream inputs change.
    Supports homogeneous GCN and mean SAGE without projection/normalization.
    Preserves source storage, device, dtype, module modes and parameter flags.
    The returned model needs a fresh optimizer if later training is authorized.
    """
    _validate_model(model)
    if not _is_integer(layer_index) or not 0 <= layer_index < len(model.convs) - 1:
        raise ValueError("layer_index must identify a hidden layer.")
    if not _is_integer(min_channels_per_layer) or min_channels_per_layer < 1:
        raise ValueError("min_channels_per_layer must be a positive integer.")
    keep = list(keep_indices)
    if any(not _is_integer(index) for index in keep):
        raise ValueError("keep_indices must contain integer channel indices.")
    if len(set(keep)) != len(keep):
        raise ValueError("keep_indices must be unique.")
    keep = sorted(keep)
    width = model.convs[layer_index].out_channels
    if len(keep) < min_channels_per_layer:
        raise ValueError("keep_indices violates min_channels_per_layer.")
    if keep[0] < 0 or keep[-1] >= width:
        raise ValueError("keep_indices out of bounds.")
    if len(keep) == width:
        raise ValueError("Local Surgery must remove at least one channel.")

    # PyG Linear.__deepcopy__ invokes initializers and resets child training mode.
    # Isolate its CPU RNG consumption and restore every module's exact mode.
    with torch.random.fork_rng(devices=[]):
        result = copy.deepcopy(model)
    source_modules = dict(model.named_modules())
    for name, module in result.named_modules():
        module.training = source_modules[name].training

    selected, downstream = result.convs[layer_index:layer_index + 2]
    if isinstance(selected, GCNConv):
        _slice_linear(selected.lin, keep, axis=0)
        if selected.bias is not None:
            selected.bias = _slice_parameter(selected.bias, keep, axis=0)
        _slice_linear(downstream.lin, keep, axis=1)
        for conv in (selected, downstream):
            conv._cached_edge_index = None
            conv._cached_adj_t = None
    else:
        for name in ("lin_l", "lin_r"):
            _slice_linear(getattr(selected, name), keep, axis=0)
            _slice_linear(getattr(downstream, name), keep, axis=1)

    selected.out_channels = len(keep)
    if isinstance(downstream.in_channels, tuple):
        downstream.in_channels = (len(keep), len(keep))
    else:
        downstream.in_channels = len(keep)
    result.hidden_channel_dims = [int(conv.out_channels) for conv in result.convs[:-1]]
    result.hidden_channels = result.hidden_channel_dims[0]
    return result


def _is_integer(value: object) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


def _slice_parameter(parameter: nn.Parameter, keep: list[int], axis: int) -> nn.Parameter:
    indices = torch.tensor(keep, dtype=torch.long, device=parameter.device)
    values = parameter.detach().index_select(axis, indices).clone()
    return nn.Parameter(values, requires_grad=parameter.requires_grad)


def _slice_linear(linear: nn.Module, keep: list[int], axis: int) -> None:
    linear.weight = _slice_parameter(linear.weight, keep, axis)
    if axis == 0 and linear.bias is not None:
        linear.bias = _slice_parameter(linear.bias, keep, axis=0)
    linear.out_channels, linear.in_channels = map(int, linear.weight.shape)


def _validate_model(model: Classifier) -> None:
    if not isinstance(model, (GCNNodeClassifier, GraphSAGENodeClassifier)):
        raise TypeError("V2 Local Surgery requires a GCN or GraphSAGE classifier.")
    conv_type = GCNConv if isinstance(model, GCNNodeClassifier) else SAGEConv
    for i, conv in enumerate(model.convs):
        if type(conv) is not conv_type:
            raise ValueError("V2 Local Surgery requires homogeneous supported conv layers.")
        if isinstance(conv, SAGEConv) and (
            conv.aggr != "mean" or conv.project or conv.normalize or not conv.root_weight
        ):
            raise ValueError("SAGE Local Surgery supports mean/root_weight=True/project=False/normalize=False only.")
        if isinstance(conv, GCNConv) and conv.aggr not in ("add", "sum"):
            raise ValueError("GCN Local Surgery requires additive aggregation.")
        if i > 0:
            channels = conv.in_channels
            inputs = channels if isinstance(channels, tuple) else (channels, channels)
            if any(width != model.convs[i - 1].out_channels for width in inputs):
                raise ValueError("Adjacent conv dimensions must agree before Local Surgery.")

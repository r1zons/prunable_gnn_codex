"""Physical surgery on a processor MLP's second hidden layer."""

import copy
from typing import Sequence

import torch
from torch import nn

from gnn_pruning.models.meshgraphnet import MeshGraphNet, MIN_WIDTH, PROCESSORS


def prune_processor_hidden(
    model: MeshGraphNet, block: int, kind: str, keep: Sequence[int]
) -> MeshGraphNet:
    """Return an independent model with selected current-channel IDs retained."""
    if not 0 <= block < PROCESSORS or kind not in {"edge", "node"}:
        raise ValueError("Expected a processor block and edge or node MLP")
    width = getattr(model.processors[block], kind)[2].out_features
    if any(type(i) is not int or not 0 <= i < width for i in keep):
        raise ValueError("Invalid channel IDs")
    if not MIN_WIDTH <= len(keep) < width or len(set(keep)) != len(keep):
        raise ValueError("Channel IDs must be unique and remove channels above the width floor")
    result = copy.deepcopy(model)
    mlp = getattr(result.processors[block], kind)
    current, following = mlp[2], mlp[4]
    indices = torch.tensor(sorted(keep), device=current.weight.device)
    for layer, field, axis in ((current, "weight", 0), (current, "bias", 0),
                               (following, "weight", 1)):
        parameter = getattr(layer, field)
        sliced = parameter.detach().index_select(axis, indices)
        setattr(layer, field, nn.Parameter(sliced, requires_grad=parameter.requires_grad))
    current.out_features = following.in_features = len(keep)
    return result

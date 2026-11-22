"""Physical channel removal for independently normalized Zhou branches."""

import copy

from .local_v2 import _is_integer, _slice_parameter


def prune_zhou(model, layer, keep, minimum=4):
    """Remove outputs and both corresponding downstream input slices."""
    width = model.hidden_channel_dims[layer] if layer in (0, 1) else 0
    keep = sorted(keep)
    if (not minimum <= len(keep) < width or any(not _is_integer(i) for i in keep)
            or len(set(keep)) != len(keep) or keep[0] < 0 or keep[-1] >= width):
        raise ValueError("Invalid Zhou hidden-channel surgery")
    result = copy.deepcopy(model)
    current = result.layers[layer]
    split = current.self_branch.projection.out_features
    branches = (current.self_branch, current.neighbor_branch)
    indices = ([i for i in keep if i < split], [i - split for i in keep if i >= split])
    for branch, selected in zip(branches, indices):
        linear = branch.projection
        linear.weight = _slice_parameter(linear.weight, selected, axis=0)
        linear.out_features = len(selected)
        for name in ("bias", "offset", "scale"):
            setattr(branch, name, _slice_parameter(getattr(branch, name), selected, axis=0))
    if layer == 0:
        following = result.layers[1]
        downstream = (following.self_branch.projection, following.neighbor_branch.projection)
    else:
        downstream = (result.classifier,)
    for linear in downstream:
        linear.weight = _slice_parameter(linear.weight, keep, axis=1)
        linear.in_features = len(keep)
    return result

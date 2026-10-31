"""Independent physical oracle: no environment previews, scores, rewards or planner.

Only the established physical slicing operation is shared. Legality, incident
magnitude ranking, validation forward, compression, stopping and return are explicit.
"""
from dataclasses import dataclass

import torch

from gnn_pruning.surgery.local_v2 import prune_hidden_channels_local_v2


@dataclass
class Node:
    model: object
    ids: list
    steps: int
    sparsity: float
    accuracy: float
    reward: float = 0.
    penalty: float = 0.
    reason: str | None = None


def actions(model, minimum):
    return [(layer,k) for layer,conv in enumerate(model.convs[:-1])
            for k in (1,2,4,8) if conv.out_channels-k >= minimum]


def choose_channels(model, layer, deletion):
    """Independent incident absolute-weight/bias sums; exact ties by current ID."""
    values = torch.zeros(model.convs[layer].out_channels, dtype=torch.float64)
    for name, parameter in model.named_parameters():
        index = int(name.split('.')[1])
        weight = parameter.detach().abs()
        if index == layer:
            values += (weight.sum(dim=1) if weight.ndim == 2 else weight).cpu().double()
        elif index == layer+1 and weight.ndim == 2:
            values += weight.sum(dim=0).cpu().double()
    ranked = sorted(range(len(values)), key=lambda i:(-float(values[i]),i))
    return sorted(ranked[:len(ranked)-deletion])


def enumerate_paths(root, evaluate, p0, dense_accuracy, target, tolerance=.01,
                    maximum=3, minimum=4, penalty_lambda=2., depth=2):
    """Enumerate full legal paths with real forward-derived rewards and zero tail."""
    def transition(parent, action):
        layer,k = action
        keep = choose_channels(parent.model,layer,k)
        model = prune_hidden_channels_local_v2(parent.model,layer,keep,minimum)
        ids = [list(v) for v in parent.ids]
        ids[layer] = [parent.ids[layer][i] for i in keep]
        compression = 1-sum(p.numel() for p in model.parameters())/p0
        accuracy = evaluate(model)
        steps = parent.steps+1
        reason = ('max_steps' if steps >= maximum else 'no_valid_actions'
                  if not actions(model,minimum) else 'target_reached'
                  if compression >= target-tolerance else None)
        penalty = -penalty_lambda*max(0.,abs(compression-target)-tolerance) if reason else 0.
        reward = .6*(compression-parent.sparsity)-.4*max(0.,dense_accuracy-accuracy)+penalty
        return Node(model,ids,steps,compression,accuracy,reward,penalty,reason)

    paths = []
    for a in actions(root.model,minimum):
        first = transition(root,a)
        if first.reason or depth == 1:
            paths.append(dict(actions=[a],nodes=[first],score=first.reward))
        else:
            for b in actions(first.model,minimum):
                second = transition(first,b)
                paths.append(dict(actions=[a,b],nodes=[first,second],
                                  score=first.reward+.9*second.reward))
    return paths

"""Fixed dense rankings and unique-parameter static Local V2 search."""
from __future__ import annotations

import copy
import itertools
import math
import random
from dataclasses import dataclass
from fractions import Fraction
from typing import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from gnn_pruning.surgery.local_v2 import _validate_model, prune_hidden_channels_local_v2


@dataclass
class StaticPlan:
    """A complete hidden-channel decision, using original dense channel IDs."""
    criterion: str
    widths: list[int]
    kept_indices: list[list[int]]
    requested_sparsity: float
    achieved_sparsity: float
    target_gap: float
    target_violation: bool
    fallback: bool
    parameters_before: int
    parameters_after: int
    removed_score: float | None
    candidate_count: int
    eligible_candidate_count: int


def parameter_count_for_widths(model, widths: Sequence[int]) -> int:
    """Count all physical parameters, including actual per-conv bias presence."""
    _validate_model(model)
    if len(widths) != len(model.convs)-1:
        raise ValueError("Expected one width per hidden conv.")
    if any(not isinstance(w, int) or isinstance(w, bool) or w < 1 or w > c.out_channels
           for w, c in zip(widths, model.convs[:-1])):
        raise ValueError("Invalid hidden widths.")
    dims = [model.in_channels, *widths, model.out_channels]
    result = 0
    for i, conv in enumerate(model.convs):
        if hasattr(conv, "lin_l"):
            result += 2*dims[i]*dims[i+1]
            has_bias = conv.lin_l.bias is not None
        else:
            result += dims[i]*dims[i+1]
            has_bias = conv.bias is not None
        result += int(has_bias)*dims[i+1]
    return result


def _validate_scores(model, scores):
    params = dict(model.named_parameters())
    if scores.keys() != params.keys():
        raise ValueError("Scores must cover every physical parameter, including biases.")
    for name, value in scores.items():
        if value.shape != params[name].shape:
            raise ValueError(f"Score shape mismatch: {name}.")
        if not torch.isfinite(value).all():
            raise ValueError("All saliency scores must be finite.")


def removed_parameter_score(model, scores: Mapping[str, torch.Tensor], keeps) -> float:
    """Sum the union of all removed coordinates, counting shared weights once."""
    _validate_scores(model, scores)
    dims = [model.in_channels, *[c.out_channels for c in model.convs[:-1]], model.out_channels]
    ids = [list(range(dims[0])), *keeps, list(range(dims[-1]))]
    total = 0.0
    for name, value in scores.items():
        layer = int(name.split(".")[1])
        row = torch.zeros(dims[layer+1], dtype=torch.bool, device=value.device)
        row[ids[layer+1]] = True
        retained = row
        if value.ndim == 2:
            col = torch.zeros(dims[layer], dtype=torch.bool, device=value.device)
            col[ids[layer]] = True
            retained = row[:, None] & col[None, :]
        total += float(value[~retained].sum(dtype=torch.float64))
    return total


def channel_rankings(model, scores, maximize: bool = False) -> list[list[int]]:
    """Rank channels by incident saliency, including hidden bias; tie by ID."""
    _validate_scores(model, scores)
    rankings = []
    for layer, conv in enumerate(model.convs[:-1]):
        values = torch.zeros(conv.out_channels, dtype=torch.float64)
        for name, score in scores.items():
            i = int(name.split(".")[1])
            if i == layer:
                marginal = score.sum(dim=1) if score.ndim == 2 else score
                values += marginal.detach().cpu().double()
            elif i == layer+1 and score.ndim == 2:
                values += score.sum(dim=0).detach().cpu().double()
        rankings.append(sorted(range(len(values)), key=lambda c: (float(values[c]) if maximize else -float(values[c]), c)))
    return rankings


def saliency_from_loss(parameters: Mapping[str, nn.Parameter], loss, criterion: str):
    """Compute SNIP or signed GraSP with fixed-vector H*g, including bias terms."""
    if criterion not in {"snip", "grasp"}:
        raise ValueError("Gradient criterion must be snip or grasp.")
    params = tuple(parameters.values())
    grads = torch.autograd.grad(loss, params, create_graph=criterion == "grasp", allow_unused=True)
    if criterion == "snip":
        values = [(p*g).abs() if g is not None else torch.zeros_like(p) for p, g in zip(params, grads)]
    else:
        terms = [(g*g.detach()).sum() for g in grads if g is not None and g.requires_grad]
        hg = torch.autograd.grad(sum(terms), params, allow_unused=True) if terms else [None]*len(params)
        values = [-p*h if h is not None else torch.zeros_like(p) for p, h in zip(params, hg)]
    result = {name: value.detach().clone() for name, value in zip(parameters, values)}
    if any(not torch.isfinite(value).all() for value in result.values()):
        raise ValueError("All saliency scores must be finite.")
    return result


def compute_scores(model, criterion: str, data=None, train_idx=None):
    """Score once on a working copy, eval/dropout-off, train-node CE mean only."""
    if criterion == "random":
        return {}
    if criterion == "magnitude":
        scores = {name: p.detach().abs().clone() for name, p in model.named_parameters()}
    elif criterion in {"snip", "grasp"}:
        if data is None or train_idx is None or len(train_idx) == 0:
            raise ValueError("Train data and nonempty train_idx required for gradient scoring.")
        with torch.random.fork_rng(devices=[]), torch.enable_grad():
            working = copy.deepcopy(model).eval()
            for p in working.parameters():
                p.requires_grad_(True)
            device = next(working.parameters()).device
            batch = data.clone().to(device)
            indices = torch.as_tensor(train_idx, dtype=torch.long, device=device)
            loss = F.cross_entropy(working(batch)[indices], batch.y[indices], reduction="mean")
            scores = saliency_from_loss(dict(working.named_parameters()), loss, criterion)
    else:
        raise ValueError(f"Unknown static V2 criterion: {criterion}.")
    _validate_scores(model, scores)
    return scores


def plan_static(model, scores, criterion: str, target: float, tolerance: float = 0.01,
                min_channels: int = 4, seed: int = 42) -> StaticPlan:
    """Search all width pairs with fixed rankings; fallback to nearest if empty band."""
    _validate_model(model)
    if len(model.convs) != 3:
        raise ValueError("Static V2 protocol requires two hidden layers.")
    if criterion not in {"random", "magnitude", "snip", "grasp"}:
        raise ValueError("Unknown static criterion.")
    if not math.isfinite(target) or not 0 <= target <= 1 or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Target and tolerance must be finite valid fractions.")
    if not isinstance(min_channels, int) or isinstance(min_channels, bool) or min_channels < 1:
        raise ValueError("Invalid minimum width.")
    original = [c.out_channels for c in model.convs[:-1]]
    p0 = sum(p.numel() for p in model.parameters())
    nominal, eps = Fraction(str(target)), Fraction(str(tolerance))
    candidates = []
    for widths in itertools.product(*(range(min_channels, w+1) for w in original)):
        count = parameter_count_for_widths(model, widths)
        if count < p0:
            candidates.append((list(widths), count, Fraction(p0-count, p0)))
    if not candidates:
        raise ValueError("No physically smaller architecture satisfies minimum widths.")
    band = [c for c in candidates if abs(c[2]-nominal) <= eps]
    gap = min(abs(c[2]-nominal) for c in candidates)
    eligible = band or [c for c in candidates if abs(c[2]-nominal) == gap]
    if criterion == "random":
        rng = random.Random(seed)
        widths, count, achieved = rng.choice(eligible)
        keeps = [sorted(rng.sample(range(w), k)) for w, k in zip(original, widths)]
        selected_score = None
    else:
        ranks = channel_rankings(model, scores, maximize=criterion == "grasp")
        choices = []
        for widths, count, achieved in eligible:
            keeps = [sorted(rank[:w]) for rank, w in zip(ranks, widths)]
            score = removed_parameter_score(model, scores, keeps)
            choices.append(((-score if criterion == "grasp" else score, *widths), widths, count, achieved, keeps, score))
        _, widths, count, achieved, keeps, selected_score = min(choices, key=lambda item: item[0])
    return StaticPlan(criterion, widths, keeps, target, float(achieved), float(abs(achieved-nominal)),
                      abs(achieved-nominal) > eps, not bool(band), p0, count, selected_score, len(candidates), len(eligible))


def apply_static_plan(model, plan: StaticPlan):
    """Apply a complete static decision via physical Local Surgery, without scoring."""
    if sum(p.numel() for p in model.parameters()) != plan.parameters_before:
        raise ValueError("Plan/model parameter count mismatch.")
    result = model
    for layer, keep in enumerate(plan.kept_indices):
        if len(keep) < result.convs[layer].out_channels:
            result = prune_hidden_channels_local_v2(result, layer, keep, min_channels_per_layer=1)
    if sum(p.numel() for p in result.parameters()) != plan.parameters_after:
        raise ValueError("Physical parameter count differs from plan.")
    return result

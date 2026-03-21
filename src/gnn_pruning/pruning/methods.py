"""First concrete pruning methods."""

from __future__ import annotations

import copy
import math
import time
from typing import Any, Dict, Tuple

import torch

from gnn_pruning.models import GCNNodeClassifier, GraphSAGENodeClassifier
from gnn_pruning.surgery import structurally_prune_hidden_channels

from .base import BasePruner, PruningContext, PruningPlan
from .registry import register_pruner


def _named_prunable_parameters(model: torch.nn.Module):
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and parameter.ndim > 1:
            yield name, parameter


def _parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _zero_sparsity(model: torch.nn.Module) -> float:
    total = 0
    zeros = 0
    for _, parameter in model.named_parameters():
        total += parameter.numel()
        zeros += int((parameter == 0).sum().item())
    return float(zeros / total) if total else 0.0


def _first_hidden_channels(model: Any) -> int:
    if hasattr(model, "convs") and len(model.convs) >= 2:
        return int(model.convs[0].out_channels)
    raise ValueError("Model does not have supported hidden conv layers.")


def _structured_scores(model: Any, random: bool = False) -> torch.Tensor:
    channels = _first_hidden_channels(model)
    if random:
        return torch.rand(channels)

    conv0 = model.convs[0]
    conv1 = model.convs[1]

    if isinstance(model, GCNNodeClassifier):
        s0 = conv0.lin.weight.abs().sum(dim=1)
        s1 = conv1.lin.weight.abs().sum(dim=0)
        return s0 + s1
    if isinstance(model, GraphSAGENodeClassifier):
        s0 = conv0.lin_l.weight.abs().sum(dim=1)
        if conv0.root_weight:
            s0 = s0 + conv0.lin_r.weight.abs().sum(dim=1)
        s1 = conv1.lin_l.weight.abs().sum(dim=0)
        if conv1.root_weight:
            s1 = s1 + conv1.lin_r.weight.abs().sum(dim=0)
        return s0 + s1

    raise TypeError("Unsupported model for structured pruning scores.")


def _apply_structured(model: Any, scores: torch.Tensor, target_sparsity: float, pruner_name: str, category: str) -> Tuple[Any, PruningPlan]:
    channels = int(scores.numel())
    keep = max(1, int(round((1.0 - target_sparsity) * channels)))
    _, indices = torch.topk(scores, k=keep, largest=True)
    pruned_model = structurally_prune_hidden_channels(model, layer_index=0, keep_indices=indices.tolist())

    achieved = 1.0 - (keep / channels)
    plan = PruningPlan(
        name=pruner_name,
        category=category,
        target_sparsity=float(target_sparsity),
        achieved_sparsity=float(achieved),
        details={"mode": "structured", "layer_index": 0, "kept_channels": keep, "total_channels": channels},
    )
    return pruned_model, plan


def _apply_unstructured_global(model: torch.nn.Module, scores: Dict[str, torch.Tensor], target_sparsity: float, pruner_name: str, category: str) -> Tuple[Any, PruningPlan]:
    pruned_model = copy.deepcopy(model)
    flat_scores = torch.cat([score.flatten() for score in scores.values()])
    prune_count = int(math.floor(target_sparsity * flat_scores.numel()))
    threshold = torch.topk(flat_scores, k=max(flat_scores.numel() - prune_count, 1), largest=True).values.min()

    with torch.no_grad():
        for name, parameter in _named_prunable_parameters(pruned_model):
            mask = scores[name].to(parameter.device) >= threshold
            parameter.mul_(mask)

    plan = PruningPlan(
        name=pruner_name,
        category=category,
        target_sparsity=float(target_sparsity),
        achieved_sparsity=_zero_sparsity(pruned_model),
        details={"mode": "unstructured", "scope": "global"},
    )
    return pruned_model, plan


def _apply_unstructured_layerwise(model: torch.nn.Module, scores: Dict[str, torch.Tensor], target_sparsity: float, pruner_name: str, category: str) -> Tuple[Any, PruningPlan]:
    pruned_model = copy.deepcopy(model)

    with torch.no_grad():
        for name, parameter in _named_prunable_parameters(pruned_model):
            layer_scores = scores[name].flatten()
            prune_count = int(math.floor(target_sparsity * layer_scores.numel()))
            keep_count = max(layer_scores.numel() - prune_count, 1)
            threshold = torch.topk(layer_scores, k=keep_count, largest=True).values.min()
            mask = scores[name].to(parameter.device) >= threshold
            parameter.mul_(mask)

    plan = PruningPlan(
        name=pruner_name,
        category=category,
        target_sparsity=float(target_sparsity),
        achieved_sparsity=_zero_sparsity(pruned_model),
        details={"mode": "unstructured", "scope": "layerwise"},
    )
    return pruned_model, plan


@register_pruner
class RandomPruner(BasePruner):
    name = "random"
    category = "random"
    supports_unstructured = True
    supports_structured = True

    def score(self, model: Any, context: PruningContext) -> Any:
        torch.manual_seed(context.seed)
        param_scores = {name: torch.rand_like(parameter) for name, parameter in _named_prunable_parameters(model)}
        structured_scores = _structured_scores(model, random=True)
        return {"params": param_scores, "structured": structured_scores}

    def apply(self, model: Any, scores: Any, context: PruningContext, target_sparsity: float, structured: bool) -> Tuple[Any, PruningPlan]:
        _ = context
        start = time.perf_counter()
        if structured:
            pruned_model, plan = _apply_structured(model, scores["structured"], target_sparsity, self.name, self.category)
        else:
            pruned_model, plan = _apply_unstructured_global(model, scores["params"], target_sparsity, self.name, self.category)
        plan.pruning_time_sec = float(time.perf_counter() - start)
        return pruned_model, plan


@register_pruner
class GlobalMagnitudePruner(BasePruner):
    name = "global_magnitude"
    category = "magnitude"
    supports_unstructured = True
    supports_structured = True

    def score(self, model: Any, context: PruningContext) -> Any:
        _ = context
        param_scores = {name: parameter.detach().abs().clone() for name, parameter in _named_prunable_parameters(model)}
        structured_scores = _structured_scores(model, random=False)
        return {"params": param_scores, "structured": structured_scores}

    def apply(self, model: Any, scores: Any, context: PruningContext, target_sparsity: float, structured: bool) -> Tuple[Any, PruningPlan]:
        _ = context
        start = time.perf_counter()
        if structured:
            pruned_model, plan = _apply_structured(model, scores["structured"], target_sparsity, self.name, self.category)
        else:
            pruned_model, plan = _apply_unstructured_global(model, scores["params"], target_sparsity, self.name, self.category)
        plan.pruning_time_sec = float(time.perf_counter() - start)
        return pruned_model, plan


@register_pruner
class LayerWiseMagnitudePruner(BasePruner):
    name = "layerwise_magnitude"
    category = "magnitude"
    supports_unstructured = True
    supports_structured = True

    def score(self, model: Any, context: PruningContext) -> Any:
        _ = context
        param_scores = {name: parameter.detach().abs().clone() for name, parameter in _named_prunable_parameters(model)}
        structured_scores = _structured_scores(model, random=False)
        return {"params": param_scores, "structured": structured_scores}

    def apply(self, model: Any, scores: Any, context: PruningContext, target_sparsity: float, structured: bool) -> Tuple[Any, PruningPlan]:
        _ = context
        start = time.perf_counter()
        if structured:
            pruned_model, plan = _apply_structured(model, scores["structured"], target_sparsity, self.name, self.category)
        else:
            pruned_model, plan = _apply_unstructured_layerwise(model, scores["params"], target_sparsity, self.name, self.category)
        plan.pruning_time_sec = float(time.perf_counter() - start)
        return pruned_model, plan

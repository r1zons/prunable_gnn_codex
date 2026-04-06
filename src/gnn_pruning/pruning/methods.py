"""Concrete pruning methods.

This module includes magnitude/random baselines and gradient-saliency pruners
(SNIP and GraSP) that plug into the same score/apply flow.
"""

from __future__ import annotations

import copy
import math
import time
from typing import Any, Dict, Tuple

import torch
from torch import nn

from gnn_pruning.models import GCNNodeClassifier, GraphSAGENodeClassifier
from gnn_pruning.surgery import structurally_prune_hidden_channels

from .base import BasePruner, PruningContext, PruningPlan
from .registry import register_pruner


def _named_prunable_parameters(model: torch.nn.Module):
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and parameter.ndim > 1:
            yield name, parameter


def _named_prunable_parameter_list(model: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    """Materialize prunable parameters as a list for deterministic reuse."""
    return list(_named_prunable_parameters(model))


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


def _parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _hidden_width(model: Any, layer_index: int = 0) -> int:
    if hasattr(model, "convs") and len(model.convs) > layer_index:
        conv = model.convs[layer_index]
        if hasattr(conv, "out_channels"):
            return int(conv.out_channels)
    raise ValueError("Model does not expose hidden width for requested layer.")


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


def _extract_training_batch(context: PruningContext) -> tuple[Any, torch.Tensor, torch.device]:
    """Extract graph data and train indices used for gradient saliency scoring."""
    if not isinstance(context.data, dict):
        raise ValueError("Gradient-based pruners require context.data with {'data', 'train_idx'}.")
    if "data" not in context.data or "train_idx" not in context.data:
        raise ValueError("Gradient-based pruners require context.data keys: data and train_idx.")

    device = torch.device(context.device or "cpu")
    data = context.data["data"].to(device)
    train_idx = context.data["train_idx"]
    if not torch.is_tensor(train_idx):
        train_idx = torch.tensor(train_idx, dtype=torch.long)
    train_idx = train_idx.to(device=device, dtype=torch.long)
    return data, train_idx, device


def _compute_snip_saliency(model: nn.Module, context: PruningContext) -> Dict[str, torch.Tensor]:
    """Compute SNIP saliency: |∂L/∂w * w| on a training subset."""
    data, train_idx, _ = _extract_training_batch(context)
    score_model = copy.deepcopy(model)
    score_model.train()
    score_model.zero_grad(set_to_none=True)

    logits = score_model(data)
    loss = nn.functional.cross_entropy(logits[train_idx], data.y[train_idx])
    loss.backward()

    scores: Dict[str, torch.Tensor] = {}
    for name, parameter in _named_prunable_parameter_list(score_model):
        if parameter.grad is None:
            scores[name] = torch.zeros_like(parameter, device="cpu")
            continue
        scores[name] = (parameter.grad * parameter).detach().abs().cpu()
    return scores


def _compute_grasp_saliency(model: nn.Module, context: PruningContext) -> Dict[str, torch.Tensor]:
    """Compute GraSP saliency via Hessian-gradient product approximation."""
    data, train_idx, _ = _extract_training_batch(context)
    score_model = copy.deepcopy(model)
    score_model.train()
    score_model.zero_grad(set_to_none=True)

    logits = score_model(data)
    loss = nn.functional.cross_entropy(logits[train_idx], data.y[train_idx])
    params = [parameter for _, parameter in _named_prunable_parameter_list(score_model)]
    grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=False)

    grad_dot_weights = torch.zeros((), device=loss.device)
    for grad, param in zip(grads, params):
        grad_dot_weights = grad_dot_weights + (grad * param).sum()
    hessian_grad = torch.autograd.grad(grad_dot_weights, params, create_graph=False, allow_unused=False)

    scores: Dict[str, torch.Tensor] = {}
    for (name, parameter), grad2 in zip(_named_prunable_parameter_list(score_model), hessian_grad):
        scores[name] = (-(parameter * grad2)).detach().abs().cpu()
    return scores


def _structured_scores_from_saliency(model: Any, saliency: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Aggregate per-parameter saliency into hidden-channel scores for surgery."""
    if isinstance(model, GCNNodeClassifier):
        conv0 = saliency["convs.0.lin.weight"].sum(dim=1)
        conv1 = saliency["convs.1.lin.weight"].sum(dim=0)
        return conv0 + conv1
    if isinstance(model, GraphSAGENodeClassifier):
        conv0 = saliency["convs.0.lin_l.weight"].sum(dim=1)
        if getattr(model.convs[0], "root_weight", False):
            conv0 = conv0 + saliency["convs.0.lin_r.weight"].sum(dim=1)
        conv1 = saliency["convs.1.lin_l.weight"].sum(dim=0)
        if getattr(model.convs[1], "root_weight", False):
            conv1 = conv1 + saliency["convs.1.lin_r.weight"].sum(dim=0)
        return conv0 + conv1
    raise TypeError("Unsupported model for structured gradient-based pruning.")


def _validate_sparsity(value: float) -> None:
    if value < 0.0 or value >= 1.0:
        raise ValueError("target_sparsity must be in [0.0, 1.0).")


def _extract_apply_inputs(pruning_plan: Any, context: Any, kwargs: Dict[str, Any]) -> Tuple[PruningPlan, PruningContext]:
    if pruning_plan is None and "plan" in kwargs:
        pruning_plan = kwargs.pop("plan")
    if pruning_plan is None and "scores" in kwargs:
        pruning_plan = kwargs.pop("scores")
    if not isinstance(pruning_plan, PruningPlan):
        raise ValueError("apply requires a valid PruningPlan instance.")
    return pruning_plan, PruningContext.from_input(context)


def _sync_plan_from_kwargs(plan: PruningPlan, kwargs: Dict[str, Any]) -> None:
    target_sparsity = kwargs.get("target_sparsity")
    if target_sparsity is not None:
        _validate_sparsity(float(target_sparsity))
        plan.requested_sparsity = float(target_sparsity)
        plan.target_sparsity = float(target_sparsity)


def _build_plan(name: str, category: str, target_sparsity: float, mode: str, score_payload: Any, layer_index: int = 0) -> PruningPlan:
    _validate_sparsity(target_sparsity)
    return PruningPlan(
        name=name,
        category=category,
        requested_sparsity=float(target_sparsity),
        target_sparsity=float(target_sparsity),
        mode=mode,
        layer_index=layer_index,
        score_payload=score_payload,
        details={"mode": mode, "layer_index": layer_index},
    )


def _apply_structured(model: Any, plan: PruningPlan) -> Any:
    scores = plan.score_payload
    if not isinstance(scores, torch.Tensor):
        raise ValueError("Structured pruning requires tensor channel scores in plan.score_payload.")

    channels = int(scores.numel())
    keep_count = max(1, int(round((1.0 - plan.requested_sparsity) * channels)))

    if plan.target_units:
        if len(plan.target_units) == 0:
            raise ValueError("target_units cannot be empty when provided.")
        keep_indices = sorted(set(int(i) for i in plan.target_units))
    else:
        _, indices = torch.topk(scores, k=keep_count, largest=True)
        keep_indices = indices.tolist()

    layer_index = plan.layer_index or 0
    before_hidden = _hidden_width(model, layer_index=layer_index)
    before_params = _parameter_count(model)

    pruned_model = structurally_prune_hidden_channels(model, layer_index=layer_index, keep_indices=keep_indices)
    after_hidden = _hidden_width(pruned_model, layer_index=layer_index)
    after_params = _parameter_count(pruned_model)

    plan.target_units = keep_indices
    plan.achieved_sparsity = float(1.0 - (len(keep_indices) / channels))
    plan.details.update(
        {
            "scope": "structured_hidden_channels",
            "layer_index": layer_index,
            "kept_channel_indices": keep_indices,
            "kept_channels": len(keep_indices),
            "total_channels": channels,
            "hidden_dim_before": before_hidden,
            "hidden_dim_after": after_hidden,
            "parameter_count_before": before_params,
            "parameter_count_after": after_params,
            "structural_param_reduction": int(before_params - after_params),
        }
    )
    return pruned_model


def _apply_unstructured_global(model: torch.nn.Module, plan: PruningPlan) -> Any:
    scores = plan.score_payload
    if not isinstance(scores, dict):
        raise ValueError("Unstructured pruning requires parameter score dict in plan.score_payload.")

    pruned_model = copy.deepcopy(model)
    flat_scores = torch.cat([score.flatten() for score in scores.values()])
    prune_count = int(math.floor(plan.requested_sparsity * flat_scores.numel()))
    threshold = torch.topk(flat_scores, k=max(flat_scores.numel() - prune_count, 1), largest=True).values.min()

    with torch.no_grad():
        for name, parameter in _named_prunable_parameters(pruned_model):
            mask = scores[name].to(parameter.device) >= threshold
            parameter.mul_(mask)

    plan.achieved_sparsity = _zero_sparsity(pruned_model)
    plan.details.update({"scope": "global", "structural_compression": False})
    return pruned_model


def _apply_unstructured_layerwise(model: torch.nn.Module, plan: PruningPlan) -> Any:
    scores = plan.score_payload
    if not isinstance(scores, dict):
        raise ValueError("Unstructured pruning requires parameter score dict in plan.score_payload.")

    pruned_model = copy.deepcopy(model)

    with torch.no_grad():
        for name, parameter in _named_prunable_parameters(pruned_model):
            layer_scores = scores[name].flatten()
            prune_count = int(math.floor(plan.requested_sparsity * layer_scores.numel()))
            keep_count = max(layer_scores.numel() - prune_count, 1)
            threshold = torch.topk(layer_scores, k=keep_count, largest=True).values.min()
            mask = scores[name].to(parameter.device) >= threshold
            parameter.mul_(mask)

    plan.achieved_sparsity = _zero_sparsity(pruned_model)
    plan.details.update({"scope": "layerwise", "structural_compression": False})
    return pruned_model


@register_pruner
class RandomPruner(BasePruner):
    name = "random"
    category = "random"
    supports_unstructured = True
    supports_structured = True

    def score(self, model: Any, context: Any, **kwargs: Any) -> PruningPlan:
        context = PruningContext.from_input(context)
        target_sparsity = float(kwargs.get("target_sparsity", context.structure_target or 0.5))
        structured = bool(kwargs.get("structured", context.mode == "structured"))

        torch.manual_seed(context.seed)
        if structured:
            payload = _structured_scores(model, random=True)
            return _build_plan(self.name, self.category, target_sparsity, "structured", payload)

        payload = {name: torch.rand_like(parameter) for name, parameter in _named_prunable_parameters(model)}
        return _build_plan(self.name, self.category, target_sparsity, "unstructured", payload)

    def apply(self, model: Any, pruning_plan: PruningPlan = None, context: Any = None, **kwargs: Any) -> Any:
        plan, _ = _extract_apply_inputs(pruning_plan, context, kwargs)
        _sync_plan_from_kwargs(plan, kwargs)
        start = time.perf_counter()
        structured = bool(kwargs.get("structured", plan.mode == "structured"))
        if structured and plan.mode != "structured":
            plan.mode = "structured"
            plan.score_payload = _structured_scores(model, random=True)
        elif not structured and plan.mode != "unstructured":
            plan.mode = "unstructured"
            plan.score_payload = {name: torch.rand_like(parameter) for name, parameter in _named_prunable_parameters(model)}
        plan.details["mode"] = plan.mode
        pruned_model = _apply_structured(model, plan) if structured else _apply_unstructured_global(model, plan)
        plan.pruning_time_sec = float(time.perf_counter() - start)
        plan.score_payload = None
        return pruned_model, plan


@register_pruner
class GlobalMagnitudePruner(BasePruner):
    name = "global_magnitude"
    category = "magnitude"
    supports_unstructured = True
    supports_structured = True

    def score(self, model: Any, context: Any, **kwargs: Any) -> PruningPlan:
        context = PruningContext.from_input(context)
        target_sparsity = float(kwargs.get("target_sparsity", context.structure_target or 0.5))
        structured = bool(kwargs.get("structured", context.mode == "structured"))

        if structured:
            payload = _structured_scores(model, random=False)
            return _build_plan(self.name, self.category, target_sparsity, "structured", payload)

        payload = {name: parameter.detach().abs().clone() for name, parameter in _named_prunable_parameters(model)}
        return _build_plan(self.name, self.category, target_sparsity, "unstructured", payload)

    def apply(self, model: Any, pruning_plan: PruningPlan = None, context: Any = None, **kwargs: Any) -> Any:
        plan, _ = _extract_apply_inputs(pruning_plan, context, kwargs)
        _sync_plan_from_kwargs(plan, kwargs)
        start = time.perf_counter()
        structured = bool(kwargs.get("structured", plan.mode == "structured"))
        if structured and plan.mode != "structured":
            plan.mode = "structured"
            plan.score_payload = _structured_scores(model, random=False)
        elif not structured and plan.mode != "unstructured":
            plan.mode = "unstructured"
            plan.score_payload = {name: parameter.detach().abs().clone() for name, parameter in _named_prunable_parameters(model)}
        plan.details["mode"] = plan.mode
        pruned_model = _apply_structured(model, plan) if structured else _apply_unstructured_global(model, plan)
        plan.pruning_time_sec = float(time.perf_counter() - start)
        plan.score_payload = None
        return pruned_model, plan


@register_pruner
class LayerWiseMagnitudePruner(BasePruner):
    name = "layerwise_magnitude"
    category = "magnitude"
    supports_unstructured = True
    supports_structured = True

    def score(self, model: Any, context: Any, **kwargs: Any) -> PruningPlan:
        context = PruningContext.from_input(context)
        target_sparsity = float(kwargs.get("target_sparsity", context.structure_target or 0.5))
        structured = bool(kwargs.get("structured", context.mode == "structured"))

        if structured:
            payload = _structured_scores(model, random=False)
            return _build_plan(self.name, self.category, target_sparsity, "structured", payload)

        payload = {name: parameter.detach().abs().clone() for name, parameter in _named_prunable_parameters(model)}
        return _build_plan(self.name, self.category, target_sparsity, "unstructured", payload)

    def apply(self, model: Any, pruning_plan: PruningPlan = None, context: Any = None, **kwargs: Any) -> Any:
        plan, _ = _extract_apply_inputs(pruning_plan, context, kwargs)
        _sync_plan_from_kwargs(plan, kwargs)
        start = time.perf_counter()
        structured = bool(kwargs.get("structured", plan.mode == "structured"))
        if structured and plan.mode != "structured":
            plan.mode = "structured"
            plan.score_payload = _structured_scores(model, random=False)
        elif not structured and plan.mode != "unstructured":
            plan.mode = "unstructured"
            plan.score_payload = {name: parameter.detach().abs().clone() for name, parameter in _named_prunable_parameters(model)}
        plan.details["mode"] = plan.mode
        pruned_model = _apply_structured(model, plan) if structured else _apply_unstructured_layerwise(model, plan)
        plan.pruning_time_sec = float(time.perf_counter() - start)
        plan.score_payload = None
        return pruned_model, plan


@register_pruner
class SNIPPruner(BasePruner):
    """Single-shot Network Pruning using connection sensitivity."""

    name = "snip"
    category = "gradient_saliency"
    supports_unstructured = True
    supports_structured = True

    def score(self, model: Any, context: Any, **kwargs: Any) -> PruningPlan:
        context = PruningContext.from_input(context)
        target_sparsity = float(kwargs.get("target_sparsity", context.structure_target or 0.5))
        structured = bool(kwargs.get("structured", context.mode == "structured"))

        saliency = _compute_snip_saliency(model, context)
        if structured:
            payload = _structured_scores_from_saliency(model, saliency)
            return _build_plan(self.name, self.category, target_sparsity, "structured", payload)
        return _build_plan(self.name, self.category, target_sparsity, "unstructured", saliency)

    def apply(self, model: Any, pruning_plan: PruningPlan = None, context: Any = None, **kwargs: Any) -> Any:
        plan, parsed_context = _extract_apply_inputs(pruning_plan, context, kwargs)
        _sync_plan_from_kwargs(plan, kwargs)
        start = time.perf_counter()
        structured = bool(kwargs.get("structured", plan.mode == "structured"))
        if structured and plan.mode != "structured":
            plan.mode = "structured"
            plan.score_payload = _structured_scores_from_saliency(model, _compute_snip_saliency(model, parsed_context))
        elif not structured and plan.mode != "unstructured":
            plan.mode = "unstructured"
            plan.score_payload = _compute_snip_saliency(model, parsed_context)
        plan.details["mode"] = plan.mode
        pruned_model = _apply_structured(model, plan) if structured else _apply_unstructured_global(model, plan)
        plan.pruning_time_sec = float(time.perf_counter() - start)
        plan.score_payload = None
        return pruned_model, plan


@register_pruner
class GraSPPruner(BasePruner):
    """Gradient Signal Preservation pruning via second-order saliency."""

    name = "grasp"
    category = "gradient_saliency"
    supports_unstructured = True
    supports_structured = True

    def score(self, model: Any, context: Any, **kwargs: Any) -> PruningPlan:
        context = PruningContext.from_input(context)
        target_sparsity = float(kwargs.get("target_sparsity", context.structure_target or 0.5))
        structured = bool(kwargs.get("structured", context.mode == "structured"))

        saliency = _compute_grasp_saliency(model, context)
        if structured:
            payload = _structured_scores_from_saliency(model, saliency)
            return _build_plan(self.name, self.category, target_sparsity, "structured", payload)
        return _build_plan(self.name, self.category, target_sparsity, "unstructured", saliency)

    def apply(self, model: Any, pruning_plan: PruningPlan = None, context: Any = None, **kwargs: Any) -> Any:
        plan, parsed_context = _extract_apply_inputs(pruning_plan, context, kwargs)
        _sync_plan_from_kwargs(plan, kwargs)
        start = time.perf_counter()
        structured = bool(kwargs.get("structured", plan.mode == "structured"))
        if structured and plan.mode != "structured":
            plan.mode = "structured"
            plan.score_payload = _structured_scores_from_saliency(model, _compute_grasp_saliency(model, parsed_context))
        elif not structured and plan.mode != "unstructured":
            plan.mode = "unstructured"
            plan.score_payload = _compute_grasp_saliency(model, parsed_context)
        plan.details["mode"] = plan.mode
        pruned_model = _apply_structured(model, plan) if structured else _apply_unstructured_global(model, plan)
        plan.pruning_time_sec = float(time.perf_counter() - start)
        plan.score_payload = None
        return pruned_model, plan

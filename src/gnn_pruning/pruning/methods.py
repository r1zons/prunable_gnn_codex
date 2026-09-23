"""Concrete pruning methods.

This module includes magnitude/random baselines and gradient-saliency pruners
(SNIP and GraSP) that plug into the same score/apply flow.
"""

from __future__ import annotations

import copy
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch import nn

from gnn_pruning.models import GCNNodeClassifier, GraphSAGENodeClassifier
from gnn_pruning.rl import StructuralPruningEnv, action_key, select_action, state_key, update_q
from gnn_pruning.surgery import can_apply_structural_prune, structurally_prune_hidden_channels

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


def _structured_scores_for_layer(model: Any, layer_index: int) -> torch.Tensor:
    if not hasattr(model, "convs") or len(model.convs) < 2:
        raise ValueError("Model does not expose hidden layers for structured pruning.")
    if layer_index < 0 or layer_index >= len(model.convs) - 1:
        raise ValueError("layer_index must target a hidden layer with a downstream layer.")

    conv = model.convs[layer_index]
    next_conv = model.convs[layer_index + 1]

    if isinstance(model, GCNNodeClassifier):
        left = conv.lin.weight.abs().sum(dim=1)
        right = next_conv.lin.weight.abs().sum(dim=0)
        return left + right
    if isinstance(model, GraphSAGENodeClassifier):
        left = conv.lin_l.weight.abs().sum(dim=1)
        if getattr(conv, "root_weight", False):
            left = left + conv.lin_r.weight.abs().sum(dim=1)
        right = next_conv.lin_l.weight.abs().sum(dim=0)
        if getattr(next_conv, "root_weight", False):
            right = right + next_conv.lin_r.weight.abs().sum(dim=0)
        return left + right
    raise TypeError("Unsupported model for layer-wise structured scores.")


def _adaptive_cfg(context: PruningContext) -> Dict[str, Any]:
    config = context.config if isinstance(context.config, dict) else {}
    payload = config.get("adaptive_pruning", {})
    if not isinstance(payload, dict):
        payload = {}
    reward = payload.get("reward", {}) if isinstance(payload.get("reward", {}), dict) else {}
    return {
        "step_prune_ratio": float(payload.get("step_prune_ratio", 0.1)),
        "max_steps": int(payload.get("max_steps", 10)),
        "max_accuracy_drop": float(payload.get("max_accuracy_drop", 0.05)),
        "min_channels_per_layer": int(payload.get("min_channels_per_layer", 4)),
        "alpha": float(reward.get("alpha", 0.4)),
        "beta": float(reward.get("beta", 0.2)),
        "gamma": float(reward.get("gamma", 0.4)),
    }


def _is_stop_action(action: Dict[str, Any]) -> bool:
    return str(action.get("type", "")).strip().lower() == "stop"


def _stop_action() -> Dict[str, str]:
    return {"type": "stop"}


def _valid_pruning_actions(
    *,
    model: Any,
    actions: List[Dict[str, float]],
    min_channels_per_layer: int,
    pruned_layers: set[int],
) -> List[Dict[str, float]]:
    snapshot = _analyze_action_space(
        model=model,
        actions=actions,
        min_channels_per_layer=min_channels_per_layer,
        pruned_layers=pruned_layers,
    )
    return [dict(action) for action in snapshot["valid_actions"]]


def _hidden_widths(model: Any) -> list[int]:
    widths: list[int] = []
    for conv in getattr(model, "convs", [])[:-1]:
        if hasattr(conv, "out_channels"):
            widths.append(int(conv.out_channels))
    return widths


def _analyze_action_space(
    *,
    model: Any,
    actions: List[Dict[str, float]],
    min_channels_per_layer: int,
    pruned_layers: set[int],
    allow_nonmonotonic_layer_order: bool = False,
    structural_pruning_mode: str = "cascade",
    max_examples: int = 5,
) -> Dict[str, Any]:
    reasons = {
        "invalid_layer_index": 0,
        "invalid_layer_order": 0,
        "min_channels_per_layer": 0,
        "score_width_mismatch": 0,
        "invalid_keep_count": 0,
        "keep_indices_out_of_bounds": 0,
        "structural_feasibility_failed": 0,
    }
    valid: List[Dict[str, float]] = []
    valid_by_layer: Dict[str, int] = {}
    examples: List[Dict[str, Any]] = []
    hidden_layers = max(0, len(getattr(model, "convs", [])) - 1)

    def mark(reason: str, action: Dict[str, float]) -> None:
        reasons[reason] = int(reasons.get(reason, 0)) + 1
        if len(examples) < int(max_examples):
            examples.append({"action": dict(action), "reason": reason})

    for action in actions:
        layer_index = int(action.get("layer_index", -1))
        prune_ratio = float(action.get("prune_ratio", 0.0))
        if layer_index < 0 or layer_index >= hidden_layers:
            mark("invalid_layer_index", action)
            continue
        if (not allow_nonmonotonic_layer_order) and pruned_layers and layer_index < max(pruned_layers):
            mark("invalid_layer_order", action)
            continue

        width = _hidden_width(model, layer_index=layer_index)
        if width <= int(min_channels_per_layer):
            mark("min_channels_per_layer", action)
            continue

        prune_count = max(1, int(round(prune_ratio * width)))
        keep_count = max(int(min_channels_per_layer), width - prune_count)
        keep_count = min(keep_count, width - 1) if width > int(min_channels_per_layer) else width
        if keep_count <= 0 or keep_count < int(min_channels_per_layer) or keep_count >= width:
            mark("invalid_keep_count", action)
            continue

        scores = _structured_scores_for_layer(model, layer_index=layer_index)
        if int(scores.numel()) != int(width):
            mark("score_width_mismatch", action)
            continue

        _, keep_idx_tensor = torch.topk(scores, k=keep_count, largest=True)
        keep_indices = [int(index) for index in keep_idx_tensor.tolist()]
        if not keep_indices or min(keep_indices) < 0 or max(keep_indices) >= width:
            mark("keep_indices_out_of_bounds", action)
            continue

        feasibility = can_apply_structural_prune(
            model,
            layer_index=layer_index,
            keep_indices=keep_indices,
            min_channels_per_layer=int(min_channels_per_layer),
            mode=str(structural_pruning_mode).strip().lower(),
        )
        if not feasibility.valid:
            mark("structural_feasibility_failed", action)
            if len(examples) <= int(max_examples):
                examples[-1]["reason_detail"] = str(feasibility.reason)
            continue

        valid.append(dict(action))
        key = str(layer_index)
        valid_by_layer[key] = int(valid_by_layer.get(key, 0)) + 1

    return {
        "valid_actions": valid,
        "total_candidate_actions": int(len(actions)),
        "num_valid_actions": int(len(valid)),
        "valid_actions_by_layer": valid_by_layer,
        "filtered_actions_by_reason": reasons,
        "filtered_action_examples": examples,
        "hidden_widths_before": _hidden_widths(model),
        "last_pruned_layer": int(max(pruned_layers)) if pruned_layers else -1,
        "pruning_order_state": sorted(int(v) for v in pruned_layers),
        "min_channels_per_layer": int(min_channels_per_layer),
    }


def _target_gap(*, target_sparsity: float, current_sparsity: float) -> float:
    return float(max(0.0, float(target_sparsity) - float(current_sparsity)))


def _current_rollout_metrics(env: StructuralPruningEnv) -> Tuple[float, float, float]:
    current_params = _parameter_count(env.current_model)
    current_sparsity = float(1.0 - (current_params / max(1, env.initial_param_count)))
    current_val = env._validation_accuracy(env.current_model)  # type: ignore[attr-defined]
    accuracy_drop = max(0.0, float(env.baseline_val_acc) - float(current_val))
    target_gap = _target_gap(target_sparsity=float(env.target_sparsity), current_sparsity=current_sparsity)
    return current_sparsity, float(accuracy_drop), float(target_gap)


def _gap_penalty_factor(target_gap: float, terminal_cfg: Dict[str, Any]) -> Tuple[float, float, float]:
    target_tolerance = float(terminal_cfg.get("target_tolerance", 0.05))
    gap_scale = max(float(terminal_cfg.get("gap_scale", 0.25)), 1e-8)
    effective_gap = max(0.0, float(target_gap) - float(target_tolerance))
    severity = min(1.0, effective_gap / gap_scale)
    factor = float(severity * severity)
    return factor, float(target_tolerance), float(gap_scale)


def _terminal_reward(
    *,
    stop_selected: bool,
    achieved_sparsity: float,
    accuracy_drop: float,
    target_sparsity: float,
    max_accuracy_drop: float,
    stop_reason: str,
    terminal_cfg: Dict[str, Any],
) -> float:
    target_reached_bonus = float(terminal_cfg.get("target_reached_bonus", 0.10))
    accuracy_failure_penalty = float(
        terminal_cfg.get("accuracy_failure_penalty", terminal_cfg.get("accuracy_exceeded_penalty", 0.10))
    )
    early_stop_max_penalty = float(
        terminal_cfg.get("early_stop_max_penalty", terminal_cfg.get("stop_target_gap_penalty", 0.08))
    )
    no_valid_max_penalty = float(terminal_cfg.get("no_valid_max_penalty", 0.12))
    stop_bonus = float(terminal_cfg.get("stop_bonus", 0.0))

    reward = 0.0
    target_gap = _target_gap(target_sparsity=target_sparsity, current_sparsity=achieved_sparsity)
    gap_penalty_factor, target_tolerance, _ = _gap_penalty_factor(target_gap, terminal_cfg)
    normalized_stop_reason = str(stop_reason).strip()

    if target_gap <= 0.0 and accuracy_drop <= max_accuracy_drop:
        reward += target_reached_bonus
    if target_gap > 0.0 and accuracy_drop > max_accuracy_drop:
        reward -= accuracy_failure_penalty
    if normalized_stop_reason == "agent_stop" and target_gap > target_tolerance:
        reward -= early_stop_max_penalty * gap_penalty_factor
    if normalized_stop_reason == "no_valid_action" and target_gap > target_tolerance:
        reward -= no_valid_max_penalty * gap_penalty_factor
    if stop_selected and normalized_stop_reason == "agent_stop":
        if target_gap <= target_tolerance and achieved_sparsity > 0.0 and accuracy_drop <= max_accuracy_drop:
            reward += stop_bonus
    return float(reward)


def _enrich_trace_entries(trace: List[Dict[str, Any]], *, target_sparsity: float, terminal_cfg: Dict[str, Any]) -> None:
    for entry in trace:
        info = entry.get("info", {})
        current_sparsity = float(info.get("current_sparsity", 0.0))
        target_gap = _target_gap(target_sparsity=target_sparsity, current_sparsity=current_sparsity)
        gap_penalty_factor, target_tolerance, gap_scale = _gap_penalty_factor(target_gap, terminal_cfg)
        state = entry.get("state", {})
        next_state = entry.get("next_state", {})
        target_gap_bucket = int(next_state.get("target_gap_bucket", state.get("target_gap_bucket", 0)))
        stop_reason = str(info.get("stop_reason", "")).strip()

        entry["target_sparsity"] = float(target_sparsity)
        entry["current_sparsity"] = float(current_sparsity)
        entry["target_gap"] = float(target_gap)
        entry["target_gap_bucket"] = int(target_gap_bucket)
        entry["terminal_reward"] = float(entry.get("terminal_reward", 0.0))
        entry["stop_selected"] = bool(entry.get("stop_selected", False))
        entry["stop_reason"] = stop_reason
        entry["gap_penalty_factor"] = float(gap_penalty_factor)
        entry["target_tolerance"] = float(target_tolerance)
        entry["gap_scale"] = float(gap_scale)
        if isinstance(info, dict):
            info.setdefault("target_sparsity", float(target_sparsity))
            info.setdefault("current_sparsity", float(current_sparsity))
            info.setdefault("target_gap", float(target_gap))
            info.setdefault("target_gap_bucket", int(target_gap_bucket))
            info.setdefault("gap_penalty_factor", float(gap_penalty_factor))
            info.setdefault("target_tolerance", float(target_tolerance))
            info.setdefault("gap_scale", float(gap_scale))
            info.setdefault("stop_reason", stop_reason)


def _diagnostic_step_fields(
    *,
    snapshot: Dict[str, Any],
    current_sparsity: float,
    target_sparsity: float,
    target_gap: float,
    stop_available: bool,
) -> Dict[str, Any]:
    return {
        "hidden_widths_before": list(snapshot.get("hidden_widths_before", [])),
        "min_channels_per_layer": int(snapshot.get("min_channels_per_layer", 0)),
        "last_pruned_layer": int(snapshot.get("last_pruned_layer", -1)),
        "pruning_order_state": list(snapshot.get("pruning_order_state", [])),
        "total_candidate_actions": int(snapshot.get("total_candidate_actions", 0)),
        "num_valid_actions": int(snapshot.get("num_valid_actions", 0)),
        "valid_actions_by_layer": dict(snapshot.get("valid_actions_by_layer", {})),
        "filtered_actions_by_reason": dict(snapshot.get("filtered_actions_by_reason", {})),
        "filtered_action_examples": list(snapshot.get("filtered_action_examples", [])),
        "stop_available": bool(stop_available),
        "target_reached": bool(float(current_sparsity) >= float(target_sparsity)),
        "current_sparsity": float(current_sparsity),
        "target_sparsity": float(target_sparsity),
        "target_gap": float(target_gap),
    }


def _aggregate_filter_reasons(entries: List[Dict[str, Any]]) -> Dict[str, int]:
    aggregate: Dict[str, int] = {}
    for entry in entries:
        reasons = entry.get("filtered_actions_by_reason", {})
        if not isinstance(reasons, dict):
            continue
        for key, value in reasons.items():
            aggregate[key] = int(aggregate.get(key, 0)) + int(value)
    return aggregate


def _dominant_reason(reasons: Dict[str, int]) -> str:
    if not reasons:
        return ""
    key, value = max(reasons.items(), key=lambda item: int(item[1]))
    return str(key) if int(value) > 0 else ""


def _validation_accuracy(model: Any, context: PruningContext) -> float:
    if not isinstance(context.data, dict) or "data" not in context.data or "val_idx" not in context.data:
        return 0.0
    data = context.data["data"]
    val_idx = context.data["val_idx"]
    device = torch.device(context.device or "cpu")
    model = model.to(device)
    graph = data.to(device)
    if not torch.is_tensor(val_idx):
        val_idx = torch.tensor(val_idx, dtype=torch.long)
    val_idx = val_idx.to(device=device, dtype=torch.long)
    model.eval()
    with torch.no_grad():
        logits = model(graph)
        pred = logits.argmax(dim=-1)
        correct = (pred[val_idx] == graph.y[val_idx]).float().mean()
    return float(correct.item())


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


def _group_lasso_channel_penalty(model: Any) -> torch.Tensor:
    """Return channel-wise Group Lasso penalties for the first hidden layer."""
    if isinstance(model, GCNNodeClassifier):
        conv0 = model.convs[0].lin.weight.norm(p=2, dim=1)
        conv1 = model.convs[1].lin.weight.norm(p=2, dim=0)
        return conv0 + conv1
    if isinstance(model, GraphSAGENodeClassifier):
        conv0 = model.convs[0].lin_l.weight.norm(p=2, dim=1)
        if getattr(model.convs[0], "root_weight", False):
            conv0 = conv0 + model.convs[0].lin_r.weight.norm(p=2, dim=1)
        conv1 = model.convs[1].lin_l.weight.norm(p=2, dim=0)
        if getattr(model.convs[1], "root_weight", False):
            conv1 = conv1 + model.convs[1].lin_r.weight.norm(p=2, dim=0)
        return conv0 + conv1
    raise TypeError("Unsupported model for Group Lasso channel penalties.")


def _compute_l1_regularized_saliency(model: nn.Module, context: PruningContext, reg_strength: float) -> Dict[str, torch.Tensor]:
    """Compute saliency with L1 regularization in a pruning-time scoring pass."""
    data, train_idx, _ = _extract_training_batch(context)
    score_model = copy.deepcopy(model)
    score_model.train()
    score_model.zero_grad(set_to_none=True)

    logits = score_model(data)
    loss = nn.functional.cross_entropy(logits[train_idx], data.y[train_idx])
    l1_term = torch.zeros((), device=loss.device)
    for _, parameter in _named_prunable_parameter_list(score_model):
        l1_term = l1_term + parameter.abs().sum()
    objective = loss + (float(reg_strength) * l1_term)
    objective.backward()

    scores: Dict[str, torch.Tensor] = {}
    for name, parameter in _named_prunable_parameter_list(score_model):
        if parameter.grad is None:
            scores[name] = torch.zeros_like(parameter, device="cpu")
            continue
        scores[name] = (parameter.grad * parameter).detach().abs().cpu()
    return scores


def _compute_group_lasso_saliency(model: nn.Module, context: PruningContext, reg_strength: float) -> torch.Tensor:
    """Compute channel saliency using Group-Lasso-regularized objective."""
    data, train_idx, _ = _extract_training_batch(context)
    score_model = copy.deepcopy(model)
    score_model.train()
    score_model.zero_grad(set_to_none=True)

    logits = score_model(data)
    loss = nn.functional.cross_entropy(logits[train_idx], data.y[train_idx])
    group_penalty = _group_lasso_channel_penalty(score_model).sum()
    objective = loss + (float(reg_strength) * group_penalty)
    objective.backward()

    saliency = {}
    for name, parameter in _named_prunable_parameter_list(score_model):
        if parameter.grad is None:
            saliency[name] = torch.zeros_like(parameter, device="cpu")
            continue
        saliency[name] = (parameter.grad * parameter).detach().abs().cpu()
    channel_scores = _structured_scores_from_saliency(score_model, saliency)
    channel_scores = channel_scores + (float(reg_strength) * _group_lasso_channel_penalty(score_model).detach().cpu())
    return channel_scores


def _compute_movement_scores(
    model: nn.Module,
    context: PruningContext,
    steps: int,
    lr: float,
) -> Dict[str, torch.Tensor]:
    """Compute movement scores by accumulating gradient-driven weight updates."""
    data, train_idx, _ = _extract_training_batch(context)
    score_model = copy.deepcopy(model)
    score_model.train()
    params = _named_prunable_parameter_list(score_model)
    movement = {name: torch.zeros_like(parameter, device="cpu") for name, parameter in params}

    for _ in range(max(1, int(steps))):
        score_model.zero_grad(set_to_none=True)
        logits = score_model(data)
        loss = nn.functional.cross_entropy(logits[train_idx], data.y[train_idx])
        loss.backward()
        with torch.no_grad():
            for name, parameter in params:
                if parameter.grad is None:
                    continue
                delta = -float(lr) * parameter.grad
                movement[name] = movement[name] + (delta * parameter).detach().abs().cpu()
                parameter.add_(delta)
    return movement


def _hard_concrete_sample(log_alpha: torch.Tensor, beta: float, gamma: float, zeta: float, training: bool) -> torch.Tensor:
    """Sample (or deterministically estimate) stretched hard-concrete gates."""
    if training:
        u = torch.rand_like(log_alpha).clamp_(1e-6, 1.0 - 1e-6)
        s = torch.sigmoid((u.log() - (1 - u).log() + log_alpha) / beta)
    else:
        s = torch.sigmoid(log_alpha)
    s_bar = s * (zeta - gamma) + gamma
    return s_bar.clamp(0.0, 1.0)


def _compute_hard_concrete_scores(
    model: nn.Module,
    context: PruningContext,
    steps: int,
    lr: float,
    l0_lambda: float,
    beta: float,
    gamma: float = -0.1,
    zeta: float = 1.1,
) -> Dict[str, torch.Tensor]:
    """Optimize hard-concrete gate logits and return expected gate probabilities."""
    data, train_idx, _ = _extract_training_batch(context)
    score_model = copy.deepcopy(model)
    score_model.train()
    params = _named_prunable_parameter_list(score_model)
    log_alpha = {
        name: torch.zeros_like(parameter, device=parameter.device, requires_grad=True)
        for name, parameter in params
    }
    optimizer = torch.optim.Adam(log_alpha.values(), lr=float(lr))

    for _ in range(max(1, int(steps))):
        score_model.zero_grad(set_to_none=True)
        logits = score_model(data)
        cls_loss = nn.functional.cross_entropy(logits[train_idx], data.y[train_idx])
        cls_loss.backward()

        optimizer.zero_grad()
        gates = {name: _hard_concrete_sample(alpha, beta=beta, gamma=gamma, zeta=zeta, training=True) for name, alpha in log_alpha.items()}
        importance_term = torch.zeros((), device=logits.device)
        for name, parameter in params:
            if parameter.grad is None:
                continue
            importance = (parameter.grad * parameter).detach().abs()
            importance_term = importance_term - (importance * gates[name]).mean()
        expected_l0 = torch.stack([torch.sigmoid(alpha - beta * math.log(-gamma / zeta)).mean() for alpha in log_alpha.values()]).mean()
        loss = importance_term + float(l0_lambda) * expected_l0
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        scores = {
            name: _hard_concrete_sample(alpha, beta=beta, gamma=gamma, zeta=zeta, training=False).detach().cpu()
            for name, alpha in log_alpha.items()
        }
    return scores


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
            "selected_layer_indices": [int(layer_index)],
            "prunable_channel_groups": 1,
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


def _apply_unstructured_threshold(model: torch.nn.Module, plan: PruningPlan) -> Any:
    """Apply saliency threshold pruning using score payload and fixed threshold."""
    scores = plan.score_payload
    if not isinstance(scores, dict):
        raise ValueError("Threshold pruning requires parameter score dict in plan.score_payload.")
    threshold = float(plan.details.get("score_threshold", 0.0))

    pruned_model = copy.deepcopy(model)
    with torch.no_grad():
        for name, parameter in _named_prunable_parameters(pruned_model):
            mask = scores[name].to(parameter.device) >= threshold
            parameter.mul_(mask)

    plan.achieved_sparsity = _zero_sparsity(pruned_model)
    plan.details.update({"scope": "threshold", "structural_compression": False})
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


@register_pruner
class L1ThresholdPruner(BasePruner):
    """L1-regularized saliency with explicit threshold-based pruning."""

    name = "l1_threshold"
    category = "regularization"
    supports_unstructured = True
    supports_structured = False

    def score(self, model: Any, context: Any, **kwargs: Any) -> PruningPlan:
        context = PruningContext.from_input(context)
        target_sparsity = float(kwargs.get("target_sparsity", context.structure_target or 0.5))
        reg_strength = float(kwargs.get("reg_strength", context.config.get("pruning", {}).get("reg_strength", 1e-4)))
        saliency = _compute_l1_regularized_saliency(model, context, reg_strength=reg_strength)
        flat = torch.cat([score.flatten() for score in saliency.values()])
        prune_count = int(math.floor(target_sparsity * flat.numel()))
        keep_count = max(flat.numel() - prune_count, 1)
        threshold = float(torch.topk(flat, k=keep_count, largest=True).values.min().item())
        plan = _build_plan(self.name, self.category, target_sparsity, "unstructured", saliency)
        plan.details.update({"regularization": "l1", "reg_strength": reg_strength, "score_threshold": threshold})
        return plan

    def apply(self, model: Any, pruning_plan: PruningPlan = None, context: Any = None, **kwargs: Any) -> Any:
        plan, parsed_context = _extract_apply_inputs(pruning_plan, context, kwargs)
        _sync_plan_from_kwargs(plan, kwargs)
        start = time.perf_counter()
        if plan.mode != "unstructured":
            plan.mode = "unstructured"
            reg_strength = float(kwargs.get("reg_strength", 1e-4))
            plan.score_payload = _compute_l1_regularized_saliency(model, parsed_context, reg_strength=reg_strength)
            plan.details["reg_strength"] = reg_strength
        if "score_threshold" not in plan.details:
            flat = torch.cat([score.flatten() for score in plan.score_payload.values()])
            prune_count = int(math.floor(plan.requested_sparsity * flat.numel()))
            keep_count = max(flat.numel() - prune_count, 1)
            plan.details["score_threshold"] = float(torch.topk(flat, k=keep_count, largest=True).values.min().item())
        plan.details["mode"] = "unstructured"
        pruned_model = _apply_unstructured_threshold(model, plan)
        plan.pruning_time_sec = float(time.perf_counter() - start)
        plan.score_payload = None
        return pruned_model, plan


@register_pruner
class GroupLassoPruner(BasePruner):
    """Group-Lasso channel saliency with structural hidden-channel pruning."""

    name = "group_lasso"
    category = "regularization"
    supports_unstructured = False
    supports_structured = True

    def score(self, model: Any, context: Any, **kwargs: Any) -> PruningPlan:
        context = PruningContext.from_input(context)
        target_sparsity = float(kwargs.get("target_sparsity", context.structure_target or 0.5))
        reg_strength = float(kwargs.get("reg_strength", context.config.get("pruning", {}).get("reg_strength", 1e-4)))
        payload = _compute_group_lasso_saliency(model, context, reg_strength=reg_strength)
        plan = _build_plan(self.name, self.category, target_sparsity, "structured", payload)
        plan.details.update({"regularization": "group_lasso", "reg_strength": reg_strength})
        return plan

    def apply(self, model: Any, pruning_plan: PruningPlan = None, context: Any = None, **kwargs: Any) -> Any:
        plan, parsed_context = _extract_apply_inputs(pruning_plan, context, kwargs)
        _sync_plan_from_kwargs(plan, kwargs)
        start = time.perf_counter()
        if plan.mode != "structured":
            plan.mode = "structured"
            reg_strength = float(kwargs.get("reg_strength", 1e-4))
            plan.score_payload = _compute_group_lasso_saliency(model, parsed_context, reg_strength=reg_strength)
            plan.details["reg_strength"] = reg_strength
        plan.details["mode"] = "structured"
        pruned_model = _apply_structured(model, plan)
        plan.pruning_time_sec = float(time.perf_counter() - start)
        plan.score_payload = None
        return pruned_model, plan


@register_pruner
class MovementPruner(BasePruner):
    """Movement pruning (unstructured).

    Structural compression: not directly supported. This method zeroes weights and
    therefore requires a later compaction/surgery step for true model shrinkage.
    """

    name = "movement"
    category = "learnable"
    supports_unstructured = True
    supports_structured = False

    def score(self, model: Any, context: Any, **kwargs: Any) -> PruningPlan:
        context = PruningContext.from_input(context)
        target_sparsity = float(kwargs.get("target_sparsity", context.structure_target or 0.5))
        steps = int(kwargs.get("movement_steps", context.config.get("pruning", {}).get("movement_steps", 3)))
        lr = float(kwargs.get("movement_lr", context.config.get("pruning", {}).get("movement_lr", 1e-2)))
        payload = _compute_movement_scores(model, context, steps=steps, lr=lr)
        plan = _build_plan(self.name, self.category, target_sparsity, "unstructured", payload)
        plan.details.update(
            {
                "learnable_pruning": True,
                "movement_steps": steps,
                "movement_lr": lr,
                "structural_compression_support": "requires_compaction_step",
            }
        )
        return plan

    def apply(self, model: Any, pruning_plan: PruningPlan = None, context: Any = None, **kwargs: Any) -> Any:
        plan, parsed_context = _extract_apply_inputs(pruning_plan, context, kwargs)
        _sync_plan_from_kwargs(plan, kwargs)
        start = time.perf_counter()
        if plan.mode != "unstructured":
            plan.mode = "unstructured"
            steps = int(kwargs.get("movement_steps", 3))
            lr = float(kwargs.get("movement_lr", 1e-2))
            plan.score_payload = _compute_movement_scores(model, parsed_context, steps=steps, lr=lr)
            plan.details["movement_steps"] = steps
            plan.details["movement_lr"] = lr
        plan.details["mode"] = "unstructured"
        pruned_model = _apply_unstructured_global(model, plan)
        plan.pruning_time_sec = float(time.perf_counter() - start)
        plan.score_payload = None
        return pruned_model, plan


@register_pruner
class HardConcretePruner(BasePruner):
    """Hard-concrete / L0-style gating (unstructured).

    Structural compression: not directly supported. Gates induce sparse masks and
    require subsequent structural compaction if true shrinkage is desired.
    """

    name = "hard_concrete_l0"
    category = "learnable"
    supports_unstructured = True
    supports_structured = False

    def score(self, model: Any, context: Any, **kwargs: Any) -> PruningPlan:
        context = PruningContext.from_input(context)
        target_sparsity = float(kwargs.get("target_sparsity", context.structure_target or 0.5))
        steps = int(kwargs.get("gate_steps", context.config.get("pruning", {}).get("gate_steps", 5)))
        lr = float(kwargs.get("gate_lr", context.config.get("pruning", {}).get("gate_lr", 1e-2)))
        l0_lambda = float(kwargs.get("l0_lambda", context.config.get("pruning", {}).get("l0_lambda", 1e-3)))
        beta = float(kwargs.get("gate_beta", context.config.get("pruning", {}).get("gate_beta", 2.0 / 3.0)))
        payload = _compute_hard_concrete_scores(
            model,
            context,
            steps=steps,
            lr=lr,
            l0_lambda=l0_lambda,
            beta=beta,
        )
        plan = _build_plan(self.name, self.category, target_sparsity, "unstructured", payload)
        plan.details.update(
            {
                "learnable_pruning": True,
                "gate_steps": steps,
                "gate_lr": lr,
                "l0_lambda": l0_lambda,
                "gate_beta": beta,
                "structural_compression_support": "requires_compaction_step",
            }
        )
        return plan

    def apply(self, model: Any, pruning_plan: PruningPlan = None, context: Any = None, **kwargs: Any) -> Any:
        plan, parsed_context = _extract_apply_inputs(pruning_plan, context, kwargs)
        _sync_plan_from_kwargs(plan, kwargs)
        start = time.perf_counter()
        if plan.mode != "unstructured":
            plan.mode = "unstructured"
            steps = int(kwargs.get("gate_steps", 5))
            lr = float(kwargs.get("gate_lr", 1e-2))
            l0_lambda = float(kwargs.get("l0_lambda", 1e-3))
            beta = float(kwargs.get("gate_beta", 2.0 / 3.0))
            plan.score_payload = _compute_hard_concrete_scores(
                model,
                parsed_context,
                steps=steps,
                lr=lr,
                l0_lambda=l0_lambda,
                beta=beta,
            )
            plan.details.update({"gate_steps": steps, "gate_lr": lr, "l0_lambda": l0_lambda, "gate_beta": beta})
        plan.details["mode"] = "unstructured"
        pruned_model = _apply_unstructured_global(model, plan)
        plan.pruning_time_sec = float(time.perf_counter() - start)
        plan.score_payload = None
        return pruned_model, plan


@register_pruner
class AdaptiveLayerWisePruner(BasePruner):
    """Rule-based adaptive structural pruning with state/action/reward traces."""

    name = "adaptive_layerwise"
    category = "adaptive"
    supports_unstructured = False
    supports_structured = True

    def score(self, model: Any, context: Any, **kwargs: Any) -> PruningPlan:
        context = PruningContext.from_input(context)
        target_sparsity = float(kwargs.get("target_sparsity", context.structure_target or 0.5))
        payload = {"layer_scores": []}
        for layer_index in range(max(0, len(getattr(model, "convs", [])) - 1)):
            scores = _structured_scores_for_layer(model, layer_index=layer_index)
            payload["layer_scores"].append(
                {
                    "layer_index": int(layer_index),
                    "mean_importance": float(scores.mean().item()) if scores.numel() else 0.0,
                    "min_importance": float(scores.min().item()) if scores.numel() else 0.0,
                    "max_importance": float(scores.max().item()) if scores.numel() else 0.0,
                    "num_channels": int(scores.numel()),
                }
            )
        return _build_plan(self.name, self.category, target_sparsity, "structured", payload)

    def apply(self, model: Any, pruning_plan: PruningPlan = None, context: Any = None, **kwargs: Any) -> Any:
        plan, parsed_context = _extract_apply_inputs(pruning_plan, context, kwargs)
        _sync_plan_from_kwargs(plan, kwargs)
        start = time.perf_counter()
        cfg = _adaptive_cfg(parsed_context)

        current_model = copy.deepcopy(model)
        initial_params = _parameter_count(current_model)
        baseline_val = _validation_accuracy(current_model, parsed_context)
        prev_val = baseline_val
        trace = []
        selected_layers = []
        stop_reason = "max_steps_reached"
        final_reward = 0.0

        max_steps = max(1, int(cfg["max_steps"]))
        for step in range(1, max_steps + 1):
            layer_stats = []
            valid_layers = []
            min_selectable_layer = max(selected_layers) if selected_layers else 0
            for layer_index in range(max(0, len(getattr(current_model, "convs", [])) - 1)):
                scores = _structured_scores_for_layer(current_model, layer_index=layer_index)
                width = int(scores.numel())
                mean_importance = float(scores.mean().item()) if width else 0.0
                layer_stats.append(
                    {
                        "layer_index": int(layer_index),
                        "num_channels": width,
                        "mean_importance": mean_importance,
                        "min_importance": float(scores.min().item()) if width else 0.0,
                        "max_importance": float(scores.max().item()) if width else 0.0,
                    }
                )
                if layer_index >= min_selectable_layer and width > int(cfg["min_channels_per_layer"]):
                    valid_layers.append((mean_importance, layer_index, scores))

            if not valid_layers:
                stop_reason = "no_valid_layer"
                break

            _, layer_index, scores = sorted(valid_layers, key=lambda item: item[0])[0]
            width = int(scores.numel())
            min_channels = int(cfg["min_channels_per_layer"])
            prune_count = max(1, int(round(float(cfg["step_prune_ratio"]) * width)))
            keep_count = max(min_channels, width - prune_count)
            keep_count = min(keep_count, width - 1) if width > min_channels else width
            if keep_count < min_channels or keep_count >= width:
                stop_reason = "no_valid_layer"
                break

            _, keep_indices_tensor = torch.topk(scores, k=keep_count, largest=True)
            keep_indices = sorted(int(idx) for idx in keep_indices_tensor.tolist())

            params_before = _parameter_count(current_model)
            sparsity_before = 1.0 - (params_before / max(initial_params, 1))
            val_before = prev_val

            current_model = structurally_prune_hidden_channels(current_model, layer_index=layer_index, keep_indices=keep_indices)

            params_after = _parameter_count(current_model)
            sparsity_after = 1.0 - (params_after / max(initial_params, 1))
            val_after = _validation_accuracy(current_model, parsed_context)
            accuracy_drop = max(0.0, baseline_val - val_after)
            compression_gain = max(0.0, sparsity_after - sparsity_before)
            speedup = 0.0
            reward = (
                float(cfg["alpha"]) * compression_gain
                + float(cfg["beta"]) * speedup
                - float(cfg["gamma"]) * accuracy_drop
            )
            final_reward = float(reward)
            prev_val = val_after
            selected_layers.append(int(layer_index))

            trace.append(
                {
                    "step": int(step),
                    "selected_layer": int(layer_index),
                    "prune_ratio": float(cfg["step_prune_ratio"]),
                    "num_channels_pruned": int(width - keep_count),
                    "sparsity_before": float(sparsity_before),
                    "sparsity_after": float(sparsity_after),
                    "val_accuracy_before": float(val_before),
                    "val_accuracy_after": float(val_after),
                    "parameter_count_before": int(params_before),
                    "parameter_count_after": int(params_after),
                    "state": {
                        "step": int(step),
                        "current_sparsity": float(sparsity_after),
                        "val_accuracy": float(val_after),
                        "parameter_count": int(params_after),
                        "layer_widths": [int(stat["num_channels"]) for stat in layer_stats],
                        "layer_importance_stats": layer_stats,
                    },
                    "action": {
                        "selected_layer": int(layer_index),
                        "prune_ratio": float(cfg["step_prune_ratio"]),
                        "num_channels_pruned": int(width - keep_count),
                    },
                    "reward": {
                        "value": float(reward),
                        "compression_gain": float(compression_gain),
                        "speedup": float(speedup),
                        "accuracy_drop": float(accuracy_drop),
                        "final_reward": float(reward),
                    },
                }
            )

            if sparsity_after >= plan.requested_sparsity:
                stop_reason = "target_sparsity_reached"
                break
            if accuracy_drop > float(cfg["max_accuracy_drop"]):
                stop_reason = "max_accuracy_drop_exceeded"
                break

        if trace:
            trace[-1]["stop_reason"] = stop_reason
            plan.layer_index = int(trace[-1]["selected_layer"])
            plan.target_units = []
            plan.achieved_sparsity = float(trace[-1]["sparsity_after"])
        else:
            params = _parameter_count(current_model)
            plan.achieved_sparsity = float(1.0 - (params / max(initial_params, 1)))

        plan.details.update(
            {
                "mode": "structured",
                "scope": "adaptive_layerwise_structured",
                "adaptive_trace": trace,
                "selected_layer_indices": selected_layers,
                "prunable_channel_groups": max(1, len(getattr(model, "convs", [])) - 1),
                "initial_val_accuracy": float(baseline_val),
                "final_val_accuracy": float(prev_val),
                "initial_parameter_count": int(initial_params),
                "final_parameter_count": int(_parameter_count(current_model)),
                "final_reward": float(final_reward),
                "num_adaptive_steps": int(len(trace)),
                "stop_reason": stop_reason,
            }
        )
        plan.pruning_time_sec = float(time.perf_counter() - start)
        plan.score_payload = None
        return current_model, plan


@register_pruner
class TabularQLearningPruner(BasePruner):
    """Minimal graph-aware tabular Q-learning structural pruner."""

    name = "q_learning_tabular"
    category = "adaptive"
    supports_unstructured = False
    supports_structured = True

    def score(self, model: Any, context: Any, **kwargs: Any) -> PruningPlan:
        parsed = PruningContext.from_input(context)
        target_sparsity = float(kwargs.get("target_sparsity", parsed.structure_target or 0.5))
        plan = _build_plan(self.name, self.category, target_sparsity, "structured", {"algorithm": "tabular_q_learning"})
        plan.details.update({"mode": "structured", "scope": "q_learning_tabular_structured"})
        return plan

    def apply(self, model: Any, pruning_plan: PruningPlan = None, context: Any = None, **kwargs: Any) -> Any:
        plan, parsed_context = _extract_apply_inputs(pruning_plan, context, kwargs)
        _sync_plan_from_kwargs(plan, kwargs)
        start = time.perf_counter()

        if not isinstance(parsed_context.data, dict):
            raise ValueError("q_learning_tabular requires context.data with graph and val_idx.")
        data = parsed_context.data.get("data")
        val_idx = parsed_context.data.get("val_idx")
        if data is None or val_idx is None:
            raise ValueError("q_learning_tabular requires context.data keys: data and val_idx.")

        config = parsed_context.config if isinstance(parsed_context.config, dict) else {}
        q_cfg = config.get("q_learning", {})
        if not isinstance(q_cfg, dict):
            q_cfg = {}

        episodes = int(q_cfg.get("episodes", 20))
        max_steps = int(q_cfg.get("max_steps", 8))
        step_prune_ratios = [float(value) for value in q_cfg.get("step_prune_ratios", [0.05, 0.10])]
        min_channels = int(q_cfg.get("min_channels_per_layer", 4))
        max_accuracy_drop = float(q_cfg.get("max_accuracy_drop", 0.05))
        alpha = float(q_cfg.get("alpha", 0.3))
        gamma = float(q_cfg.get("gamma", 0.9))
        epsilon_start = float(q_cfg.get("epsilon_start", 0.4))
        epsilon_end = float(q_cfg.get("epsilon_end", 0.05))
        epsilon_decay = float(q_cfg.get("epsilon_decay", 0.95))
        reward = q_cfg.get("reward", {}) if isinstance(q_cfg.get("reward", {}), dict) else {}
        reward_alpha = float(reward.get("alpha", 0.4))
        reward_beta = float(reward.get("beta", 0.2))
        reward_gamma = float(reward.get("gamma", 0.4))
        allow_nonmonotonic_layer_order = bool(q_cfg.get("allow_nonmonotonic_layer_order", False))
        structural_pruning_mode = str(q_cfg.get("structural_pruning_mode", "cascade")).strip().lower()
        if structural_pruning_mode not in {"cascade", "local"}:
            structural_pruning_mode = "cascade"
        terminal_cfg = q_cfg.get("terminal_reward", {}) if isinstance(q_cfg.get("terminal_reward", {}), dict) else {}
        buckets = q_cfg.get("state_buckets", {}) if isinstance(q_cfg.get("state_buckets", {}), dict) else {}
        state_buckets = {
            "num_nodes": [1000, 5000, 10000, 20000],
            "num_edges": [5000, 25000, 50000, 100000],
            "avg_degree": [2.0, 5.0, 10.0, 20.0],
            "density": [0.0001, 0.0005, 0.001, 0.005],
            "num_features": [100, 500, 1000, 2000],
            "num_classes": [3, 5, 10, 20],
            "current_sparsity": [0.1, 0.3, 0.5, 0.7, 0.9],
            "target_gap": [0.05, 0.15, 0.30, 0.50],
            "accuracy_drop": [0.01, 0.03, 0.05, 0.1, 0.2],
            "remaining_channels": [0.2, 0.4, 0.6, 0.8],
        }
        for key, default_values in state_buckets.items():
            values = buckets.get(key, default_values)
            if isinstance(values, list) and values:
                state_buckets[key] = [float(v) for v in values]

        num_hidden_layers = max(0, len(getattr(model, "convs", [])) - 1)
        pruning_actions = [
            {"layer_index": int(layer_index), "prune_ratio": float(prune_ratio)}
            for layer_index in range(num_hidden_layers)
            for prune_ratio in step_prune_ratios
        ]
        if not pruning_actions:
            raise ValueError("q_learning_tabular requires at least one hidden layer and one prune ratio action.")

        env = StructuralPruningEnv(
            model=model,
            data=data,
            val_idx=val_idx,
            device=parsed_context.device or "cpu",
            target_sparsity=float(plan.requested_sparsity),
            min_channels_per_layer=min_channels,
            max_steps=max_steps,
            reward_alpha=reward_alpha,
            reward_beta=reward_beta,
            reward_gamma=reward_gamma,
            state_buckets=state_buckets,
            structural_pruning_mode=structural_pruning_mode,
        )
        rng = random.Random(parsed_context.seed)
        q_table: Dict[str, Dict[str, float]] = {}
        training_trace = []
        deployment_trace = []
        epsilon = epsilon_start
        final_reward = 0.0
        training_attempts = 0
        training_valid_steps = 0
        training_invalid_attempts = 0

        # Phase A: Q-table training over reset episodes.
        for episode in range(1, episodes + 1):
            state = env.reset()
            episode_pruned_layers: set[int] = set()
            for step in range(1, max_steps + 1):
                current_sparsity, current_accuracy_drop, current_target_gap = _current_rollout_metrics(env)
                if current_sparsity >= float(plan.requested_sparsity):
                    break
                state_key_value = state_key(state)
                snapshot = _analyze_action_space(
                    model=env.current_model,
                    actions=pruning_actions,
                    min_channels_per_layer=min_channels,
                    pruned_layers=episode_pruned_layers,
                    allow_nonmonotonic_layer_order=allow_nonmonotonic_layer_order,
                    structural_pruning_mode=structural_pruning_mode,
                )
                valid_pruning = [dict(action) for action in snapshot["valid_actions"]]
                stop_allowed = float(current_sparsity) > 0.0
                step_diag = _diagnostic_step_fields(
                    snapshot=snapshot,
                    current_sparsity=float(current_sparsity),
                    target_sparsity=float(plan.requested_sparsity),
                    target_gap=float(current_target_gap),
                    stop_available=bool(stop_allowed),
                )
                if not valid_pruning:
                    terminal_component = _terminal_reward(
                        stop_selected=False,
                        achieved_sparsity=float(current_sparsity),
                        accuracy_drop=float(current_accuracy_drop),
                        target_sparsity=float(plan.requested_sparsity),
                        max_accuracy_drop=max_accuracy_drop,
                        stop_reason="no_valid_action",
                        terminal_cfg=terminal_cfg,
                    )
                    training_trace.append(
                        {
                            "phase": "training",
                            "episode": int(episode),
                            "step": int(step),
                            "attempt": 1,
                            "state": state,
                            "action": {},
                            "reward": float(terminal_component),
                            "next_state": state,
                            "done": True,
                            "selected_from_valid_actions": True,
                            "num_valid_actions": int(snapshot.get("num_valid_actions", 0)),
                            "terminal_reward": float(terminal_component),
                            "stop_selected": False,
                            **step_diag,
                            "info": {
                                "stop_reason": "no_valid_action",
                                "invalid_action_reason": "all_pruning_actions_filtered_out",
                                "current_layer_width": 0,
                                "num_keep": 0,
                                "min_channels_per_layer": int(min_channels),
                                "current_sparsity": float(current_sparsity),
                                "accuracy_drop": float(current_accuracy_drop),
                                "hidden_widths": list(step_diag["hidden_widths_before"]),
                                "total_candidate_actions": int(step_diag["total_candidate_actions"]),
                                "filtered_actions_by_reason": dict(step_diag["filtered_actions_by_reason"]),
                                "filtered_action_examples": list(step_diag["filtered_action_examples"]),
                                "last_pruned_layer": int(step_diag["last_pruned_layer"]),
                                "stop_available": bool(step_diag["stop_available"]),
                                "target_reached": bool(step_diag["target_reached"]),
                            },
                        }
                    )
                    final_reward = float(terminal_component)
                    break

                valid_actions: List[Dict[str, Any]] = [dict(action) for action in valid_pruning]
                if stop_allowed:
                    valid_actions.append(_stop_action())
                selected_action = select_action(
                    state_key_value=state_key_value,
                    actions=valid_actions,
                    q_table=q_table,
                    epsilon=epsilon,
                    rng=rng,
                )
                remaining_actions = [dict(action) for action in valid_actions if action_key(action) != action_key(selected_action)]
                rng.shuffle(remaining_actions)
                action_candidates = [dict(selected_action)] + remaining_actions

                applied = False
                for attempt_idx, action in enumerate(action_candidates, start=1):
                    training_attempts += 1
                    if _is_stop_action(action):
                        terminal_component = _terminal_reward(
                            stop_selected=True,
                            achieved_sparsity=current_sparsity,
                            accuracy_drop=current_accuracy_drop,
                            target_sparsity=float(plan.requested_sparsity),
                            max_accuracy_drop=max_accuracy_drop,
                            stop_reason="agent_stop",
                            terminal_cfg=terminal_cfg,
                        )
                        training_trace.append(
                            {
                                "phase": "training",
                                "episode": int(episode),
                                "step": int(step),
                                "attempt": int(attempt_idx),
                                "state": state,
                                "action": action,
                                "reward": float(terminal_component),
                                "next_state": state,
                                "done": True,
                                "selected_from_valid_actions": True,
                                "num_valid_actions": len(valid_actions),
                                "terminal_reward": float(terminal_component),
                                "stop_selected": True,
                                **step_diag,
                                "info": {
                                    "stop_reason": "agent_stop",
                                    "invalid_action_reason": "",
                                    "current_layer_width": 0,
                                    "num_keep": 0,
                                    "min_channels_per_layer": int(min_channels),
                                    "current_sparsity": float(current_sparsity),
                                    "target_sparsity": float(plan.requested_sparsity),
                                    "target_gap": float(current_target_gap),
                                    "accuracy_drop": float(current_accuracy_drop),
                                    "hidden_widths": list(step_diag["hidden_widths_before"]),
                                },
                            }
                        )
                        update_q(
                            q_table=q_table,
                            state_key_value=state_key_value,
                            action_key_value=action_key(action),
                            reward=float(terminal_component),
                            next_state_key_value=state_key_value,
                            alpha=alpha,
                            gamma=gamma,
                            terminal=True,
                        )
                        final_reward = float(terminal_component)
                        training_valid_steps += 1
                        applied = True
                        break

                    outcome = env.step(action)
                    next_state = outcome.next_state
                    info = dict(outcome.info)
                    invalid_reason = str(info.get("invalid_action_reason", "")).strip()
                    training_trace.append(
                        {
                            "phase": "training",
                            "episode": int(episode),
                            "step": int(step),
                            "attempt": int(attempt_idx),
                            "state": state,
                            "action": action,
                            "reward": float(outcome.reward),
                            "next_state": next_state,
                            "done": bool(outcome.done),
                            "selected_from_valid_actions": True,
                            "num_valid_actions": len(valid_actions),
                            "terminal_reward": 0.0,
                            "stop_selected": False,
                            **step_diag,
                            "info": info,
                        }
                    )
                    if invalid_reason:
                        training_invalid_attempts += 1
                        continue

                    accuracy_drop_value = float(info.get("accuracy_drop", 0.0))
                    done = bool(outcome.done) or accuracy_drop_value > max_accuracy_drop
                    if accuracy_drop_value > max_accuracy_drop:
                        training_trace[-1]["done"] = True
                        training_trace[-1]["info"]["stop_reason"] = "max_accuracy_drop_exceeded"
                    terminal_component = 0.0
                    if done:
                        terminal_component = _terminal_reward(
                            stop_selected=False,
                            achieved_sparsity=float(info.get("current_sparsity", current_sparsity)),
                            accuracy_drop=accuracy_drop_value,
                            target_sparsity=float(plan.requested_sparsity),
                            max_accuracy_drop=max_accuracy_drop,
                            stop_reason=str(training_trace[-1]["info"].get("stop_reason", "")).strip(),
                            terminal_cfg=terminal_cfg,
                        )
                        training_trace[-1]["terminal_reward"] = float(terminal_component)
                    combined_reward = float(outcome.reward) + float(terminal_component)
                    training_trace[-1]["reward"] = float(combined_reward)
                    training_trace[-1]["hidden_widths_after"] = _hidden_widths(env.current_model)

                    update_q(
                        q_table=q_table,
                        state_key_value=state_key_value,
                        action_key_value=action_key(action),
                        reward=float(combined_reward),
                        next_state_key_value=state_key(next_state),
                        alpha=alpha,
                        gamma=gamma,
                        terminal=bool(done),
                    )
                    final_reward = float(combined_reward)
                    training_valid_steps += 1
                    episode_pruned_layers.add(int(action.get("layer_index", -1)))
                    state = next_state
                    applied = True
                    break

                if not applied:
                    terminal_component = _terminal_reward(
                        stop_selected=False,
                        achieved_sparsity=float(current_sparsity),
                        accuracy_drop=float(current_accuracy_drop),
                        target_sparsity=float(plan.requested_sparsity),
                        max_accuracy_drop=max_accuracy_drop,
                        stop_reason="no_valid_action",
                        terminal_cfg=terminal_cfg,
                    )
                    training_trace.append(
                        {
                            "phase": "training",
                            "episode": int(episode),
                            "step": int(step),
                            "attempt": int(len(action_candidates) + 1),
                            "state": state,
                            "action": {},
                            "reward": float(terminal_component),
                            "next_state": state,
                            "done": True,
                            "selected_from_valid_actions": True,
                            "num_valid_actions": len(valid_actions),
                            "terminal_reward": float(terminal_component),
                            "stop_selected": False,
                            **step_diag,
                            "info": {
                                "stop_reason": "no_valid_action",
                                "invalid_action_reason": "all_actions_invalid_for_current_model_state",
                                "current_layer_width": 0,
                                "num_keep": 0,
                                "min_channels_per_layer": int(min_channels),
                                "current_sparsity": float(current_sparsity),
                                "accuracy_drop": float(current_accuracy_drop),
                                "hidden_widths": list(step_diag["hidden_widths_before"]),
                                "total_candidate_actions": int(step_diag["total_candidate_actions"]),
                                "filtered_actions_by_reason": dict(step_diag["filtered_actions_by_reason"]),
                                "filtered_action_examples": list(step_diag["filtered_action_examples"]),
                                "last_pruned_layer": int(step_diag["last_pruned_layer"]),
                                "stop_available": bool(step_diag["stop_available"]),
                                "target_reached": bool(step_diag["target_reached"]),
                            },
                        }
                    )
                    final_reward = float(terminal_component)
                    break

                if training_trace and training_trace[-1].get("done"):
                    break
            epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # Phase B: deterministic deployment rollout from dense reset.
        deployment_state = env.reset()
        deployment_steps = 0
        deployment_invalid_attempts = 0
        deployment_stop_reason = "max_steps_reached"
        deployment_pruned_layers: set[int] = set()
        for step in range(1, max_steps + 1):
            current_sparsity, current_accuracy_drop, current_target_gap = _current_rollout_metrics(env)
            if current_sparsity >= float(plan.requested_sparsity):
                deployment_stop_reason = "target_sparsity_reached"
                break
            state_key_value = state_key(deployment_state)
            snapshot = _analyze_action_space(
                model=env.current_model,
                actions=pruning_actions,
                min_channels_per_layer=min_channels,
                pruned_layers=deployment_pruned_layers,
                allow_nonmonotonic_layer_order=allow_nonmonotonic_layer_order,
                structural_pruning_mode=structural_pruning_mode,
            )
            valid_pruning = [dict(action) for action in snapshot["valid_actions"]]
            stop_allowed = float(current_sparsity) > 0.0
            step_diag = _diagnostic_step_fields(
                snapshot=snapshot,
                current_sparsity=float(current_sparsity),
                target_sparsity=float(plan.requested_sparsity),
                target_gap=float(current_target_gap),
                stop_available=bool(stop_allowed),
            )
            if not valid_pruning:
                deployment_stop_reason = "no_valid_action"
                terminal_component = _terminal_reward(
                    stop_selected=False,
                    achieved_sparsity=float(current_sparsity),
                    accuracy_drop=float(current_accuracy_drop),
                    target_sparsity=float(plan.requested_sparsity),
                    max_accuracy_drop=max_accuracy_drop,
                    stop_reason="no_valid_action",
                    terminal_cfg=terminal_cfg,
                )
                deployment_trace.append(
                    {
                        "phase": "deployment",
                        "step": int(step),
                        "attempt": 1,
                        "state": deployment_state,
                        "action": {},
                        "reward": float(terminal_component),
                        "next_state": deployment_state,
                        "done": True,
                        "selected_from_valid_actions": True,
                        "num_valid_actions": int(snapshot.get("num_valid_actions", 0)),
                        "terminal_reward": float(terminal_component),
                        "stop_selected": False,
                        **step_diag,
                        "info": {
                            "stop_reason": "no_valid_action",
                            "invalid_action_reason": "all_pruning_actions_filtered_out",
                            "current_layer_width": 0,
                            "num_keep": 0,
                            "min_channels_per_layer": int(min_channels),
                            "current_sparsity": float(current_sparsity),
                            "accuracy_drop": float(current_accuracy_drop),
                            "hidden_widths": list(step_diag["hidden_widths_before"]),
                            "total_candidate_actions": int(step_diag["total_candidate_actions"]),
                            "filtered_actions_by_reason": dict(step_diag["filtered_actions_by_reason"]),
                            "filtered_action_examples": list(step_diag["filtered_action_examples"]),
                            "last_pruned_layer": int(step_diag["last_pruned_layer"]),
                            "stop_available": bool(step_diag["stop_available"]),
                            "target_reached": bool(step_diag["target_reached"]),
                        },
                    }
                )
                final_reward = float(terminal_component)
                break

            valid_actions: List[Dict[str, Any]] = sorted((dict(action) for action in valid_pruning), key=action_key)
            if stop_allowed:
                valid_actions.append(_stop_action())
            selected_action = select_action(
                state_key_value=state_key_value,
                actions=valid_actions,
                q_table=q_table,
                epsilon=0.0,
                rng=rng,
            )
            remaining_actions = [dict(action) for action in valid_actions if action_key(action) != action_key(selected_action)]
            action_candidates = [dict(selected_action)] + remaining_actions

            applied = False
            for attempt_idx, action in enumerate(action_candidates, start=1):
                if _is_stop_action(action):
                    terminal_component = _terminal_reward(
                        stop_selected=True,
                        achieved_sparsity=current_sparsity,
                        accuracy_drop=current_accuracy_drop,
                        target_sparsity=float(plan.requested_sparsity),
                        max_accuracy_drop=max_accuracy_drop,
                        stop_reason="agent_stop",
                        terminal_cfg=terminal_cfg,
                    )
                    deployment_trace.append(
                        {
                            "phase": "deployment",
                            "step": int(step),
                            "attempt": int(attempt_idx),
                            "state": deployment_state,
                            "action": action,
                            "reward": float(terminal_component),
                            "next_state": deployment_state,
                            "done": True,
                            "selected_from_valid_actions": True,
                            "num_valid_actions": len(valid_actions),
                            "terminal_reward": float(terminal_component),
                            "stop_selected": True,
                            **step_diag,
                            "info": {
                                "stop_reason": "agent_stop",
                                "invalid_action_reason": "",
                                "current_layer_width": 0,
                                "num_keep": 0,
                                "min_channels_per_layer": int(min_channels),
                                "current_sparsity": float(current_sparsity),
                                "target_sparsity": float(plan.requested_sparsity),
                                "target_gap": float(current_target_gap),
                                "accuracy_drop": float(current_accuracy_drop),
                                "hidden_widths": list(step_diag["hidden_widths_before"]),
                            },
                        }
                    )
                    deployment_stop_reason = "agent_stop"
                    final_reward = float(terminal_component)
                    applied = True
                    break

                layer_idx = int(action.get("layer_index", -1))
                outcome = env.step(action)
                next_state = outcome.next_state
                info = dict(outcome.info)
                invalid_reason = str(info.get("invalid_action_reason", "")).strip()
                deployment_trace.append(
                    {
                        "phase": "deployment",
                        "step": int(step),
                        "attempt": int(attempt_idx),
                        "state": deployment_state,
                        "action": action,
                        "reward": float(outcome.reward),
                        "next_state": next_state,
                        "done": bool(outcome.done),
                        "selected_from_valid_actions": True,
                        "num_valid_actions": len(valid_actions),
                        "terminal_reward": 0.0,
                        "stop_selected": False,
                        **step_diag,
                        "info": info,
                    }
                )
                if invalid_reason:
                    deployment_invalid_attempts += 1
                    continue

                deployment_steps += 1
                deployment_state = next_state
                deployment_pruned_layers.add(layer_idx)
                accuracy_drop_value = float(info.get("accuracy_drop", 0.0))
                done = bool(outcome.done) or accuracy_drop_value > max_accuracy_drop
                if accuracy_drop_value > max_accuracy_drop:
                    deployment_trace[-1]["done"] = True
                    deployment_trace[-1]["info"]["stop_reason"] = "max_accuracy_drop_exceeded"
                    deployment_stop_reason = "max_accuracy_drop_exceeded"
                else:
                    deployment_stop_reason = str(info.get("stop_reason", deployment_stop_reason)) or deployment_stop_reason
                terminal_component = 0.0
                if done:
                    terminal_component = _terminal_reward(
                        stop_selected=False,
                        achieved_sparsity=float(info.get("current_sparsity", current_sparsity)),
                        accuracy_drop=accuracy_drop_value,
                        target_sparsity=float(plan.requested_sparsity),
                        max_accuracy_drop=max_accuracy_drop,
                        stop_reason=str(deployment_trace[-1]["info"].get("stop_reason", "")).strip(),
                        terminal_cfg=terminal_cfg,
                    )
                    deployment_trace[-1]["terminal_reward"] = float(terminal_component)
                final_reward = float(outcome.reward) + float(terminal_component)
                deployment_trace[-1]["reward"] = float(final_reward)
                deployment_trace[-1]["hidden_widths_after"] = _hidden_widths(env.current_model)
                applied = True
                break

            if not applied:
                deployment_stop_reason = "no_valid_action"
                terminal_component = _terminal_reward(
                    stop_selected=False,
                    achieved_sparsity=float(current_sparsity),
                    accuracy_drop=float(current_accuracy_drop),
                    target_sparsity=float(plan.requested_sparsity),
                    max_accuracy_drop=max_accuracy_drop,
                    stop_reason="no_valid_action",
                    terminal_cfg=terminal_cfg,
                )
                deployment_trace.append(
                    {
                        "phase": "deployment",
                        "step": int(step),
                        "attempt": int(len(action_candidates) + 1),
                        "state": deployment_state,
                        "action": {},
                        "reward": float(terminal_component),
                        "next_state": deployment_state,
                        "done": True,
                        "selected_from_valid_actions": True,
                        "num_valid_actions": len(valid_actions),
                        "terminal_reward": float(terminal_component),
                        "stop_selected": False,
                        **step_diag,
                        "info": {
                            "stop_reason": "no_valid_action",
                            "invalid_action_reason": "all_actions_invalid_for_current_model_state",
                            "current_layer_width": 0,
                            "num_keep": 0,
                            "min_channels_per_layer": int(min_channels),
                            "current_sparsity": float(current_sparsity),
                            "accuracy_drop": float(current_accuracy_drop),
                            "hidden_widths": list(step_diag["hidden_widths_before"]),
                            "total_candidate_actions": int(step_diag["total_candidate_actions"]),
                            "filtered_actions_by_reason": dict(step_diag["filtered_actions_by_reason"]),
                            "filtered_action_examples": list(step_diag["filtered_action_examples"]),
                            "last_pruned_layer": int(step_diag["last_pruned_layer"]),
                            "stop_available": bool(step_diag["stop_available"]),
                            "target_reached": bool(step_diag["target_reached"]),
                        },
                    }
                )
                final_reward = float(terminal_component)
                break

            if deployment_trace and deployment_trace[-1].get("done"):
                break

        _enrich_trace_entries(training_trace, target_sparsity=float(plan.requested_sparsity), terminal_cfg=terminal_cfg)
        _enrich_trace_entries(deployment_trace, target_sparsity=float(plan.requested_sparsity), terminal_cfg=terminal_cfg)

        output_dir_value = config.get("output_dir")
        output_dir = Path(str(output_dir_value)).expanduser() if output_dir_value else None
        q_table_path = None
        rl_trace_path = None
        deployment_trace_path = None
        action_space_diagnostics_path = None
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            q_table_path = output_dir / "q_table.json"
            rl_trace_path = output_dir / "rl_trace.json"
            deployment_trace_path = output_dir / "deployment_trace.json"
            with q_table_path.open("w", encoding="utf-8") as handle:
                json.dump(q_table, handle, indent=2)
            with rl_trace_path.open("w", encoding="utf-8") as handle:
                json.dump(training_trace + deployment_trace, handle, indent=2)
            with deployment_trace_path.open("w", encoding="utf-8") as handle:
                json.dump(deployment_trace, handle, indent=2)

        # Return deployment rollout model, not last training episode model.
        final_model = copy.deepcopy(env.current_model)
        plan.achieved_sparsity = float(1.0 - (_parameter_count(final_model) / max(1, _parameter_count(model))))
        if output_dir is not None:
            training_no_valid = [entry for entry in training_trace if str(entry.get("stop_reason", "")) == "no_valid_action"]
            deployment_no_valid = [entry for entry in deployment_trace if str(entry.get("stop_reason", "")) == "no_valid_action"]
            deployment_filter_reasons = _aggregate_filter_reasons(deployment_trace)
            dominant_reason = _dominant_reason(deployment_filter_reasons)
            action_space_diagnostics = {
                "training": {
                    "num_steps": int(len(training_trace)),
                    "num_no_valid_action_terminations": int(len(training_no_valid)),
                    "most_common_filter_reasons": _aggregate_filter_reasons(training_trace),
                },
                "deployment": {
                    "num_steps": int(len(deployment_trace)),
                    "num_no_valid_action_terminations": int(len(deployment_no_valid)),
                    "most_common_filter_reasons": deployment_filter_reasons,
                    "final_hidden_widths": _hidden_widths(final_model),
                    "final_achieved_sparsity": float(plan.achieved_sparsity),
                    "final_target_gap": float(_target_gap(target_sparsity=float(plan.requested_sparsity), current_sparsity=float(plan.achieved_sparsity))),
                    "final_stop_reason": str(deployment_stop_reason),
                    "invalid_layer_order_filtered_count": int(deployment_filter_reasons.get("invalid_layer_order", 0)),
                    "min_channels_filtered_count": int(deployment_filter_reasons.get("min_channels_per_layer", 0)),
                    "dominant_filter_reason": dominant_reason,
                },
            }
            action_space_diagnostics_path = output_dir / "action_space_diagnostics.json"
            with action_space_diagnostics_path.open("w", encoding="utf-8") as handle:
                json.dump(action_space_diagnostics, handle, indent=2)
        plan.details.update(
            {
                "mode": "structured",
                "scope": "q_learning_tabular_structured",
                "adaptive_trace": training_trace + deployment_trace,
                "final_reward": float(final_reward),
                "num_adaptive_steps": int(deployment_steps),
                "stop_reason": deployment_stop_reason,
                "q_table_path": str(q_table_path) if q_table_path is not None else "",
                "rl_trace_path": str(rl_trace_path) if rl_trace_path is not None else "",
                "deployment_trace_path": str(deployment_trace_path) if deployment_trace_path is not None else "",
                "action_space_diagnostics_path": str(action_space_diagnostics_path) if action_space_diagnostics_path is not None else "",
                "episodes": int(episodes),
                "max_steps": int(max_steps),
                "training_attempts": int(training_attempts),
                "training_valid_steps": int(training_valid_steps),
                "training_invalid_attempts": int(training_invalid_attempts),
                "deployment_steps": int(deployment_steps),
                "deployment_invalid_attempts": int(deployment_invalid_attempts),
            }
        )
        plan.pruning_time_sec = float(time.perf_counter() - start)
        plan.score_payload = None
        return final_model, plan

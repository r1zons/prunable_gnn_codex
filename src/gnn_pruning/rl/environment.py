"""Minimal RL environment for structural pruning."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict

import torch

from gnn_pruning.rl.state import build_bucketed_state
from gnn_pruning.surgery import (
    can_apply_structural_prune,
    structurally_prune_hidden_channels,
    structurally_prune_hidden_channels_local,
)


@dataclass
class StepOutcome:
    """Container for one RL step transition."""

    next_state: Dict[str, int]
    reward: float
    done: bool
    info: Dict[str, Any]


class StructuralPruningEnv:
    """Environment wrapping structural channel pruning transitions."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        data: Any,
        val_idx: torch.Tensor,
        device: str,
        target_sparsity: float,
        min_channels_per_layer: int,
        max_steps: int,
        reward_alpha: float,
        reward_beta: float,
        reward_gamma: float,
        state_buckets: Dict[str, list[float]],
        structural_pruning_mode: str = "cascade",
    ) -> None:
        self.initial_model = copy.deepcopy(model)
        self.current_model = copy.deepcopy(model)
        self.data = data
        self.val_idx = val_idx
        self.device = torch.device(device)
        self.target_sparsity = float(target_sparsity)
        self.min_channels_per_layer = int(min_channels_per_layer)
        self.max_steps = int(max_steps)
        self.reward_alpha = float(reward_alpha)
        self.reward_beta = float(reward_beta)
        self.reward_gamma = float(reward_gamma)
        self.state_buckets = state_buckets
        self.structural_pruning_mode = str(structural_pruning_mode).strip().lower()
        if self.structural_pruning_mode not in {"cascade", "local"}:
            self.structural_pruning_mode = "cascade"
        self.step_count = 0
        self.initial_param_count = _parameter_count(self.initial_model)
        self.baseline_val_acc = self._validation_accuracy(self.initial_model)

    def reset(self) -> Dict[str, int]:
        """Reset episode and return initial state."""
        self.current_model = copy.deepcopy(self.initial_model)
        self.step_count = 0
        current_sparsity = _current_sparsity(self.initial_param_count, _parameter_count(self.current_model))
        return build_bucketed_state(
            data=self.data,
            current_sparsity=current_sparsity,
            target_sparsity=self.target_sparsity,
            accuracy_drop=0.0,
            remaining_channels=_remaining_channels(self.current_model),
            total_channels=max(1, _remaining_channels(self.initial_model)),
            buckets=self.state_buckets,
        )

    def step(self, action: Dict[str, float]) -> StepOutcome:
        """Apply structural action and return transition."""
        self.step_count += 1
        layer_index = int(action["layer_index"])
        prune_ratio = float(action["prune_ratio"])
        before_params = _parameter_count(self.current_model)
        current_sparsity_before = _current_sparsity(self.initial_param_count, before_params)
        current_val = self._validation_accuracy(self.current_model)
        next_state = self._state_from_model(self.current_model, current_val)

        current_hidden_layers = max(0, len(getattr(self.current_model, "convs", [])) - 1)
        if layer_index < 0 or layer_index >= current_hidden_layers:
            return StepOutcome(
                next_state=next_state,
                reward=0.0,
                done=False,
                info={
                    "layer_index": layer_index,
                    "prune_ratio": prune_ratio,
                    "current_layer_width": 0,
                    "num_keep": 0,
                    "min_channels_per_layer": int(self.min_channels_per_layer),
                    "invalid_action_reason": "layer_index_out_of_range",
                    "stop_reason": "",
                },
            )

        width = _current_layer_width(self.current_model, layer_index=layer_index)
        if width <= 0:
            return StepOutcome(
                next_state=next_state,
                reward=0.0,
                done=False,
                info={
                    "layer_index": layer_index,
                    "prune_ratio": prune_ratio,
                    "current_layer_width": int(width),
                    "num_keep": 0,
                    "min_channels_per_layer": int(self.min_channels_per_layer),
                    "invalid_action_reason": "non_positive_layer_width",
                    "stop_reason": "",
                },
            )

        scores = _layer_scores(self.current_model, layer_index=layer_index)
        if int(scores.numel()) != int(width):
            return StepOutcome(
                next_state=next_state,
                reward=0.0,
                done=False,
                info={
                    "layer_index": layer_index,
                    "prune_ratio": prune_ratio,
                    "current_layer_width": int(width),
                    "num_keep": 0,
                    "min_channels_per_layer": int(self.min_channels_per_layer),
                    "invalid_action_reason": "score_width_mismatch",
                    "stop_reason": "",
                },
            )
        prune_count = max(1, int(round(prune_ratio * width)))
        keep_count = max(self.min_channels_per_layer, width - prune_count)
        keep_count = min(keep_count, width - 1) if width > self.min_channels_per_layer else width
        if keep_count >= width or keep_count < self.min_channels_per_layer:
            return StepOutcome(
                next_state=next_state,
                reward=0.0,
                done=False,
                info={
                    "layer_index": layer_index,
                    "prune_ratio": prune_ratio,
                    "current_layer_width": int(width),
                    "num_keep": int(keep_count),
                    "min_channels_per_layer": int(self.min_channels_per_layer),
                    "invalid_action_reason": "keep_count_violates_constraints",
                    "stop_reason": "",
                },
            )

        _, keep_idx_tensor = torch.topk(scores, k=keep_count, largest=True)
        keep_indices = sorted(int(idx) for idx in keep_idx_tensor.tolist())
        if not keep_indices:
            return StepOutcome(
                next_state=next_state,
                reward=0.0,
                done=False,
                info={
                    "layer_index": layer_index,
                    "prune_ratio": prune_ratio,
                    "current_layer_width": int(width),
                    "num_keep": 0,
                    "min_channels_per_layer": int(self.min_channels_per_layer),
                    "invalid_action_reason": "empty_keep_indices",
                    "stop_reason": "",
                },
            )
        if min(keep_indices) < 0 or max(keep_indices) >= width:
            return StepOutcome(
                next_state=next_state,
                reward=0.0,
                done=False,
                info={
                    "layer_index": layer_index,
                    "prune_ratio": prune_ratio,
                    "current_layer_width": int(width),
                    "num_keep": int(len(keep_indices)),
                    "min_channels_per_layer": int(self.min_channels_per_layer),
                    "invalid_action_reason": "keep_indices_out_of_bounds",
                    "stop_reason": "",
                },
            )

        feasibility = can_apply_structural_prune(
            self.current_model,
            layer_index=layer_index,
            keep_indices=keep_indices,
            min_channels_per_layer=int(self.min_channels_per_layer),
            mode=self.structural_pruning_mode,
        )
        if not feasibility.valid:
            return StepOutcome(
                next_state=next_state,
                reward=0.0,
                done=False,
                info={
                    "layer_index": layer_index,
                    "prune_ratio": prune_ratio,
                    "current_layer_width": int(width),
                    "num_keep": int(len(keep_indices)),
                    "min_channels_per_layer": int(self.min_channels_per_layer),
                    "invalid_action_reason": f"structural_feasibility_failed:{feasibility.reason}",
                    "stop_reason": "",
                },
            )

        try:
            if self.structural_pruning_mode == "local":
                self.current_model = structurally_prune_hidden_channels_local(
                    self.current_model,
                    layer_index=layer_index,
                    keep_indices=keep_indices,
                )
            else:
                self.current_model = structurally_prune_hidden_channels(
                    self.current_model,
                    layer_index=layer_index,
                    keep_indices=keep_indices,
                )
        except Exception as exc:
            return StepOutcome(
                next_state=next_state,
                reward=0.0,
                done=False,
                info={
                    "layer_index": layer_index,
                    "prune_ratio": prune_ratio,
                    "current_layer_width": int(width),
                    "num_keep": int(len(keep_indices)),
                    "min_channels_per_layer": int(self.min_channels_per_layer),
                    "invalid_action_reason": f"surgery_failed:{exc.__class__.__name__}",
                    "stop_reason": "",
                },
            )

        after_params = _parameter_count(self.current_model)
        current_sparsity_after = _current_sparsity(self.initial_param_count, after_params)
        val_after = self._validation_accuracy(self.current_model)
        accuracy_drop = max(0.0, self.baseline_val_acc - val_after)
        compression_gain = max(0.0, current_sparsity_after - current_sparsity_before)
        speed_proxy = compression_gain
        reward = (
            self.reward_alpha * compression_gain
            + self.reward_beta * speed_proxy
            - self.reward_gamma * accuracy_drop
        )

        done = current_sparsity_after >= self.target_sparsity or self.step_count >= self.max_steps
        stop_reason = "target_sparsity_reached" if current_sparsity_after >= self.target_sparsity else "max_steps_reached"
        next_state = self._state_from_model(self.current_model, val_after)
        return StepOutcome(
            next_state=next_state,
            reward=float(reward),
            done=bool(done),
            info={
                "layer_index": layer_index,
                "prune_ratio": prune_ratio,
                "current_layer_width": int(width),
                "num_keep": int(len(keep_indices)),
                "min_channels_per_layer": int(self.min_channels_per_layer),
                "invalid_action_reason": "",
                "compression_gain": float(compression_gain),
                "speed_proxy": float(speed_proxy),
                "accuracy_drop": float(accuracy_drop),
                "current_sparsity": float(current_sparsity_after),
                "stop_reason": stop_reason if done else "",
            },
        )

    def _state_from_model(self, model: torch.nn.Module, val_acc: float) -> Dict[str, int]:
        params = _parameter_count(model)
        current_sparsity = _current_sparsity(self.initial_param_count, params)
        accuracy_drop = max(0.0, self.baseline_val_acc - val_acc)
        return build_bucketed_state(
            data=self.data,
            current_sparsity=current_sparsity,
            target_sparsity=self.target_sparsity,
            accuracy_drop=accuracy_drop,
            remaining_channels=_remaining_channels(model),
            total_channels=max(1, _remaining_channels(self.initial_model)),
            buckets=self.state_buckets,
        )

    def _validation_accuracy(self, model: torch.nn.Module) -> float:
        model = model.to(self.device)
        graph = self.data.to(self.device)
        val_idx = self.val_idx.to(self.device, dtype=torch.long)
        model.eval()
        with torch.no_grad():
            logits = model(graph)
            pred = logits.argmax(dim=-1)
            correct = (pred[val_idx] == graph.y[val_idx]).float().mean()
        return float(correct.item())


def _parameter_count(model: torch.nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def _current_sparsity(initial_params: int, current_params: int) -> float:
    return float(1.0 - (current_params / max(1, initial_params)))


def _remaining_channels(model: torch.nn.Module) -> int:
    total = 0
    if hasattr(model, "convs"):
        for conv in model.convs[:-1]:
            if hasattr(conv, "out_channels"):
                total += int(conv.out_channels)
    return int(total)


def _layer_scores(model: Any, layer_index: int) -> torch.Tensor:
    conv = model.convs[layer_index]
    next_conv = model.convs[layer_index + 1]
    if hasattr(conv, "lin") and hasattr(next_conv, "lin"):
        return conv.lin.weight.abs().sum(dim=1) + next_conv.lin.weight.abs().sum(dim=0)
    left = conv.lin_l.weight.abs().sum(dim=1)
    if getattr(conv, "root_weight", False):
        left = left + conv.lin_r.weight.abs().sum(dim=1)
    right = next_conv.lin_l.weight.abs().sum(dim=0)
    if getattr(next_conv, "root_weight", False):
        right = right + next_conv.lin_r.weight.abs().sum(dim=0)
    return left + right


def _current_layer_width(model: Any, layer_index: int) -> int:
    conv = model.convs[layer_index]
    if hasattr(conv, "out_channels"):
        return int(conv.out_channels)
    raise ValueError("Layer does not expose out_channels.")

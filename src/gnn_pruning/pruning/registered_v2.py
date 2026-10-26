"""Separate V2 registry entries; historical V1 implementations are unchanged."""
from __future__ import annotations

from .base import BasePruner, PruningContext
from .registry import register_pruner
from .static_v2 import compute_scores, plan_static, apply_static_plan
from .adaptive_v2 import run_greedy, run_q_learning


class LocalV2Pruner(BasePruner):
    """Explicit score/plan/apply protocol used by the dedicated V2 workflow."""
    supports_structured = True
    category = "local_structured_v2"
    criterion = None

    def score(self, model, context, **kwargs):
        """Compute static saliency once on dense; adaptive saliency is state-local."""
        context = PruningContext.from_input(context)
        if self.criterion is None:
            return {}
        data = context.data or {}
        return compute_scores(model, self.criterion, data.get("data"), data.get("train_idx"))

    def plan(self, model, scores, context, target_sparsity):
        """Search a complete decision without mutating the source model."""
        context = PruningContext.from_input(context)
        cfg = context.config.get("v2", {})
        common = {"tolerance": cfg.get("target_sparsity_tolerance", 0.01),
                  "min_channels": cfg.get("min_channels_per_layer", 4)}
        if common["min_channels"] < 4:
            raise ValueError("V2 research protocol requires min_channels_per_layer>=4.")
        if self.criterion is not None:
            return plan_static(model, scores, self.criterion, target_sparsity,
                               seed=context.seed, **common)
        if "penalty_lambda" not in cfg:
            raise ValueError("V2 adaptive methods require explicit penalty_lambda.")
        common.update(target=target_sparsity, penalty_lambda=cfg["penalty_lambda"],
                      max_steps=cfg.get("max_steps", 20))
        evaluate = context.data["evaluate_validation"]
        if self.name == "adaptive_greedy_local_v2":
            return run_greedy(model, evaluate, **common)
        return run_q_learning(model, evaluate, seed=context.seed, episodes=cfg.get("episodes", 30),
                              epsilon=cfg.get("epsilon", 0.2), alpha=cfg.get("alpha", 0.1),
                              gamma=cfg.get("gamma", 0.9), **common)

    def apply(self, model, pruning_plan, context, **kwargs):
        """Physically materialize the static plan, or return the actual adaptive final."""
        if self.criterion is not None:
            return apply_static_plan(model, pruning_plan)
        return pruning_plan.model


@register_pruner
class RandomLocalV2Pruner(LocalV2Pruner):
    """Uniform width-pair/subset baseline without an importance oracle."""
    name, criterion = "random_local_v2", "random"


@register_pruner
class MagnitudeLocalV2Pruner(LocalV2Pruner):
    """Dense fixed-rank absolute-parameter static search."""
    name, criterion = "magnitude_local_v2", "magnitude"


@register_pruner
class SNIPLocalV2Pruner(LocalV2Pruner):
    """Post-training train-node abs(w*g) structural adaptation."""
    name, criterion = "snip_pt_local_v2", "snip"


@register_pruner
class GraSPLocalV2Pruner(LocalV2Pruner):
    """Conditional post-training signed -w(Hg) structural adaptation."""
    name, criterion = "grasp_pt_local_v2", "grasp"


@register_pruner
class AdaptiveGreedyLocalV2Pruner(LocalV2Pruner):
    """Validation one-step lookahead with integer Local Surgery actions."""
    name = "adaptive_greedy_local_v2"


@register_pruner
class QLearningLocalV2Pruner(LocalV2Pruner):
    """Aggregated-state tabular policy with explicit post-removal STOP."""
    name = "q_learning_local_v2"

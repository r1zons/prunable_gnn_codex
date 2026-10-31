"""Opt-in frozen-checkpoint Lookahead adapter; historical pruners are unchanged."""
from __future__ import annotations

from typing import Any
from torch import nn

from .base import PruningContext
from .registered_v2 import LocalV2Pruner
from .registry import register_pruner
from .lookahead import LookaheadResult, run_lookahead


@register_pruner
class LookaheadLocalV4Pruner(LocalV2Pruner):
    """Native Greedy-threshold Lookahead; registry name retained for compatibility."""

    name = 'lookahead_local_v4'
    category = 'local_structured_lookahead'

    def plan(self, model: nn.Module, scores: Any, context: Any,
             target_sparsity: float) -> LookaheadResult:
        """Use only the supplied validation callback and explicit shared settings."""
        context = PruningContext.from_input(context)
        cfg = context.config.get('v2', {})
        planning = context.config.get('lookahead', {})
        if set(planning)-{'depth', 'gamma'}:
            raise ValueError('Unknown Lookahead settings; only depth and gamma are supported.')
        if 'penalty_lambda' not in cfg:
            raise ValueError('Lookahead requires explicit v2.penalty_lambda for comparison.')
        return run_lookahead(model, context.data['evaluate_validation'], target_sparsity,
            depth=planning.get('depth', 2), gamma=planning.get('gamma', cfg.get('gamma', .9)),
            penalty_lambda=cfg['penalty_lambda'], max_steps=cfg.get('max_steps', 20),
            min_channels=cfg.get('min_channels_per_layer', 4),
            tolerance=cfg.get('target_sparsity_tolerance', .01))

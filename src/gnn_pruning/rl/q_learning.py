"""Minimal tabular Q-learning utilities."""

from __future__ import annotations

import random
from typing import Any, Dict, List


def action_key(action: Dict[str, Any]) -> str:
    """Serialize action dict to stable key."""
    if str(action.get("type", "")).strip().lower() == "stop":
        return "stop"
    layer = int(action["layer_index"])
    ratio = float(action["prune_ratio"])
    return f"layer:{layer}|ratio:{ratio:.6f}"


def select_action(
    *,
    state_key_value: str,
    actions: List[Dict[str, Any]],
    q_table: Dict[str, Dict[str, float]],
    epsilon: float,
    rng: random.Random,
) -> Dict[str, Any]:
    """Epsilon-greedy action selection."""
    if rng.random() < float(epsilon):
        return dict(rng.choice(actions))
    action_values = q_table.get(state_key_value, {})
    ranked = sorted(actions, key=action_key)
    best = max(ranked, key=lambda action: float(action_values.get(action_key(action), 0.0)))
    return dict(best)


def update_q(
    *,
    q_table: Dict[str, Dict[str, float]],
    state_key_value: str,
    action_key_value: str,
    reward: float,
    next_state_key_value: str,
    alpha: float,
    gamma: float,
    terminal: bool = False,
) -> None:
    """Apply one tabular Q-learning update."""
    state_row = q_table.setdefault(state_key_value, {})
    old_q = float(state_row.get(action_key_value, 0.0))
    next_max = 0.0
    if not terminal:
        next_row = q_table.get(next_state_key_value, {})
        next_max = max((float(value) for value in next_row.values()), default=0.0)
    target = float(reward) + float(gamma) * next_max
    state_row[action_key_value] = old_q + float(alpha) * (target - old_q)

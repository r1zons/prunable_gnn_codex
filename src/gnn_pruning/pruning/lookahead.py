"""Deterministic exhaustive depth-1/2 planning on the unchanged V4 environment.

Only a selected first-action preview is adopted. The following plan can reuse its
already evaluated children by concrete parent identity, never by aggregate state.
Evaluators must be deterministic validation-only functions of the frozen model.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from torch import nn

from .adaptive_v2 import AdaptiveResult, action_key
from .experimental_v4 import Settings, V4Env

Action = tuple[int, int]


@dataclass
class LookaheadPlan:
    """First-action successor, serializable search tree and actual preview costs."""

    next_env: V4Env
    decision: dict[str, Any]
    evaluations: int
    explored_transitions: int
    cache_hits: int
    scoring_seconds: float
    surgery_seconds: float


@dataclass
class LookaheadResult(AdaptiveResult):
    """Physical final model plus independent deployment and planning traces."""

    decisions: list[dict[str, Any]] = field(default_factory=list)


class LookaheadPlanner:
    """Exhaustive short-horizon planner; exact ties retain legal enumeration order."""

    def __init__(self, depth: int = 2, gamma: float = .9) -> None:
        if type(depth) is not int or depth not in (1, 2):
            raise ValueError('Lookahead depth must be integer 1 or 2.')
        if gamma != .9:
            raise ValueError('Lookahead requires gamma=0.9 for V4 comparability.')
        self.depth, self.gamma = depth, gamma
        self._parent: V4Env | None = None
        self._children: dict[Action, V4Env] = {}
        self._parent_state: tuple | None = None

    @staticmethod
    def _concrete_state(env: V4Env) -> tuple:
        # Identity alone is insufficient if a caller advances the same environment.
        # Model weights and evaluator inputs must still obey the frozen contract.
        return (env.state(), tuple(map(tuple, env.original_kept_ids)),
                id(getattr(env, 'model', None)),
                tuple(getattr(env, name, None) for name in
                      ('accuracy', 'dense_accuracy', 'sparsity', 'target', 'tolerance',
                       'penalty_lambda', 'min_channels', 'p0')))

    def plan(self, env: V4Env) -> LookaheadPlan:
        """Score all feasible leaves; leave the parent model/state/trace unchanged.

        Nonterminal branches must extend to the requested depth. Terminal branches
        end immediately; cutoff leaves receive neither bootstrap nor fake terminal
        penalties. A cached child is valid only for this exact parent object.
        """
        if not isinstance(env, V4Env) or env.settings.menu != 'standard' or env.settings.incremental_drop or env.greedy:
            raise ValueError('Lookahead requires the standard repeated-drop V4 environment.')
        if env.terminal:
            raise ValueError('Cannot plan from a terminal environment.')
        started = time.perf_counter()
        actions = env.legal_actions()
        reusable = (self._children if self._parent is env and
                    self._parent_state == self._concrete_state(env) else {})
        first: dict[Action, V4Env] = {}
        seconds: dict[Action, dict[Action, V4Env]] = {}
        transitions, paths = [], []
        evaluations = hits = 0
        scoring = surgery = 0.

        def preview(parent: V4Env, action: Action, path: list[str], cached: V4Env | None = None) -> V4Env:
            nonlocal evaluations, hits, scoring, surgery
            child = cached if cached is not None else parent.preview(action)
            if cached is None:
                evaluations += 1
                scoring += child.scoring_seconds-parent.scoring_seconds
                surgery += child.surgery_seconds-parent.surgery_seconds
            else:
                hits += 1
            transitions.append(dict(path=path, from_state=parent.state(), state=child.state(),
                                    evaluation_reused=cached is not None,
                                    transition=copy.deepcopy(child.trace[-1])))
            return child

        # Depth 1 uses exactly the historical Greedy preview enumeration order.
        for action in actions:
            first[action] = preview(env, action, [action_key(action)], reusable.get(action))
        for action, child in first.items():
            first_key, reward = action_key(action), child.trace[-1]['reward']
            seconds[action] = {}
            if self.depth == 1 or child.terminal:
                paths.append(dict(actions=[first_key], rewards=[reward], score=reward,
                                  terminal=child.terminal, terminal_reason=child.terminal_reason))
                continue
            for second_action in child.legal_actions():
                keys = [first_key, action_key(second_action)]
                leaf = preview(child, second_action, keys)
                seconds[action][second_action] = leaf
                next_reward = leaf.trace[-1]['reward']
                paths.append(dict(actions=keys, rewards=[reward, next_reward],
                                  score=reward+self.gamma*next_reward,
                                  terminal=leaf.terminal, terminal_reason=leaf.terminal_reason))
        # max is stable: first root action, then first continuation on exact ties.
        best = max(paths, key=lambda path: path['score'])
        selected = next(action for action in actions if action_key(action) == best['actions'][0])
        chosen = first[selected]
        self._parent, self._children = chosen, seconds[selected]
        self._parent_state = self._concrete_state(chosen)
        decision = dict(step=env.steps+1, state=env.state(),
                        original_kept_ids=copy.deepcopy(env.original_kept_ids),
                        legal_actions=[action_key(a) for a in actions],
                        selected_action=action_key(selected), best_planned_path=best['actions'],
                        selected_score=best['score'], candidate_paths=paths,
                        candidate_transitions=transitions, candidate_evaluations=evaluations,
                        explored_transitions=len(transitions), reused_evaluations=hits,
                        planning_seconds=time.perf_counter()-started)
        return LookaheadPlan(chosen, decision, evaluations, len(transitions), hits, scoring, surgery)


def run_lookahead(model: nn.Module, evaluate: Callable[[nn.Module], float], target: float,
                  depth: int = 2, gamma: float = .9, penalty_lambda: float = 2.,
                  max_steps: int = 20, min_channels: int = 4,
                  tolerance: float = .01) -> LookaheadResult:
    """Run receding-horizon search without training or mutating the dense input.

    Search cost includes dense validation and all previews. Candidate evaluations
    exclude dense validation, reused previews and adopting an evaluated successor.
    A deterministic evaluator is required; stochastic callbacks invalidate reuse.
    """
    planner = LookaheadPlanner(depth, gamma)
    started = time.perf_counter()
    env = V4Env(model, evaluate, target, Settings(), penalty_lambda=penalty_lambda,
                max_steps=max_steps, min_channels=min_channels, tolerance=tolerance)
    evaluations, explored, hits, leaf_paths = 0, 0, 0, 0
    scoring = surgery = 0.
    decisions = []
    while not env.terminal:
        plan = planner.plan(env)
        decisions.append(plan.decision)
        evaluations += plan.evaluations
        explored += plan.explored_transitions
        hits += plan.cache_hits
        leaf_paths += len(plan.decision['candidate_paths'])
        scoring += plan.scoring_seconds
        surgery += plan.surgery_seconds
        env = plan.next_env  # Execute only the first action; never adopt the leaf.
    statistics = dict(depth=depth, gamma=gamma, menu='standard', state_version='local_v2',
        objective=('r1' if depth == 1 else 'r1 + 0.9*r2; terminal ends path; zero cutoff bootstrap'),
        validation_evaluations=1+evaluations, candidate_evaluations=evaluations,
        explored_candidate_transitions=explored, candidate_paths=leaf_paths,
        reused_evaluations=hits, scoring_seconds=scoring, surgery_seconds=surgery,
        search_seconds=time.perf_counter()-started, terminal_reason=env.terminal_reason,
        reuse_scope='children of the identical concrete parent; no aggregate-state cache')
    return LookaheadResult(env.model, env.original_kept_ids, env.trace, statistics, decisions=decisions)

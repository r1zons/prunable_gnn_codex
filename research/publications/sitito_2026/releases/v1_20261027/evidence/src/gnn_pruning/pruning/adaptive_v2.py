"""Local V2 integer actions, soft-target reward, Greedy and tabular Q search."""
from __future__ import annotations

import copy
import math
import random
import time
from dataclasses import dataclass, field

from gnn_pruning.surgery.local_v2 import prune_hidden_channels_local_v2
from .static_v2 import channel_rankings, compute_scores


STOP = "stop"


def action_key(action) -> str:
    """Stable action identity for sparse Q rows and serialized traces."""
    return STOP if action == STOP else f"prune:{action[0]}:{action[1]}"


def _clone(model):
    import torch
    with torch.random.fork_rng(devices=[]):
        return copy.deepcopy(model).eval()


class LocalV2Env:
    """A frozen-dense search episode; no fit, test feedback or legacy bonuses."""

    def __init__(self, model, evaluate, target: float, penalty_lambda: float,
                 max_steps: int = 20, min_channels: int = 4, tolerance: float = 0.01,
                 greedy: bool = False, accuracy_bucket: float = 0.01):
        if not math.isfinite(target) or not 0 <= target <= 1:
            raise ValueError("Invalid target fraction.")
        if not math.isfinite(penalty_lambda) or penalty_lambda < 0:
            raise ValueError("penalty_lambda must be finite and nonnegative.")
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or not 1 <= max_steps <= 20:
            raise ValueError("max_steps must be an integer in [1,20].")
        if not isinstance(min_channels, int) or min_channels < 4 or tolerance < 0 or not math.isfinite(tolerance):
            raise ValueError("Invalid minimum width or tolerance.")
        if not math.isfinite(accuracy_bucket) or accuracy_bucket <= 0:
            raise ValueError("accuracy_bucket must be positive and finite.")
        self.dense = _clone(model)
        if len(self.dense.convs) != 3 or any(c.out_channels < min_channels for c in self.dense.convs[:-1]):
            raise ValueError("V2 requires two hidden layers at or above minimum width.")
        self.evaluate, self.target, self.penalty_lambda = evaluate, target, penalty_lambda
        self.max_steps, self.min_channels, self.tolerance = max_steps, min_channels, tolerance
        self.greedy, self.accuracy_bucket = greedy, accuracy_bucket
        self.p0 = sum(p.numel() for p in model.parameters())
        self.reset()

    def reset(self):
        """Reset tensors, retained identities, counters and termination to dense."""
        self.model = _clone(self.dense)
        self.original_kept_ids = [list(range(c.out_channels)) for c in self.model.convs[:-1]]
        self.steps, self.sparsity, self.trace = 0, 0.0, []
        self.scoring_seconds, self.surgery_seconds = 0., 0.
        self.terminal, self.terminal_reason = False, None
        self.accuracy = self._accuracy()
        self.dense_accuracy = self.accuracy
        self.evaluation_calls = 1
        if not self._removals():
            self.terminal, self.terminal_reason = True, "no_valid_actions"
            widths = [c.out_channels for c in self.model.convs[:-1]]
            penalty = self._penalty()
            self.trace = [{"action": "no_valid_actions", "successful_deletions": 0,
                           "widths_before": widths, "widths_after": widths,
                           "sparsity": 0., "signed_target_deviation": -self.target,
                           "validation_accuracy": self.accuracy, "accuracy_drop": 0.,
                           "reward": penalty, "terminal_penalty": penalty,
                           "terminal_reason": "no_valid_actions",
                           "original_kept_ids": self.original_kept_ids}]
        return self.state()

    def _accuracy(self):
        value = float(self.evaluate(self.model))
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Validation accuracy must be a finite fraction [0,1].")
        return value

    def _removals(self):
        return [(layer, k) for layer, conv in enumerate(self.model.convs[:-1])
                for k in (1, 2, 4, 8) if conv.out_channels-k >= self.min_channels]

    def legal_actions(self):
        """Filter integer actions, including STOP only after a successful removal."""
        if self.terminal:
            return []
        return self._removals() + ([STOP] if self.steps > 0 and not self.greedy else [])

    def state(self):
        """Versioned aggregate state; retained identities deliberately not included."""
        return ("local_v2", tuple(c.out_channels for c in self.model.convs[:-1]),
                self.max_steps-self.steps, math.floor(self.accuracy/self.accuracy_bucket), self.terminal)

    def _penalty(self):
        return -self.penalty_lambda*max(0., abs(self.sparsity-self.target)-self.tolerance)

    def step(self, action):
        """Apply one legal transition; terminal penalty is paid once then locked."""
        if action != STOP and (not isinstance(action, tuple) or len(action) != 2
                               or any(type(value) is not int for value in action)):
            raise ValueError("V2 pruning actions require integer (layer,k).")
        if action not in self.legal_actions():
            raise ValueError("Invalid or terminal V2 action.")
        before = [c.out_channels for c in self.model.convs[:-1]]
        previous_sparsity = self.sparsity
        reason = None
        if action == STOP:
            reward, reason = 0., "agent_stop"
        else:
            layer, k = action
            started = time.perf_counter()
            scores = compute_scores(self.model, "magnitude")
            ranking = channel_rankings(self.model, scores)[layer]
            keep = sorted(ranking[:len(ranking)-k])
            self.scoring_seconds += time.perf_counter()-started
            started = time.perf_counter()
            self.model = prune_hidden_channels_local_v2(self.model, layer, keep, self.min_channels)
            self.surgery_seconds += time.perf_counter()-started
            self.original_kept_ids = [list(ids) for ids in self.original_kept_ids]
            self.original_kept_ids[layer] = [self.original_kept_ids[layer][i] for i in keep]
            self.steps += 1
            self.accuracy = self._accuracy()
            self.evaluation_calls += 1
            self.sparsity = 1-sum(p.numel() for p in self.model.parameters())/self.p0
            drop = max(0., self.dense_accuracy-self.accuracy)
            reward = 0.6*(self.sparsity-previous_sparsity)-0.4*drop
            if self.steps >= self.max_steps:
                reason = "max_steps"
            elif not self._removals():
                reason = "no_valid_actions"
            elif self.greedy and self.sparsity >= self.target-self.tolerance:
                reason = "target_reached"
        penalty = self._penalty() if reason else 0.
        reward += penalty
        self.terminal, self.terminal_reason = reason is not None, reason
        info = {"action": action_key(action), "successful_deletions": self.steps,
                "widths_before": before, "widths_after": [c.out_channels for c in self.model.convs[:-1]],
                "sparsity": self.sparsity, "signed_target_deviation": self.sparsity-self.target,
                "validation_accuracy": self.accuracy, "accuracy_drop": max(0., self.dense_accuracy-self.accuracy),
                "reward": reward, "terminal_penalty": penalty, "terminal_reason": reason,
                "original_kept_ids": [list(ids) for ids in self.original_kept_ids]}
        self.trace = self.trace + [info]
        return self.state(), reward, self.terminal, info

    def preview(self, action):
        """Fork the current search state; surgery copies tensors on the fork."""
        candidate = copy.copy(self)
        candidate.step(action)
        return candidate


def update_q(table, state, action, reward, next_state, next_actions, terminal: bool,
             alpha: float = 0.1, gamma: float = 0.9):
    """Bellman update over legal next actions only; unseen legal actions are zero."""
    if not 0 < alpha <= 1 or gamma != 0.9:
        raise ValueError("V2 requires alpha in (0,1] and gamma=0.9.")
    future = table.get(next_state, {})
    bootstrap = 0. if terminal else max((future.get(key, 0.) for key in next_actions), default=0.)
    row = table.setdefault(state, {})
    previous = row.get(action, 0.)
    row[action] = previous + alpha*(reward+gamma*bootstrap-previous)


def select_action(table, state, actions, rng: random.Random, epsilon: float = 0.):
    """Epsilon-greedy among legal actions; deterministic ties follow action order."""
    if not actions or not 0 <= epsilon <= 1:
        raise ValueError("Nonempty legal actions and epsilon in [0,1] required.")
    if epsilon > 0 and rng.random() < epsilon:
        return rng.choice(actions)
    row = table.get(state, {})
    return max(actions, key=lambda action: row.get(action_key(action), 0.))


@dataclass
class AdaptiveResult:
    """Actual terminal model, traces and search-resource diagnostics."""
    model: object
    kept_indices: list[list[int]]
    trace: list[dict]
    statistics: dict
    q_table: dict = field(default_factory=dict)
    training_trace: list[dict] = field(default_factory=list)


def run_greedy(model, evaluate, **settings) -> AdaptiveResult:
    """Evaluate every legal pruning candidate and adopt the highest immediate reward."""
    env = LocalV2Env(model, evaluate, greedy=True, **settings)
    candidate_evaluations = 0
    scoring_seconds, surgery_seconds = 0., 0.
    while not env.terminal:
        candidates = [env.preview(action) for action in env.legal_actions()]
        candidate_evaluations += len(candidates)
        scoring_seconds += sum(c.scoring_seconds-env.scoring_seconds for c in candidates)
        surgery_seconds += sum(c.surgery_seconds-env.surgery_seconds for c in candidates)
        env = max(candidates, key=lambda candidate: candidate.trace[-1]["reward"])
    return AdaptiveResult(env.model, env.original_kept_ids, env.trace,
                          {"candidate_evaluations": candidate_evaluations,
                           "validation_evaluations": candidate_evaluations+1,
                           "scoring_seconds": scoring_seconds, "surgery_seconds": surgery_seconds,
                           "terminal_reason": env.terminal_reason})


def run_q_learning(model, evaluate, target: float, penalty_lambda: float, episodes: int = 30,
                   epsilon: float = 0.2, alpha: float = 0.1, gamma: float = 0.9,
                   seed: int = 42, **settings) -> AdaptiveResult:
    """Reset each learning episode and deployment; report actual greedy termination."""
    if not isinstance(episodes, int) or isinstance(episodes, bool) or episodes < 1:
        raise ValueError("episodes must be a positive integer.")
    if gamma != 0.9 or not 0 < alpha <= 1 or not 0 <= epsilon <= 1:
        raise ValueError("Invalid V2 learning parameters; gamma is fixed at0.9.")
    table, visited, slots, training = {}, set(), {}, []
    rng = random.Random(seed)
    evaluations = 0
    scoring_seconds, surgery_seconds = 0., 0.
    for episode in range(episodes):
        env = LocalV2Env(model, evaluate, target, penalty_lambda, **settings)
        training.append({"event": "reset", "episode": episode, "widths": list(env.state()[1])})
        if env.terminal:
            training.extend({"episode": episode, **entry} for entry in env.trace)
        while not env.terminal:
            state, actions = env.state(), env.legal_actions()
            visited.add(state)
            slots[state] = len(actions)
            action = select_action(table, state, actions, rng, epsilon)
            next_state, reward, done, info = env.step(action)
            visited.add(next_state)
            next_keys = [action_key(a) for a in env.legal_actions()]
            slots[next_state] = len(next_keys)
            update_q(table, state, action_key(action), reward, next_state, next_keys, done, alpha, gamma)
            training.append({"episode": episode, **info})
        evaluations += env.evaluation_calls
        scoring_seconds += env.scoring_seconds
        surgery_seconds += env.surgery_seconds
    # Deployment starts dense again and does not update the learned table.
    env = LocalV2Env(model, evaluate, target, penalty_lambda, **settings)
    while not env.terminal:
        state, actions = env.state(), env.legal_actions()
        visited.add(state)
        slots[state] = len(actions)
        env.step(select_action(table, state, actions, rng, epsilon=0.))
    visited.add(env.state())
    slots[env.state()] = 0
    evaluations += env.evaluation_calls
    scoring_seconds += env.scoring_seconds
    surgery_seconds += env.surgery_seconds
    entries = sum(len(row) for row in table.values())
    total_slots = sum(slots.values())
    stats = {"unique_states": len(visited), "q_entries": entries,
             "legal_state_action_slots": total_slots,
             "q_table_sparsity": 1-entries/total_slots if total_slots else 1.,
             "episodes": episodes, "validation_evaluations": evaluations,
             "scoring_seconds": scoring_seconds, "surgery_seconds": surgery_seconds,
             "terminal_reason": env.terminal_reason, "state_version": "local_v2",
             "state_limitation": "aggregated heuristic; no retained-channel signature or MDP guarantee"}
    return AdaptiveResult(env.model, env.original_kept_ids, env.trace, stats, table, training)

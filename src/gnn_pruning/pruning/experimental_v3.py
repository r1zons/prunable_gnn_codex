"""Explicit V3 ablations; the original R2 module and registry remain untouched."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import math
import random

from .adaptive_v2 import LocalV2Env, AdaptiveResult, STOP, action_key, run_q_learning
from .static_v2 import parameter_count_for_widths

VARIANTS = (
    'r2_reference', 'q_stop_exact_v3', 'q_conservative_v3',
    'q_random_ties_v3', 'q_target_stop_v3', 'q_reachable_v3',
    'random_reachable_v3', 'greedy_reachable_v3',
)
REACHABLE = {'q_reachable_v3', 'random_reachable_v3', 'greedy_reachable_v3'}


@lru_cache(None)
def _deletion_actions(amount: int) -> int:
    if amount == 0:
        return 0
    return 1 + min(_deletion_actions(amount-k) for k in (1, 2, 4, 8) if k <= amount)


@dataclass(frozen=True)
class WidthReachability:
    """Exact parameter geometry and at-most-horizon reachability; no labels or metrics."""
    features: int
    classes: int
    dense_widths: tuple[int, int]
    factor: int
    biases: tuple[bool, bool, bool]
    target: Fraction
    tolerance: Fraction
    minimum: int = 4

    @classmethod
    def from_model(cls, model, target, tolerance=.01, minimum=4):
        if not math.isfinite(float(target)) or not 0 <= target <= 1:
            raise ValueError('Invalid target fraction.')
        if not math.isfinite(float(tolerance)) or tolerance < 0 or minimum < 4:
            raise ValueError('Invalid tolerance or minimum width.')
        widths = tuple(c.out_channels for c in model.convs[:-1])
        parameter_count_for_widths(model, widths)  # Established supported-model validation.
        biases = tuple((c.lin_l.bias if hasattr(c, 'lin_l') else c.bias) is not None
                       for c in model.convs)
        obj = cls(model.in_channels, model.out_channels, widths,
                  2 if hasattr(model.convs[0], 'lin_l') else 1, biases,
                  Fraction(str(target)), Fraction(str(tolerance)), minimum)
        if obj.count(widths) != sum(p.numel() for p in model.parameters()):
            raise ValueError('Architecture does not match exact parameter geometry.')
        return obj

    def count(self, widths) -> int:
        """All weights and physically present biases, using integer arithmetic."""
        a, b = widths
        return (self.factor*(self.features*a+a*b+b*self.classes)
                + self.biases[0]*a+self.biases[1]*b+self.biases[2]*self.classes)

    def compression(self, widths) -> float:
        return 1-self.count(widths)/self.count(self.dense_widths)

    def in_band(self, widths) -> bool:
        p0 = self.count(self.dense_widths)
        return abs(Fraction(p0-self.count(widths), p0)-self.target) <= self.tolerance

    @lru_cache(None)
    def goals(self):
        return tuple((a, b) for a in range(self.minimum, self.dense_widths[0]+1)
                     for b in range(self.minimum, self.dense_widths[1]+1)
                     if self.in_band((a, b)))

    def can_reach(self, widths, remaining: int) -> bool:
        """Exact future widths via k={1,2,4,8}; all intermediate widths stay legal."""
        return self._can_reach(tuple(widths), remaining)

    @lru_cache(None)
    def _can_reach(self, widths, remaining: int) -> bool:
        if remaining < 0:
            return False
        a, b = widths
        return any(x <= a and y <= b and _deletion_actions(a-x)+_deletion_actions(b-y) <= remaining
                   for x, y in self.goals())


class V3Env(LocalV2Env):
    """R2 transitions plus explicitly named V3 termination/action-set ablations."""
    def __init__(self, model, evaluate, target, penalty_lambda, variant, **settings):
        if variant not in VARIANTS or variant == 'r2_reference':
            raise ValueError('Use the original environment for the R2 reference.')
        self.variant = variant
        self.space = WidthReachability.from_model(model, target,
                      settings.get('tolerance', .01), settings.get('min_channels', 4))
        super().__init__(model, evaluate, target, penalty_lambda, **settings)

    def reset(self):
        super().reset()
        reason = None
        if self.variant in REACHABLE:
            if not self.space.can_reach(tuple(self.state()[1]), self.max_steps):
                reason = 'initial_target_infeasible'
            elif self.space.in_band(self.state()[1]):
                reason = 'initial_target_band'
        if reason:
            self.terminal, self.terminal_reason = True, reason
            widths = list(self.state()[1]);penalty = self._penalty()
            self.trace = [dict(action=reason, successful_deletions=0,
                widths_before=widths, widths_after=widths, sparsity=0.,
                signed_target_deviation=-self.target, validation_accuracy=self.accuracy,
                accuracy_drop=0., reward=penalty, terminal_penalty=penalty,
                terminal_reason=reason, original_kept_ids=self.original_kept_ids)]
        return self.state()

    def legal_actions(self):
        actions = super().legal_actions()
        if self.variant not in REACHABLE or self.terminal:
            return actions
        widths = self.state()[1];remaining = self.max_steps-self.steps
        filtered = []
        for action in actions:
            if action == STOP:
                continue  # Band arrival terminates; off-band STOP cannot abandon feasibility.
            layer, k = action;future = list(widths);future[layer] -= k
            if self.space.can_reach(tuple(future), remaining-1):
                filtered.append(action)
        if not filtered:
            raise RuntimeError('Reachability invariant violated before target-band arrival.')
        return filtered

    def step(self, action):
        state, reward, done, info = super().step(action)
        reason = None
        if not done and self.variant == 'q_target_stop_v3' and self.sparsity >= self.target-self.tolerance:
            reason = 'target_threshold'
        elif not done and self.variant in REACHABLE and self.space.in_band(self.state()[1]):
            reason = 'target_band'
        if reason:
            penalty = self._penalty();reward += penalty
            self.terminal, self.terminal_reason = True, reason
            info = dict(info, reward=reward, terminal_penalty=penalty, terminal_reason=reason)
            self.trace = self.trace[:-1]+[info]
            state, done = self.state(), True
        return state, reward, done, info


def action_value(table, state, key, variant, space, penalty_lambda, gamma):
    """H1 known terminal STOP value; H2 finite-horizon pessimistic unseen value."""
    if variant == 'q_stop_exact_v3' and key == STOP:
        return -penalty_lambda*max(0., abs(space.compression(state[1])-float(space.target))
                                  -float(space.tolerance))
    if key in table.get(state, {}):
        return table[state][key]
    if variant == 'q_conservative_v3':
        # Every pruning reward >=-.4; one terminal penalty >=-lambda*max_gap.
        # Charging the full penalty immediately is a conservative lower bound.
        horizon = state[2]
        max_gap = max(0., max(float(space.target), 1-float(space.target))-float(space.tolerance))
        return -.4*sum(gamma**j for j in range(horizon))-penalty_lambda*max_gap
    return 0.


def choose_action(table, state, actions, rng, epsilon, variant, space, penalty_lambda, gamma):
    if not actions or not 0 <= epsilon <= 1:
        raise ValueError('Nonempty actions and valid epsilon required.')
    if epsilon > 0 and rng.random() < epsilon:
        return rng.choice(actions)
    values = [action_value(table, state, action_key(a), variant, space, penalty_lambda, gamma)
              for a in actions]
    best = max(values);ties = [a for a, v in zip(actions, values) if v == best]
    return rng.choice(ties) if variant == 'q_random_ties_v3' and len(ties) > 1 else ties[0]


def _controls(model, evaluate, variant, target, penalty_lambda, seed, **settings):
    env = V3Env(model, evaluate, target, penalty_lambda, variant, **settings)
    rng = random.Random(seed);candidates = 0;scoring = surgery = 0.
    while not env.terminal:
        if variant == 'random_reachable_v3':
            env.step(rng.choice(env.legal_actions()))
        else:
            previews = [env.preview(a) for a in env.legal_actions()]
            candidates += len(previews)
            scoring += sum(c.scoring_seconds-env.scoring_seconds for c in previews)
            surgery += sum(c.surgery_seconds-env.surgery_seconds for c in previews)
            env = max(previews, key=lambda c: c.trace[-1]['reward'])
    return AdaptiveResult(env.model, env.original_kept_ids, env.trace,
        dict(variant=variant, candidate_evaluations=candidates,
             validation_evaluations=candidates+1 if variant == 'greedy_reachable_v3' else env.evaluation_calls,
             scoring_seconds=scoring if candidates else env.scoring_seconds,
             surgery_seconds=surgery if candidates else env.surgery_seconds,
             terminal_reason=env.terminal_reason))


def run_variant(model, evaluate, variant, target, penalty_lambda, episodes=100,
                epsilon=.2, alpha=.1, gamma=.9, seed=101, **settings):
    """Run a separate ablation; r2_reference delegates verbatim to the old routine."""
    if variant not in VARIANTS:
        raise ValueError('Unknown V3 variant.')
    if variant == 'r2_reference':
        return run_q_learning(model, evaluate, target, penalty_lambda, episodes=episodes,
                              epsilon=epsilon, alpha=alpha, gamma=gamma, seed=seed, **settings)
    if (type(episodes) is not int or episodes < 1 or gamma != .9
            or not 0 < alpha <= 1 or not 0 <= epsilon <= 1):
        raise ValueError('Invalid episode/learning settings.')
    if variant in ('random_reachable_v3', 'greedy_reachable_v3'):
        return _controls(model, evaluate, variant, target, penalty_lambda, seed, **settings)
    table = {};visited = set();slots = {};updated = set();training = []
    rng = random.Random(seed);evaluations = 0;scoring = surgery = 0.
    for episode in range(episodes+1):
        env = V3Env(model, evaluate, target, penalty_lambda, variant, **settings)
        learning = episode < episodes
        if learning:
            training.append(dict(event='reset', episode=episode, widths=list(env.state()[1])))
            if env.terminal:
                training.extend(dict(episode=episode, **e) for e in env.trace)
        while not env.terminal:
            state, actions = env.state(), env.legal_actions()
            visited.add(state);slots[state] = len(actions)
            action = choose_action(table, state, actions, rng, epsilon if learning else 0.,
                                   variant, env.space, penalty_lambda, gamma)
            next_state, reward, done, info = env.step(action)
            if learning:
                next_keys = [action_key(a) for a in env.legal_actions()]
                future = 0. if done else max(action_value(table, next_state, k, variant,
                              env.space, penalty_lambda, gamma) for k in next_keys)
                key = action_key(action)
                previous = action_value(table, state, key, variant, env.space, penalty_lambda, gamma)
                table.setdefault(state, {})[key] = (reward if variant == 'q_stop_exact_v3' and action == STOP
                    else previous+alpha*(reward+gamma*future-previous))
                updated.add((state, key));visited.add(next_state);slots[next_state] = len(next_keys)
                training.append(dict(episode=episode, **info))
        visited.add(env.state());slots[env.state()] = 0
        evaluations += env.evaluation_calls;scoring += env.scoring_seconds;surgery += env.surgery_seconds
    analytic = 0
    if variant == 'q_stop_exact_v3':
        for s in visited:
            if not s[-1] and s[2] < env.max_steps:
                table.setdefault(s, {})[STOP] = action_value(table, s, STOP, variant, env.space, penalty_lambda, gamma)
                analytic += 1
    entries = sum(map(len, table.values()));total = sum(slots.values())
    return AdaptiveResult(env.model, env.original_kept_ids, env.trace,
        dict(variant=variant, episodes=episodes, unique_states=len(visited), q_entries=entries,
             learned_q_entries=len(updated), analytic_stop_entries=analytic,
             legal_state_action_slots=total, q_table_sparsity=1-entries/total if total else 1.,
             validation_evaluations=evaluations, candidate_evaluations=0,
             scoring_seconds=scoring, surgery_seconds=surgery,
             terminal_reason=env.terminal_reason, state_version='local_v2'),table,training)

"""Isolated, single-factor V4 diagnostics on the unchanged Local V2 state/surgery."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from functools import lru_cache
import random

from .adaptive_v2 import AdaptiveResult, LocalV2Env, action_key
from .experimental_v3 import WidthReachability


@dataclass(frozen=True)
class Settings:
    """Predeclared settings; comparison C changes only B's exploration schedule."""
    menu: str = 'standard'
    episodes: int = 100
    epsilon: float = .2
    decay: bool = False
    random_ties: bool = False
    incremental_drop: bool = False
    gamma: float = .9
    greedy: bool = False


REFERENCE = 'q_v4_reference'
Q_VARIANTS = ('q_v4_menu12', 'q_v4_percent', 'q_v4_300', 'q_v4_decay300',
              'q_v4_random_ties', 'q_v4_incremental_drop', 'q_v4_gamma0')
GREEDY_VARIANTS = ('greedy_v4_menu12', 'greedy_v4_percent')
PLANS = {
    REFERENCE: Settings(),
    'q_v4_menu12': Settings(menu='one_two'),
    'q_v4_percent': Settings(menu='percent'),
    'q_v4_300': Settings(episodes=300),
    'q_v4_decay300': Settings(episodes=300, epsilon=.4, decay=True),
    'q_v4_random_ties': Settings(random_ties=True),
    'q_v4_incremental_drop': Settings(incremental_drop=True),
    'q_v4_gamma0': Settings(gamma=0.),
    'greedy_v4_menu12': Settings(menu='one_two', greedy=True),
    'greedy_v4_percent': Settings(menu='percent', greedy=True),
}


def action_sizes(menu: str, width: int) -> tuple[int, ...]:
    """Ascending unique deletions; percentages use exact ceil(p*current width)."""
    if menu == 'standard':return (1, 2, 4, 8)
    if menu == 'one_two':return (1, 2)
    if menu == 'percent':return tuple(sorted({(5*width+99)//100, (10*width+99)//100}))
    raise ValueError('Unknown action menu.')


@dataclass(frozen=True)
class MenuReachability:
    """Exact directed width-DAG distances, recomputing the menu at every width.

    Layer paths are independent until their lengths are added; all possible goal
    pairs use the frozen integer parameter geometry and exact rational band.
    """
    geometry: WidthReachability
    menu: str

    def __post_init__(self):
        action_sizes(self.menu, self.geometry.dense_widths[0])

    @lru_cache(None)
    def distance(self, start: int, goal: int) -> int:
        """Shortest legal per-layer path; 10**6 denotes an unreachable width."""
        if start == goal:return 0
        if start < goal or goal < self.geometry.minimum:return 10**6
        paths = [self.distance(start-k, goal) for k in action_sizes(self.menu, start)
                 if start-k >= goal]
        return 1+min(paths, default=10**6)

    @lru_cache(None)
    def can_reach(self, widths: tuple[int, int], remaining: int) -> bool:
        """Existence of a future in-band pair within the successful-action horizon."""
        if remaining < 0:return False
        return any(self.distance(widths[0], a)+self.distance(widths[1], b) <= remaining
                   for a,b in self.geometry.goals())


class V4Env(LocalV2Env):
    """V3 constrained transitions, new menus and optional difference-of-drop reward."""
    def __init__(self, model, evaluate, target, settings=Settings(), **kwargs):
        self.settings = settings
        self.space = WidthReachability.from_model(model, target, kwargs.get('tolerance', .01), kwargs.get('min_channels', 4))
        self.oracle = MenuReachability(self.space, settings.menu)
        horizon = kwargs.get('max_steps', 20)
        if not self.oracle.can_reach(self.space.dense_widths, horizon):
            raise ValueError('initial_target_infeasible: exact menu/horizon/band; no tolerance fallback')
        super().__init__(model, evaluate, target, **kwargs)

    def _removals(self):
        return [(layer,k) for layer,conv in enumerate(self.model.convs[:-1])
                for k in action_sizes(self.settings.menu, conv.out_channels)
                if conv.out_channels-k >= self.min_channels]

    def reset(self):
        state = super().reset()
        if self.space.in_band(state[1]):
            self.terminal, self.terminal_reason = True, 'initial_target_band'
            state = self.state()
        return state

    def legal_actions(self):
        if self.terminal:return []
        widths = self.state()[1];remaining = self.max_steps-self.steps
        actions = []
        for layer,k in self._removals():
            future = list(widths);future[layer] -= k
            if self.oracle.can_reach(tuple(future), remaining-1):actions.append((layer,k))
        if not actions:raise RuntimeError('No reachable continuation before terminal band.')
        return actions

    def step(self, action):
        previous_drop = max(0., self.dense_accuracy-self.accuracy)
        state,reward,done,info = super().step(action)
        if self.settings.incremental_drop:
            reward += .4*previous_drop
            info = dict(info, reward=reward, previous_accuracy_drop=previous_drop,
                        accuracy_drop_difference=info['accuracy_drop']-previous_drop)
        if not done and self.space.in_band(state[1]):
            penalty = self._penalty();reward += penalty
            self.terminal, self.terminal_reason = True, 'target_band'
            info = dict(info, reward=reward, terminal_penalty=penalty, terminal_reason='target_band')
            done = True;state = self.state()
        self.trace = self.trace[:-1]+[info]
        return state,reward,done,info


def epsilon_at(settings: Settings, episode: int) -> float:
    """Linear inclusive .4→.05 over all 300 training episodes; deployment is zero."""
    if not settings.decay:return settings.epsilon
    if not 0 <= episode < settings.episodes:raise ValueError('Invalid schedule episode.')
    return .4+(.05-.4)*episode/(settings.episodes-1)


def select(table, state, actions, rng, epsilon, random_ties=False):
    """Use exactly V3's RNG draws, with additional draws only for tied-max variant D."""
    if epsilon > 0 and rng.random() < epsilon:
        return rng.choice(actions), 'explore'
    row = table.get(state, {});values = [row.get(action_key(a), 0.) for a in actions]
    best = max(values);ties = [a for a,v in zip(actions,values) if v == best]
    return (rng.choice(ties) if random_ties and len(ties)>1 else ties[0]), 'exploit'


def bellman(table, state, action, reward, next_state, next_actions, done, gamma):
    """Alpha .1; bootstrap only from legal reachable successors; gamma .9 or zero."""
    if gamma not in (0., .9):raise ValueError('Unapproved gamma.')
    future = 0. if done else max((table.get(next_state, {}).get(action_key(a), 0.) for a in next_actions), default=0.)
    row = table.setdefault(state, {});previous = row.get(action_key(action), 0.)
    row[action_key(action)] = previous+.1*(reward+gamma*future-previous)
    return previous, row[action_key(action)], future


@dataclass
class V4Result(AdaptiveResult):
    snapshots: dict = field(default_factory=dict)
    decisions: list = field(default_factory=list)


def run_variant(model, evaluate, variant, target, seed=101, max_steps=20,
                min_channels=4, tolerance=.01, episodes_override=None, snapshots=(25,50,100,200,300)):
    """Run an approved single-factor plan; truncation is for labelled preflight only."""
    if variant not in PLANS:raise ValueError('Unknown V4 variant.')
    if seed not in (101,5001,5002,5003,5004):raise ValueError('Unapproved policy seed.')
    plan = PLANS[variant];episodes = plan.episodes if episodes_override is None else episodes_override
    if type(episodes) is not int or not 1 <= episodes <= plan.episodes:raise ValueError('Invalid episode budget.')
    settings = dict(target=target, penalty_lambda=2., settings=plan, max_steps=max_steps,
                    min_channels=min_channels, tolerance=tolerance)
    if plan.greedy:
        env = V4Env(model, evaluate, **settings);candidates=0;scoring=surgery=0.
        while not env.terminal:
            previews = [env.preview(a) for a in env.legal_actions()];candidates+=len(previews)
            scoring += sum(c.scoring_seconds-env.scoring_seconds for c in previews)
            surgery += sum(c.surgery_seconds-env.surgery_seconds for c in previews)
            env = max(previews, key=lambda c:c.trace[-1]['reward'])
        return V4Result(env.model,env.original_kept_ids,env.trace,
            dict(variant=variant,validation_evaluations=candidates+1,candidate_evaluations=candidates,
                 scoring_seconds=scoring,surgery_seconds=surgery,terminal_reason=env.terminal_reason))
    table={};visited=set();slots={};training=[];decisions=[];saved={};rng=random.Random(seed)
    evaluations=0;scoring=surgery=0.
    for episode in range(episodes+1):
        learning=episode<episodes
        env=V4Env(model,evaluate,**settings)
        if learning:training.append(dict(event='reset',episode=episode,widths=list(env.state()[1])))
        epsilon=epsilon_at(plan,episode) if learning else 0.
        while not env.terminal:
            state,actions=env.state(),env.legal_actions();visited.add(state);slots[state]=len(actions)
            action,mode=select(table,state,actions,rng,epsilon,plan.random_ties)
            row=table.get(state,{});values={action_key(a):row.get(action_key(a),0.) for a in actions}
            decision=dict(phase='training' if learning else 'deployment',episode=episode if learning else None,
                state=state,action=action_key(action),epsilon=epsilon,mode=mode,
                selected_action_unseen=action_key(action) not in row,
                unseen_legal_actions=sum(action_key(a) not in row for a in actions),
                equal_max_count=sum(v==max(values.values()) for v in values.values()),
                legal_q_values=values,original_kept_ids=copy.deepcopy(env.original_kept_ids))
            next_state,reward,done,info=env.step(action)
            if learning:
                next_actions=env.legal_actions();before,after,future=bellman(table,state,action,reward,next_state,next_actions,done,plan.gamma)
                decision.update(q_before=before,q_after=after,bootstrap=future,terminal=done,next_state=next_state,reward=reward)
                visited.add(next_state);slots[next_state]=len(next_actions)
                training.append(dict(episode=episode,**info))
            decisions.append(decision)
        visited.add(env.state());slots[env.state()]=0
        evaluations+=env.evaluation_calls;scoring+=env.scoring_seconds;surgery+=env.surgery_seconds
        if learning and episode+1 in snapshots:
            saved[episode+1]=dict(episodes_completed=episode+1,
                q_table=[dict(state=s,actions=dict(row)) for s,row in table.items()],
                rng_state=rng.getstate(),epsilon_last=epsilon,
                note='Q policy snapshot; no deployment, evaluation or RNG draws performed.')
    entries=sum(map(len,table.values()));total=sum(slots.values())
    return V4Result(env.model,env.original_kept_ids,env.trace,
        dict(variant=variant,episodes=episodes,planned_episodes=plan.episodes,unique_states=len(visited),q_entries=entries,
             legal_state_action_slots=total,q_table_sparsity=1-entries/total if total else 1.,
             validation_evaluations=evaluations,scoring_seconds=scoring,surgery_seconds=surgery,
             terminal_reason=env.terminal_reason,state_version='local_v2',menu=plan.menu,
             deployment_unseen_choices=sum(d['selected_action_unseen'] for d in decisions if d['phase']=='deployment')),
        table,training,saved,decisions)

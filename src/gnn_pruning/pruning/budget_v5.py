"""Versioned arbitrary-depth, exact-budget controllers; historical V2–V4 unchanged.

The action menu contains unit deletions. A horizon equal to the initial number
of removable channels therefore suffices for every descendant width tuple. A
prefix-OR goal lattice answers exact continuation queries without label access.
"""
from __future__ import annotations

import copy
import itertools
import json
import math
import random
import time
from fractions import Fraction

import numpy as np
import torch

from .adaptive_v2 import AdaptiveResult, _clone, action_key, update_q
from .static_v2 import (StaticPlan, apply_static_plan, channel_rankings,
                        compute_scores, parameter_count_for_widths)
from gnn_pruning.surgery.local_v2 import prune_hidden_channels_local_v2

VERSION = 'structural_budget_v5'
SIZES = (1, 2, 4, 8)
METHODS = ('static_random', 'static_magnitude', 'static_snip', 'sequential_random',
           'adaptive_greedy', 'lookahead_depth2', 'tabular_q')


class Budget:
    """Integer parameter geometry and downward reachability for 1–3 hidden layers."""

    def __init__(self, model, target, tolerance=.01, minimum=4):
        if not math.isfinite(target) or not 0 < target < 1:
            raise ValueError('Target must be finite and strictly between zero and one.')
        if not math.isfinite(tolerance) or tolerance < 0 or type(minimum) is not int or minimum < 4:
            raise ValueError('Invalid tolerance/minimum.')
        self.original = tuple(c.out_channels for c in model.convs[:-1])
        if not 1 <= len(self.original) <= 3 or min(self.original) < minimum:
            raise ValueError('Budget V5 supports depths2/3/4 above minimum width.')
        self.p0 = parameter_count_for_widths(model, self.original)
        self.minimum, self.target, self.tolerance = minimum, target, tolerance
        self.horizon = sum(w-minimum for w in self.original)
        self.factor = 2 if hasattr(model.convs[0], 'lin_l') else 1
        self.biases = tuple(int((c.lin_l.bias if hasattr(c, 'lin_l') else c.bias) is not None)
                            for c in model.convs)
        self.features, self.classes = model.in_channels, model.out_channels
        axes = np.ogrid[tuple(slice(0, w+1) for w in self.original)]
        dims = [self.features, *axes, self.classes]
        counts = sum(self.factor*a*b + bias*b
                     for a, b, bias in zip(dims, dims[1:], self.biases))
        # Broadcast the 1-hidden-layer expression too; invalid widths never goals.
        valid = np.ones(tuple(w+1 for w in self.original), dtype=bool)
        for axis in axes:
            valid &= axis >= minimum
        valid &= counts < self.p0
        if not valid.any():
            raise ValueError('No physically smaller legal model.')
        nominal, eps = Fraction(str(target)), Fraction(str(tolerance))
        # Compression bounds converted to exact integer parameter limits.
        lo = math.ceil((1-nominal-eps)*self.p0)
        hi = math.floor((1-nominal+eps)*self.p0)
        goals = valid & (counts >= lo) & (counts <= hi)
        self.fallback = not bool(goals.any())
        if self.fallback:
            distance = np.abs((self.p0-counts)*nominal.denominator - nominal.numerator*self.p0)
            closest = distance[valid].min()
            goals = valid & (distance == closest)
        self.goals = goals
        self.candidate_count = int(valid.sum())
        self.goal_count = int(goals.sum())
        self.prefix = goals.copy()
        for axis in range(len(self.original)):
            np.logical_or.accumulate(self.prefix, axis=axis, out=self.prefix)

    def count(self, widths):
        dims = [self.features, *widths, self.classes]
        return sum(self.factor*a*b+bias*b for a,b,bias in zip(dims,dims[1:],self.biases))

    def goal(self, widths):
        return bool(self.goals[tuple(widths)])

    def reachable(self, widths, remaining):
        """Exact for the declared unit-deletion horizon, not an arbitrary short cap."""
        if remaining < sum(w-self.minimum for w in widths):
            raise ValueError('Insufficient remaining unit-deletion horizon; no approximate oracle.')
        return bool(self.prefix[tuple(widths)])


def static_plan(model, criterion, budget, seed, data=None, train_idx=None):
    """Fixed dense ranks; exhaustive widths, exact union saliency via prefix sums."""
    rng = random.Random(seed)
    scores = compute_scores(model, criterion, data, train_idx)
    coords = np.argwhere(budget.goals)  # lexicographic width order, as historical product
    if criterion == 'random':
        widths = coords[rng.randrange(len(coords))].tolist()
        keeps = [sorted(rng.sample(range(n), k)) for n,k in zip(budget.original,widths)]
        score = None
    else:
        rankings = channel_rankings(model, scores)
        orders = [list(range(model.in_channels)), *rankings, list(range(model.out_channels))]
        retained = np.zeros(len(coords), dtype=np.float64)
        dense_score = 0.
        for name, tensor in scores.items():
            layer = int(name.split('.')[1]); value = tensor.detach().cpu().double().numpy()
            dense_score += float(value.sum(dtype=np.float64))
            rows = coords[:,layer] if layer < len(budget.original) else np.full(len(coords), model.out_channels)
            if tensor.ndim == 2:
                value = value[np.ix_(orders[layer+1], orders[layer])]
                prefix = np.pad(value.cumsum(0).cumsum(1), ((1,0),(1,0)))
                cols = coords[:,layer-1] if layer > 0 else np.full(len(coords), model.in_channels)
                retained += prefix[rows,cols]
            else:
                prefix = np.r_[0.,value[orders[layer+1]].cumsum()]
                retained += prefix[rows]
        choice = int(np.argmax(retained))  # min removed union score, stable exact ties
        widths = coords[choice].tolist()
        keeps = [sorted(rank[:width]) for rank,width in zip(rankings,widths)]
        score = dense_score-float(retained[choice])
    count = budget.count(widths); achieved = 1-count/budget.p0
    plan = StaticPlan(criterion, widths, keeps, budget.target, achieved, abs(achieved-budget.target),
                      budget.fallback, budget.fallback, budget.p0, count, score,
                      budget.candidate_count, budget.goal_count)
    return plan


class BudgetEnv:
    """Shared physical operation, legal mask, reward and termination for new variants."""

    def __init__(self, model, evaluate, budget, dense_accuracy=None, penalty_lambda=2.):
        self.model = _clone(model)
        self.evaluate, self.budget = evaluate, budget
        self.kept = [list(range(w)) for w in budget.original]
        self.steps, self.sparsity, self.trace = 0, 0., []
        self.accuracy = self._accuracy() if dense_accuracy is None else dense_accuracy
        self.dense_accuracy = self.accuracy
        self.terminal, self.reason = False, None
        self.penalty_lambda = penalty_lambda

    def _accuracy(self):
        value = float(self.evaluate(self.model))
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Validation accuracy must be a finite fraction.')
        return value

    def widths(self):
        return tuple(map(len,self.kept))

    def state(self):
        """Target, exact remaining parameters/horizon and retained identities condition Q."""
        return (VERSION, str(self.budget.target), str(self.budget.tolerance), self.widths(),
                self.budget.p0-self.budget.count(self.widths()), self.budget.horizon-self.steps,
                math.floor(self.accuracy/.01), tuple(map(tuple,self.kept)), self.terminal)

    def legal_actions(self):
        if self.terminal:
            return []
        actions=[]
        for layer,width in enumerate(self.widths()):
            for k in SIZES:
                if width-k < self.budget.minimum:
                    continue
                future=list(self.widths());future[layer]-=k
                if self.budget.reachable(future,self.budget.horizon-self.steps-1):
                    actions.append((layer,k))
        if not actions:
            raise RuntimeError('No reachable continuation; refusing unpruned success.')
        return actions

    def preview(self, action):
        child=copy.copy(self)
        child.step(action)
        return child

    def step(self, action):
        if action not in self.legal_actions():
            raise ValueError('Illegal/terminal action; STOP is unavailable outside the goal.')
        before=self.widths();old_s=self.sparsity
        layer,k=action
        ranking=channel_rankings(self.model,compute_scores(self.model,'magnitude'))[layer]
        keep=sorted(ranking[:len(ranking)-k])
        self.model=prune_hidden_channels_local_v2(self.model,layer,keep,self.budget.minimum)
        self.kept=[list(ids) for ids in self.kept]
        self.kept[layer]=[self.kept[layer][i] for i in keep]
        self.steps+=1;self.accuracy=self._accuracy()
        self.sparsity=1-self.budget.count(self.widths())/self.budget.p0
        self.terminal=self.budget.goal(self.widths())
        self.reason=('closest_feasible' if self.budget.fallback else 'target_band') if self.terminal else None
        if not self.terminal and self.steps >= self.budget.horizon:
            raise RuntimeError('Budget horizon exhausted before a reachable goal.')
        penalty=-self.penalty_lambda*max(0.,abs(self.sparsity-self.budget.target)-self.budget.tolerance) if self.terminal else 0.
        reward=.6*(self.sparsity-old_s)-.4*max(0.,self.dense_accuracy-self.accuracy)+penalty
        info=dict(action=action_key(action),widths_before=list(before),widths_after=list(self.widths()),
                  original_kept_ids=self.kept,successful_deletions=self.steps,sparsity=self.sparsity,
                  validation_accuracy=self.accuracy,accuracy_drop=max(0.,self.dense_accuracy-self.accuracy),
                  reward=reward,terminal_penalty=penalty,terminal_reason=self.reason)
        self.trace=self.trace+[info]
        return self.state(),reward,self.terminal,info


def plan(env, depth=2, cached=None):
    """Exhaustive depth1/2; stable ties, execute one first action, exact-parent reuse."""
    if depth not in (1,2):
        raise ValueError('Only depth1/2 supported.')
    signature=lambda e:(e.state(),id(e.model),id(e.evaluate),e.accuracy,e.dense_accuracy,e.penalty_lambda)
    reusable=(cached[2] if cached is not None and cached[0] is env and cached[1]==signature(env) else {})
    first={};seconds={};paths=[];new=hits=0
    for action in env.legal_actions():
        if action in reusable:
            first[action]=reusable[action];hits+=1
        else:
            first[action]=env.preview(action);new+=1
    for action,child in first.items():
        r1=child.trace[-1]['reward'];seconds[action]={}
        if depth==1 or child.terminal:
            paths.append(dict(actions=[action_key(action)],rewards=[r1],score=r1,terminal=child.terminal,
                              leaf=child.trace[-1]))
        else:
            for action2 in child.legal_actions():
                leaf=child.preview(action2);new+=1;seconds[action][action2]=leaf
                r2=leaf.trace[-1]['reward']
                paths.append(dict(actions=[action_key(action),action_key(action2)],rewards=[r1,r2],
                                  score=r1+.9*r2,terminal=leaf.terminal,leaf=leaf.trace[-1]))
    best=max(paths,key=lambda p:p['score'])
    action=next(a for a in first if action_key(a)==best['actions'][0])
    decision=dict(state=env.state(),legal_actions=[action_key(a) for a in first],
                  selected_action=action_key(action),best_planned_path=best['actions'],candidate_paths=paths,
                  candidate_evaluations=new,reused_evaluations=hits,
                  first_transitions=[dict(action=action_key(a),transition=c.trace[-1]) for a,c in first.items()])
    chosen=first[action]
    return chosen,(chosen,signature(chosen),seconds[action]),decision,new,hits


def run(model, evaluate, method, target, seed=101, tolerance=.01, minimum=4,
        episodes=30, epsilon=.2, alpha=.1, gamma=.9, depth=None, budget=None,
        max_deployment_steps=None):
    """New namespaced policies; optional bounded prefix is technical-only evidence."""
    if method not in METHODS[3:]:
        raise ValueError('Adaptive method required.')
    if type(episodes) is not int or episodes < 1 or gamma!=.9 or not 0<alpha<=1 or not 0<=epsilon<=1:
        raise ValueError('Invalid tabular learning protocol.')
    if depth is not None and (type(depth) is not int or depth not in (1,2)):
        raise ValueError('Only depth1/2 supported.')
    if max_deployment_steps is not None and (type(max_deployment_steps) is not int or max_deployment_steps<1):
        raise ValueError('A technical prefix requires a positive integer limit.')
    budget=budget or Budget(model,target,tolerance,minimum)
    if budget.target!=target or budget.original!=tuple(model.hidden_channel_dims) or budget.p0!=sum(p.numel() for p in model.parameters()):
        raise ValueError('Budget/model/target mismatch.')
    rng=random.Random(seed);calls=0
    def measured(m):
        nonlocal calls
        calls+=1
        return evaluate(m)
    started=time.perf_counter();initial=BudgetEnv(model,measured,budget)
    table={};training=[];decisions=[];visited={};unseen=ties_total=0
    def selection(env,explore):
        nonlocal unseen,ties_total
        state=env.state();actions=env.legal_actions();keys=[action_key(a) for a in actions]
        row=table.get(state,{})
        values=[row.get(k,0.) for k in keys];best=max(values)
        tied=[i for i,v in enumerate(values) if v==best]
        exploration=explore>0 and rng.random()<explore
        index=rng.randrange(len(actions)) if exploration else tied[0]
        visited[state]=len(actions)
        if not explore:
            unseen+=keys[index] not in row;ties_total+=len(tied)>1
        return actions[index],dict(state=state,legal_actions=keys,q_values=values,
                    known=[k in row for k in keys],selected_action=keys[index],
                    tied_maxima=[keys[i] for i in tied],exploration=exploration)
    if method=='tabular_q':
        for episode in range(episodes):
            env=BudgetEnv(model,measured,budget,initial.accuracy)
            while not env.terminal:
                state=env.state();action,decision=selection(env,epsilon)
                next_state,reward,done,info=env.step(action)
                key=action_key(action);previous=table.get(state,{}).get(key,0.)
                next_keys=[action_key(a) for a in env.legal_actions()]
                update_q(table,state,key,reward,next_state,next_keys,done,alpha,gamma)
                training.append(dict(episode=episode,**decision,**info,q_before=previous,
                                     q_after=table[state][key],next_state=next_state,next_legal_actions=next_keys))
    env=initial;cache=None;previews=hits=0
    while not env.terminal:
        if max_deployment_steps is not None and env.steps >= max_deployment_steps:
            break  # no research caller may mark this technical prefix completed
        if method in ('adaptive_greedy','lookahead_depth2'):
            env,cache,decision,n,h=plan(env,depth or (1 if method=='adaptive_greedy' else 2),cache)
            previews+=n;hits+=h;decisions.append(decision)
        elif method=='sequential_random':
            actions=env.legal_actions();action=rng.choice(actions)
            decisions.append(dict(state=env.state(),legal_actions=list(map(action_key,actions)),selected_action=action_key(action)))
            env.step(action)
        else:
            action,decision=selection(env,0.);decisions.append(decision);env.step(action)
    entries=sum(map(len,table.values()));slots=sum(visited.values())
    stats=dict(version=VERSION,state_version=VERSION,termination_policy='exact_band_or_global_closest',
               terminal_reason=env.reason or 'technical_prefix',complete=env.terminal,
               horizon=budget.horizon,band_feasible=not budget.fallback,
               validation_evaluations=calls,candidate_evaluations=previews,reused_evaluations=hits,
               q_entries=entries,unique_states=len(visited),legal_state_action_slots=slots,
               q_table_coverage=entries/slots if slots else 0.,deployment_unseen_choices=unseen,
               deployment_tied_maxima=ties_total,episodes=episodes if method=='tabular_q' else 0,
               search_seconds=time.perf_counter()-started)
    result=AdaptiveResult(env.model,env.kept,env.trace,stats,table,training)
    result.decisions=decisions
    return result

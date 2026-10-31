"""Synthetic independent decision/geometry oracles; no datasets or model fitting."""
from __future__ import annotations

import copy
import random

import pytest
import torch

from gnn_pruning.models import build_model
from gnn_pruning.pruning.adaptive_v2 import LocalV2Env, run_greedy
from gnn_pruning.pruning.experimental_v4 import Settings, V4Env
from gnn_pruning.pruning.lookahead import LookaheadPlanner, run_lookahead


def model(kind):
    torch.manual_seed(41)
    return build_model(kind, in_channels=5, hidden_channels=8, out_channels=3,
                       num_layers=3, dropout=.6).eval()


def count(kind, widths):
    a, b = widths
    return (2 if kind == 'graphsage' else 1)*(5*a+a*b+3*b)+a+b+3


def accuracy(widths):
    a, b = widths
    return .9-.003*(8-a)**2-.007*(8-b)**2+.002*(8-a)*(8-b)


def oracle(kind, widths, remaining, target=.44, tolerance=.01, depth=2):
    """Independent width graph + algebraic rewards, not production env previews."""
    p0 = count(kind, (8,8))
    raw = lambda w: [(i,k) for i in (0,1) for k in (1,2,4,8) if w[i]-k >= 4]
    after = lambda w,a: tuple(w[i]-(a[1] if i == a[0] else 0) for i in (0,1))

    def paths(w, h, d):
        rows = []
        for action in raw(w):
            nxt = after(w, action)
            compression = 1-count(kind, nxt)/p0
            gain = (count(kind, w)-count(kind, nxt))/p0
            terminal = h == 1 or not raw(nxt) or compression >= target-tolerance
            reward = .6*gain-.4*max(0, .9-accuracy(nxt))
            if terminal:
                reward -= 2*max(0, abs(compression-target)-tolerance)
            if d == 1 or terminal:
                rows.append(([action], reward))
            else:
                rows.extend(([action]+tail, reward+.9*score)
                            for tail, score in paths(nxt, h-1, d-1))
        return rows
    return paths(tuple(widths), remaining, depth)


def key(action):
    return f'prune:{action[0]}:{action[1]}'


def without_times(decisions):
    return [{k:v for k,v in d.items() if k != 'planning_seconds'} for d in decisions]


@pytest.mark.parametrize('kind',['gcn','graphsage'])
def test_depth_two_matches_independent_oracle_at_every_replanned_state(kind):
    r = run_lookahead(model(kind),lambda m:accuracy(m.hidden_channel_dims),.44,max_steps=3)
    for decision in r.decisions:
        _, widths, remaining, _, _ = decision['state']
        expected = oracle(kind,widths,remaining)
        actual = decision['candidate_paths']
        assert [p['actions'] for p in actual] == [[key(a) for a in path] for path,_ in expected]
        assert [p['score'] for p in actual] == pytest.approx([score for _,score in expected])
        expected_best = max(expected,key=lambda p:p[1])
        assert decision['selected_action'] == key(expected_best[0][0])
    cutoffs = [p for p in r.decisions[0]['candidate_paths'] if not p['terminal']]
    assert cutoffs  # Depth cutoff is not an environment termination or a new STOP.
    assert all(p['terminal_reason'] is None for p in cutoffs)
    assert r.trace[-1]['terminal_reason'] and r.statistics['depth'] == 2


@pytest.mark.parametrize('kind',['gcn','graphsage'])
def test_depth_one_exact_native_greedy_actions_masks_tensors_and_metrics(kind):
    m = model(kind)
    evaluate = lambda m:accuracy(m.hidden_channel_dims)
    expected = run_greedy(m, evaluate, target=.44, penalty_lambda=2., max_steps=3)
    actual = run_lookahead(m,evaluate,.44,depth=1,max_steps=3)
    assert actual.trace == expected.trace
    assert actual.kept_indices == expected.kept_indices
    assert evaluate(actual.model) == evaluate(expected.model)
    for name,value in actual.model.state_dict().items():
        assert torch.equal(value,expected.model.state_dict()[name])
    for name in ('candidate_evaluations','validation_evaluations','terminal_reason'):
        assert actual.statistics[name] == expected.statistics[name]
    assert actual.statistics['reused_evaluations'] == 0


@pytest.mark.parametrize('kind',['gcn','graphsage'])
def test_parent_dense_and_siblings_remain_unmodified(kind):
    m = model(kind)
    env = LocalV2Env(m,lambda m:accuracy(m.hidden_channel_dims),.44,penalty_lambda=2.,max_steps=3,greedy=True)
    before = copy.deepcopy((env.state(),env.trace,env.original_kept_ids,env.evaluation_calls,
                            env.scoring_seconds,env.surgery_seconds))
    source = {k:v.clone() for k,v in env.model.state_dict().items()}
    dense = {k:v.clone() for k,v in m.state_dict().items()}
    plan = LookaheadPlanner().plan(env)
    assert (env.state(),env.trace,env.original_kept_ids,env.evaluation_calls,
            env.scoring_seconds,env.surgery_seconds) == before
    assert env.model is not plan.next_env.model
    for k,v in source.items():
        assert torch.equal(v,env.model.state_dict()[k])
        assert torch.equal(dense[k],m.state_dict()[k])
    assert len(plan.next_env.trace) == 1  # Only one action, even for a 2-action plan.


@pytest.mark.parametrize('kind',['gcn','graphsage'])
def test_remaining_one_action_terminates_without_extra_preview(kind):
    calls = []
    env = LocalV2Env(model(kind),lambda m:calls.append(1) or .8,.44,penalty_lambda=2.,max_steps=2,greedy=True)
    env.step(env.legal_actions()[0])
    assert not env.terminal and env.state()[2] == 1
    old = len(calls)
    plan = LookaheadPlanner().plan(env)
    assert len(calls)-old == plan.evaluations == len(env.legal_actions())
    assert all(len(p['actions']) == 1 for p in plan.decision['candidate_paths'])
    assert plan.next_env.terminal and plan.next_env.terminal_reason == 'max_steps'
    with pytest.raises(ValueError):
        plan.next_env.step((0,1))
    with pytest.raises(ValueError,match='terminal'):
        LookaheadPlanner().plan(plan.next_env)


@pytest.mark.parametrize('kind',['gcn','graphsage'])
@pytest.mark.parametrize('target,horizon',[(.44,1),(.999,20)])
def test_unreachable_band_terminates_with_actual_gap_like_greedy(kind,target,horizon):
    calls = []
    r = run_lookahead(model(kind),lambda m:calls.append(1) or .8,
                      target,max_steps=horizon)
    assert len(calls) == r.statistics['validation_evaluations']
    assert r.statistics['candidate_evaluations'] > 0
    assert r.statistics['target_violation']
    assert r.statistics['terminal_reason'] in ('max_steps','no_valid_actions')
    assert r.statistics['target_gap'] > .01


@pytest.mark.parametrize('kind',['gcn','graphsage'])
@pytest.mark.parametrize('depth',[1,2])
def test_zero_target_follows_native_greedy_threshold_behavior(kind,depth):
    m = model(kind)
    evaluate = lambda _: .8
    actual = run_lookahead(m, evaluate, 0., depth=depth)
    expected = run_greedy(m, evaluate, target=0., penalty_lambda=2.)
    # In the historical Greedy environment target=0 still performs one deletion.
    assert actual.trace == expected.trace
    assert actual.kept_indices == expected.kept_indices
    assert actual.statistics['terminal_reason'] == 'target_reached'
    assert len(actual.trace) == 1


@pytest.mark.parametrize('kind',['gcn','graphsage'])
@pytest.mark.parametrize('depth',[1,2])
def test_overshoot_is_legal_explicit_and_penalized_once(kind,depth):
    calls = []
    r = run_lookahead(model(kind),lambda m:calls.append(1) or .8,.01,depth=depth)
    assert len(r.trace) == 1 and r.statistics['threshold_reached']
    assert r.statistics['signed_target_deviation'] > .01
    assert r.statistics['overshoot'] == r.statistics['signed_target_deviation']
    assert r.statistics['overshoot_beyond_tolerance'] and r.statistics['target_violation']
    penalty = -2*(r.statistics['overshoot']-.01)
    assert r.trace[-1]['terminal_penalty'] == pytest.approx(penalty)
    assert r.trace[-1]['reward'] == pytest.approx(.6*r.trace[-1]['sparsity']+penalty)
    assert len(calls) == r.statistics['validation_evaluations']
    assert all(len(p['actions']) == 1 for p in r.decisions[0]['candidate_paths'])


@pytest.mark.parametrize('kind',['gcn','graphsage'])
@pytest.mark.parametrize('depth',[1,2])
def test_initial_no_actions_matches_native_penalty_and_actual_counts(kind,depth):
    m = build_model(kind,in_channels=5,hidden_channels=4,out_channels=3,num_layers=3,dropout=.5).eval()
    r = run_lookahead(m,lambda m:.8,.3,depth=depth)
    native = run_greedy(m,lambda m:.8,target=.3,penalty_lambda=2.)
    assert r.trace == native.trace and r.kept_indices == native.kept_indices
    assert r.statistics['validation_evaluations'] == 1
    assert r.statistics['candidate_evaluations'] == 0 and not r.decisions
    assert r.statistics['terminal_reason'] == 'no_valid_actions'
    assert not r.statistics['threshold_reached'] and r.statistics['overshoot'] == 0


@pytest.mark.parametrize('kind',['gcn','graphsage'])
def test_actual_evaluation_accounting_and_concrete_frontier_reuse(kind):
    calls = []
    r = run_lookahead(model(kind),lambda m:calls.append(tuple(m.hidden_channel_dims)) or accuracy(m.hidden_channel_dims),.44,max_steps=3)
    nodes = [n for d in r.decisions for n in d['candidate_transitions']]
    assert len(calls) == 1+r.statistics['candidate_evaluations'] == r.statistics['validation_evaluations']
    assert sum(not n['evaluation_reused'] for n in nodes) == r.statistics['candidate_evaluations']
    assert sum(n['evaluation_reused'] for n in nodes) == r.statistics['reused_evaluations'] > 0
    assert len(nodes) == r.statistics['explored_candidate_transitions']
    assert sum(len(d['candidate_paths']) for d in r.decisions) == r.statistics['candidate_paths']
    assert r.statistics['search_seconds'] >= 0


@pytest.mark.parametrize('kind',['gcn','graphsage'])
def test_equal_aggregate_key_on_another_parent_does_not_enable_cache(kind):
    env = LocalV2Env(model(kind),lambda m:.8,.44,penalty_lambda=2.,max_steps=3,greedy=True)
    planner = LookaheadPlanner()
    chosen = planner.plan(env).next_env
    assert not chosen.terminal
    different_parent = copy.copy(chosen)
    assert different_parent.state() == chosen.state()
    plan = planner.plan(different_parent)
    assert plan.cache_hits == 0
    assert all(not n['evaluation_reused'] for n in plan.decision['candidate_transitions'])


@pytest.mark.parametrize('kind',['gcn','graphsage'])
def test_cache_invalidates_if_the_same_parent_changes_after_planning(kind):
    env = LocalV2Env(model(kind),lambda m:.8,.44,penalty_lambda=2.,max_steps=3,greedy=True)
    planner = LookaheadPlanner()
    chosen = planner.plan(env).next_env
    assert not chosen.terminal
    chosen.max_steps += 1  # Same object/model/mask, different legal horizon.
    plan = planner.plan(chosen)
    assert plan.cache_hits == 0


@pytest.mark.parametrize('kind',['gcn','graphsage'])
def test_repeatability_including_channel_identities_and_no_rng_consumption(kind):
    m = model(kind)
    torch_state, py_state = torch.random.get_rng_state().clone(), random.getstate()
    a = run_lookahead(m,lambda m:accuracy(m.hidden_channel_dims),.44,max_steps=3)
    b = run_lookahead(m,lambda m:accuracy(m.hidden_channel_dims),.44,max_steps=3)
    assert a.trace == b.trace and a.kept_indices == b.kept_indices
    assert without_times(a.decisions) == without_times(b.decisions)
    assert torch.equal(torch_state,torch.random.get_rng_state())
    assert py_state == random.getstate()


@pytest.mark.parametrize('depth',[0,3,True,1.0])
def test_invalid_depth_rejected(depth):
    with pytest.raises(ValueError,match='depth'):
        LookaheadPlanner(depth)


def test_gamma_and_wrong_environment_reward_menu_are_rejected():
    with pytest.raises(ValueError,match='gamma'):
        LookaheadPlanner(gamma=0.)
    non_greedy = LocalV2Env(model('gcn'),lambda m:.8,.44,penalty_lambda=2.)
    with pytest.raises(ValueError,match='Greedy threshold'):
        LookaheadPlanner().plan(non_greedy)
    for greedy in (False,True):
        constrained = V4Env(model('gcn'),lambda m:.8,.44,Settings(),penalty_lambda=2.,greedy=greedy)
        with pytest.raises(ValueError,match='Greedy threshold'):
            LookaheadPlanner().plan(constrained)


class TreeEnv(LocalV2Env):
    """Independent reward tree forces a change to the previously planned second action."""
    tree = {
        'root': [((0,1),'A',0.), ((1,1),'B',-10.)],
        'A': [((0,1),'Aa',5.), ((1,1),'Ab',4.)],
        'Aa': [((0,1),'end_bad',-100.)],
        'Ab': [((0,1),'end_good',10.)],
    }

    def __init__(self,node='root',trace=None):
        self.node, self.trace = node, trace or []
        self.model = torch.nn.Identity()
        self.steps = len(self.trace)
        self.greedy = True
        self.terminal = node not in self.tree
        self.terminal_reason = 'target_band' if self.terminal else None
        self.scoring_seconds = self.surgery_seconds = 0.
        self.original_kept_ids = [[0,1,2,3],[0,1,2,3]]

    def state(self):
        return ('tree',self.node,3-self.steps,self.terminal)

    def legal_actions(self):
        assert not self.terminal, 'Planner expanded a terminal node'
        return [a for a,_,_ in self.tree[self.node]]

    def preview(self,action):
        node,reward = next((n,r) for a,n,r in self.tree[self.node] if a==action)
        return type(self)(node,self.trace+[dict(action=key(action),reward=reward,
            terminal_penalty=-2. if node=='B' else 0.)])


def test_replanning_executes_only_first_and_can_reject_old_second_action(monkeypatch):
    monkeypatch.setattr('gnn_pruning.pruning.lookahead.LocalV2Env',TreeEnv)
    root = TreeEnv()
    planner = LookaheadPlanner()
    first = planner.plan(root)
    assert first.decision['best_planned_path'] == ['prune:0:1','prune:0:1']
    assert first.next_env.node == 'A' and first.next_env.steps == 1
    terminal = next(p for p in first.decision['candidate_paths'] if p['actions']==['prune:1:1'])
    assert terminal['score'] == -10.  # Already includes its penalty; no extra charge.
    second = planner.plan(first.next_env)
    assert second.decision['selected_action'] == 'prune:1:1'
    assert second.next_env.node == 'Ab' and second.cache_hits == 2
    final = planner.plan(second.next_env)
    assert final.next_env.node == 'end_good' and final.evaluations == 0 and final.cache_hits == 1
    assert root.trace == [] and root.steps == 0


@pytest.mark.parametrize('depth',[1,2])
def test_exact_path_ties_choose_first_root_and_first_continuation(depth,monkeypatch):
    class TiedTree(TreeEnv):
        tree = {'root':[((0,1),'A',0.),((1,1),'B',0.)],
                'A':[((0,1),'Aa',0.),((1,1),'Ab',0.)]}
    root = TiedTree()
    monkeypatch.setattr('gnn_pruning.pruning.lookahead.LocalV2Env',TiedTree)
    plan = LookaheadPlanner(depth).plan(root)
    assert plan.decision['selected_action'] == 'prune:0:1'
    assert plan.decision['best_planned_path'] == ['prune:0:1']+(['prune:0:1'] if depth==2 else [])

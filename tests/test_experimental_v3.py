"""Independent finite-state and transition checks for separate V3 ablations."""
import copy
from fractions import Fraction
import random

import pytest
import torch

from gnn_pruning.models import build_model
from gnn_pruning.pruning import adaptive_v2 as r2
from gnn_pruning.pruning.experimental_v3 import (
    WidthReachability, V3Env, action_value, choose_action, run_variant,
)


def model(kind='gcn', widths=8, features=5):
    torch.manual_seed(42)
    return build_model(kind, in_channels=features, hidden_channels=widths,
                       out_channels=3 if features==5 else 7, num_layers=3, dropout=.5).eval()


def exhaustive_reachable(space, widths, remaining):
    frontier={tuple(widths)}
    for _ in range(remaining+1):
        if any(space.in_band(w) for w in frontier):return True
        frontier={tuple(w[j]-k if j==layer else w[j] for j in range(2))
                  for w in frontier for layer in range(2) for k in (1,2,4,8)
                  if w[layer]-k>=space.minimum}
    return False


@pytest.mark.parametrize('kind',['gcn','graphsage'])
def test_exact_reachability_matches_enumerated_paths(kind):
    m=model(kind)
    for target in (.1,.3,.5,.8):
        space=WidthReachability.from_model(m,target,.02,4)
        for a in range(4,9):
            for b in range(4,9):
                for horizon in range(4):
                    assert space.can_reach((a,b),horizon)==exhaustive_reachable(space,(a,b),horizon)
                    # Count independently by rebuilding physical architecture.
                    rebuilt=build_model(kind,in_channels=5,hidden_channels=[a,b],out_channels=3,num_layers=3,dropout=.5)
                    assert space.count((a,b))==sum(p.numel() for p in rebuilt.parameters())


@pytest.mark.parametrize('kind',['gcn','graphsage'])
def test_cora_requires_four_actions_for_30_band(kind):
    space=WidthReachability.from_model(model(kind,32,1433),.3,.01,4)
    assert not space.can_reach((32,32),3)
    assert space.can_reach((32,32),4)
    assert space.can_reach((32,32),20)


def test_initial_infeasible_target_is_explicit_without_fallback():
    m=build_model('gcn',in_channels=3703,hidden_channels=32,out_channels=6,num_layers=3,dropout=.5)
    e=V3Env(m,lambda _: .8,target=.3,penalty_lambda=2.,variant='q_reachable_v3',max_steps=20)
    assert e.terminal and e.terminal_reason=='initial_target_infeasible'
    assert e.steps==0 and e.legal_actions()==[] and e.trace[-1]['terminal_penalty']==pytest.approx(-.58)


@pytest.mark.parametrize('kind',['gcn','graphsage'])
def test_reachability_keeps_a_feasible_continuation_to_band(kind):
    e=V3Env(model(kind,32,1433),lambda _: .8,target=.3,penalty_lambda=2.,variant='q_reachable_v3',max_steps=20)
    while not e.terminal:
        actions=e.legal_actions();assert actions and 'stop' not in actions
        for layer,k in actions:
            widths=list(e.state()[1]);widths[layer]-=k
            assert e.space.can_reach(widths,20-e.steps-1)
        e.step(actions[-1])
    assert e.space.in_band(e.state()[1]) and e.steps<=20
    assert e.trace[-1]['terminal_reason'] in ('target_band','max_steps','no_valid_actions')


def test_hard_crossing_terminal_penalty_once():
    e=V3Env(model(),lambda _: .8,target=.01,penalty_lambda=2.,variant='q_target_stop_v3')
    _,reward,done,info=e.step((0,1));assert done and info['terminal_reason']=='target_threshold'
    assert reward==pytest.approx(.6*e.sparsity-2*max(0,abs(e.sparsity-.01)-.01))
    with pytest.raises(ValueError):e.step('stop')


def test_exact_stop_value_and_conservative_unseen_values():
    space=WidthReachability.from_model(model(),.3,.01,4)
    s=('local_v2',(7,8),19,80,False)
    exact=-2*max(0,abs(space.compression(s[1])-.3)-.01)
    assert action_value({},s,'stop','q_stop_exact_v3',space,2.,.9)==pytest.approx(exact)
    assert action_value({s:{'stop':999}},s,'stop','q_stop_exact_v3',space,2.,.9)==pytest.approx(exact)
    bound=action_value({},s,'prune:0:1','q_conservative_v3',space,2.,.9)
    worst_return=-.4*sum(.9**j for j in range(19))-2*max(.3,.7)
    assert bound<=worst_return+.03
    assert action_value({s:{'prune:0:1':.2}},s,'prune:0:1','q_conservative_v3',space,2.,.9)==.2


def test_random_ties_seeded_reproducible_and_not_first_biased():
    space=WidthReachability.from_model(model(),.3,.01,4);s=('local_v2',(8,8),20,80,False);actions=[(0,1),(1,1)]
    def draw(seed):
        rng=random.Random(seed)
        return [choose_action({},s,actions,rng,0.,'q_random_ties_v3',space,2.,.9) for _ in range(1000)]
    x=draw(501);assert x==draw(501) and 400<x.count((0,1))<600
    table={s:{'prune:1:1':1.}}
    assert choose_action(table,s,actions,random.Random(1),0.,'q_random_ties_v3',space,2.,.9)==(1,1)


@pytest.mark.parametrize('kind',['gcn','graphsage'])
def test_r2_reference_is_exact_original_routine(kind):
    m=model(kind);before=copy.deepcopy(m.state_dict());settings=dict(target=.3,penalty_lambda=2.,episodes=3,max_steps=3,seed=5001)
    a=r2.run_q_learning(m,lambda _: .8,**settings)
    b=run_variant(m,lambda _: .8,variant='r2_reference',**settings)
    assert a.trace==b.trace and a.training_trace==b.training_trace and a.q_table==b.q_table
    for k,v in before.items():assert torch.equal(v,m.state_dict()[k]) and torch.equal(a.model.state_dict()[k],b.model.state_dict()[k])


@pytest.mark.parametrize('kind',['gcn','graphsage'])
@pytest.mark.parametrize('variant',['q_stop_exact_v3','q_conservative_v3','q_random_ties_v3','q_target_stop_v3','q_reachable_v3','random_reachable_v3','greedy_reachable_v3'])
def test_variants_reset_and_return_physical_final_models(kind,variant):
    m=model(kind,32,1433)
    result=run_variant(m,lambda _: .8,variant=variant,target=.3,penalty_lambda=2.,episodes=2,max_steps=20,seed=5002)
    assert sum(p.numel() for p in result.model.parameters())<sum(p.numel() for p in m.parameters())
    assert result.trace[-1]['original_kept_ids']==result.kept_indices
    if 'reachable' in variant:
        achieved=1-sum(p.numel() for p in result.model.parameters())/sum(p.numel() for p in m.parameters())
        assert abs(achieved-.3)<=.01
    if variant.startswith('q_'):assert sum(x.get('event')=='reset' for x in result.training_trace)==2


def test_conservative_terminal_bellman_uses_the_unseen_bound_not_zero():
    m=model();space=WidthReachability.from_model(m,.3,.01,4)
    result=run_variant(m,lambda _: .8,variant='q_conservative_v3',target=.3,
                       penalty_lambda=2.,episodes=1,max_steps=1,epsilon=0.,seed=101)
    s=('local_v2',(8,8),1,80,False)
    event=next(e for e in result.training_trace if e.get('event')!='reset')
    prior=-.4-2*(.7-.01)
    assert result.q_table[s]['prune:0:1']==pytest.approx(prior+.1*(event['reward']-prior))


def test_analytical_stop_saved_exactly_for_visited_nonterminal_state():
    m=model();space=WidthReachability.from_model(m,.3,.01,4)
    result=run_variant(m,lambda _: .8,variant='q_stop_exact_v3',target=.3,
                       penalty_lambda=2.,episodes=1,max_steps=2,epsilon=0.,seed=101)
    states=[s for s in result.q_table if s[2]==1 and not s[-1]]
    assert states
    for s in states:
        expected=-2*max(0,abs(space.compression(s[1])-.3)-.01)
        assert result.q_table[s]['stop']==pytest.approx(expected)


def test_reachability_filter_never_calls_validation_or_changes_state():
    calls=[]
    def evaluate(m):calls.append(1);return .8
    e=V3Env(model('gcn',32,1433),evaluate,target=.3,penalty_lambda=2.,variant='q_reachable_v3',max_steps=4)
    before=e.state();n=len(calls)
    for _ in range(10):assert e.legal_actions()
    assert before==e.state() and len(calls)==n

"""Independent menu-DAG, controlled-factor, reward, RNG and V3 regressions."""
import copy
import math
import random
from dataclasses import asdict
import pytest
import torch
from gnn_pruning.models import build_model
from gnn_pruning.pruning.experimental_v3 import WidthReachability, run_variant as run_v3
from gnn_pruning.pruning.experimental_v4 import (PLANS, Q_VARIANTS, Settings, V4Env,
    MenuReachability, action_sizes, epsilon_at, bellman, run_variant, select)


def model(kind='gcn',width=8,features=5):
    torch.manual_seed(17)
    return build_model(kind,in_channels=features,hidden_channels=width,out_channels=3 if features==5 else 7,num_layers=3,dropout=.5).eval()


def sizes(menu,width):
    return {'standard':(1,2,4,8),'one_two':(1,2),'percent':tuple(sorted({math.ceil(.05*width),math.ceil(.1*width)}))}[menu]


def brute(space,menu,start,horizon):
    frontier={start}
    for _ in range(horizon+1):
        if any(space.in_band(w) for w in frontier):return True
        frontier={tuple(w[i]-k if i==layer else w[i] for i in (0,1))
                  for w in frontier for layer in (0,1) for k in sizes(menu,w[layer])
                  if w[layer]-k>=space.minimum}
    return False


@pytest.mark.parametrize('menu',['standard','one_two','percent'])
@pytest.mark.parametrize('kind',['gcn','graphsage'])
def test_reachability_matches_independent_width_bfs(kind,menu):
    m=model(kind)
    for target in (.1,.3,.5,.8):
        space=WidthReachability.from_model(m,target,.02,4);oracle=MenuReachability(space,menu)
        for a in range(4,9):
            for b in range(4,9):
                for horizon in range(4):assert oracle.can_reach((a,b),horizon)==brute(space,menu,(a,b),horizon)


def test_percentage_rounding_and_width_dependence():
    for width in range(4,33):assert action_sizes('percent',width)==sizes('percent',width)
    assert action_sizes('percent',32)==(2,4)
    assert action_sizes('percent',30)==(2,3)
    assert action_sizes('percent',10)==(1,)
    oracle=MenuReachability(WidthReachability.from_model(model('gcn',32,1433),.3), 'percent')
    assert oracle.distance(32,23)==3  # 32 -> 28 (4) -> 25 (3) -> 23 (2)
    assert oracle.distance(10,4)==6


@pytest.mark.parametrize('kind',['gcn','graphsage'])
@pytest.mark.parametrize('menu',['standard','one_two','percent'])
def test_cora_targets_reachable_and_filter_preserves_continuation(kind,menu):
    calls=[]
    def evaluate(_):calls.append(1);return .8
    for target in (.3,.5):
        e=V4Env(model(kind,32,1433),evaluate,target,Settings(menu=menu),penalty_lambda=2.)
        while not e.terminal:
            before=len(calls);actions=e.legal_actions();assert len(calls)==before and actions
            for layer,k in actions:
                future=list(e.state()[1]);future[layer]-=k
                assert k in sizes(menu,e.state()[1][layer])
                assert e.oracle.can_reach(tuple(future),20-e.steps-1)
            e.step(actions[-1])
        assert e.space.in_band(e.state()[1]) and e.steps<=20
        assert sum(x['terminal_reason'] is not None for x in e.trace)==1
        with pytest.raises(ValueError):e.step((0,1))


def test_infeasible_target_fails_before_label_access():
    calls=[]
    with pytest.raises(ValueError,match='initial_target_infeasible'):
        V4Env(model('gcn',32,1433),lambda m:calls.append(1),.3,Settings(menu='one_two'),penalty_lambda=2.,max_steps=1)
    assert calls==[]


@pytest.mark.parametrize('kind',['gcn','graphsage'])
def test_reference_exact_v3_traces_q_weights_and_rng(kind):
    m=model(kind,32,1433);settings=dict(target=.3,seed=5001,max_steps=20)
    a=run_v3(m,lambda _: .8,'q_reachable_v3',penalty_lambda=2.,episodes=4,**settings)
    b=run_variant(m,lambda _: .8,'q_v4_reference',episodes_override=4,**settings)
    assert a.trace==b.trace and a.training_trace==b.training_trace and a.q_table==b.q_table
    assert a.kept_indices==b.kept_indices
    assert a.statistics['validation_evaluations']==b.statistics['validation_evaluations']
    for name,value in a.model.state_dict().items():assert torch.equal(value,b.model.state_dict()[name])


def test_only_predeclared_factors_change():
    base=asdict(PLANS['q_v4_reference'])
    expected={'q_v4_menu12':{'menu'},'q_v4_percent':{'menu'},'q_v4_300':{'episodes'},
              'q_v4_random_ties':{'random_ties'},'q_v4_incremental_drop':{'incremental_drop'},'q_v4_gamma0':{'gamma'}}
    for variant,changed in expected.items():assert {k for k,v in asdict(PLANS[variant]).items() if v!=base[k]}==changed
    b,c=asdict(PLANS['q_v4_300']),asdict(PLANS['q_v4_decay300'])
    assert {k for k in b if b[k]!=c[k]}=={'epsilon','decay'} and len(Q_VARIANTS)==7
    assert epsilon_at(PLANS['q_v4_decay300'],0)==.4
    assert epsilon_at(PLANS['q_v4_decay300'],299)==pytest.approx(.05)


def test_gamma_zero_has_no_future_credit_and_terminal_once():
    s=('s',);ns=('n',);table={ns:{'prune:0:1':10.}}
    before,after,future=bellman(table,s,(0,1),.5,ns,[(0,1)],False,0.)
    assert before==0 and after==.05 and future==10.
    table={ns:{'prune:0:1':10.}}
    assert bellman(table,s,(0,1),.5,ns,[(0,1)],False,.9)[1]==pytest.approx(.95)
    assert bellman({},s,(0,1),-.2,ns,[(0,1)],True,.9)[1]==pytest.approx(-.02)


def test_incremental_drop_reward_telescopes_including_recovery():
    def evaluate(m):return .8-.004*(8-m.hidden_channel_dims[0])+.002*(8-m.hidden_channel_dims[1])
    e=V4Env(model(),evaluate,.3,Settings(incremental_drop=True),penalty_lambda=2.,tolerance=.02)
    e.step((0,1));e.step((1,1))
    while not e.terminal:e.step(e.legal_actions()[-1])
    expected=.6*e.sparsity-.4*max(0,e.dense_accuracy-e.accuracy)+sum(t['terminal_penalty'] for t in e.trace)
    assert sum(t['reward'] for t in e.trace)==pytest.approx(expected)
    previous=0.
    for t in e.trace:
        assert t['previous_accuracy_drop']==pytest.approx(previous)
        assert t['accuracy_drop_difference']==pytest.approx(t['accuracy_drop']-previous)
        previous=t['accuracy_drop']
    assert any(t['accuracy_drop_difference']<0 for t in e.trace)


def test_snapshots_do_not_perturb_any_training_or_deployment_rng():
    m=model('gcn',32,1433);args=dict(target=.3,seed=5002,episodes_override=4)
    before=torch.random.get_rng_state().clone()
    a=run_variant(m,lambda _: .8,'q_v4_random_ties',snapshots=(),**args)
    b=run_variant(m,lambda _: .8,'q_v4_random_ties',snapshots=(1,2,4),**args)
    assert a.trace==b.trace and a.training_trace==b.training_trace and a.q_table==b.q_table and a.decisions==b.decisions
    assert set(b.snapshots)=={1,2,4} and torch.equal(before,torch.random.get_rng_state())


def test_random_ties_only_draw_for_equal_maxima():
    s=('s',);actions=[(0,1),(1,1)];rng=random.Random(101)
    selected=[select({},s,actions,rng,0.,True)[0] for _ in range(500)]
    assert 180<selected.count((0,1))<320
    rng=random.Random(101);state=rng.getstate()
    assert select({s:{'prune:0:1':1}},s,actions,rng,0.,True)[0]==(0,1)
    assert state==rng.getstate()


@pytest.mark.parametrize('variant',Q_VARIANTS)
def test_all_variants_are_physical_validation_only_and_keep_dense(variant):
    m=model('gcn',32,1433);before=copy.deepcopy(m.state_dict())
    r=run_variant(m,lambda _: .8,variant,.3,episodes_override=2)
    assert r.trace[-1]['terminal_reason'] and r.statistics['validation_evaluations']>0
    assert len(r.kept_indices[0])>=4 and len(r.kept_indices[1])>=4
    for k,v in before.items():assert torch.equal(v,m.state_dict()[k])

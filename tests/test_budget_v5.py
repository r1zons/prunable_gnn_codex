"""Independent budget/depth/forward oracles for the new campaign namespace."""
import copy
import itertools
from fractions import Fraction

import pytest
import torch
from torch_geometric.data import Data

from gnn_pruning.models import build_model
from gnn_pruning.pruning.budget_v5 import Budget, BudgetEnv, METHODS, plan, run, static_plan
from gnn_pruning.pruning.static_v2 import apply_static_plan, compute_scores, removed_parameter_score, channel_rankings
from gnn_pruning.surgery.local_v2 import prune_hidden_channels_local_v2


def network(kind, depth, width=8):
    torch.manual_seed(29)
    return build_model(kind,in_channels=5,hidden_channels=width,out_channels=3,num_layers=depth,dropout=.5).eval()


def graph():
    g=torch.Generator().manual_seed(91)
    return Data(x=torch.randn(12,5,generator=g),y=torch.arange(12)%3,
                edge_index=torch.stack([torch.arange(12),torch.arange(12).roll(1)]))


def count(kind, widths):
    dims=[5,*widths,3]
    return sum((2 if kind=='graphsage' else 1)*a*b+b for a,b in zip(dims,dims[1:]))


def goals(kind, widths, target, eps):
    p0=count(kind,widths)
    allw=list(itertools.product(*(range(4,w+1) for w in widths)))
    allw=[w for w in allw if count(kind,w)<p0]
    gap=lambda w:abs(Fraction(p0-count(kind,w),p0)-Fraction(str(target)))
    band=[w for w in allw if gap(w)<=Fraction(str(eps))]
    return band or [w for w in allw if gap(w)==min(map(gap,allw))]


@pytest.mark.parametrize('kind',['gcn','graphsage'])
@pytest.mark.parametrize('depth',[2,3,4])
def test_surgery_all_hidden_layers_masked_reference_parent_reload(kind,depth):
    dense=network(kind,depth);data=graph();before=copy.deepcopy(dense.state_dict())
    masks=[list(range(0,8,2)) for _ in range(depth-1)]
    ref=copy.deepcopy(dense);physical=dense
    for layer,ids in enumerate(masks):
        def mask(_module,_inputs,out,ids=ids):
            value=torch.zeros_like(out);value[:,ids]=out[:,ids];return value
        ref.convs[layer].register_forward_hook(mask)
        physical=prune_hidden_channels_local_v2(physical,layer,ids,4)
    assert sum(p.numel() for p in physical.parameters())==count(kind,[4]*(depth-1))
    assert torch.allclose(physical(data),ref(data),atol=1e-6,rtol=1e-5)
    cfg=physical.export_architecture_config();cfg.pop('name')
    loaded=build_model(kind,**cfg).eval();loaded.load_state_dict(physical.state_dict(),strict=True)
    assert torch.equal(loaded(data),physical(data))
    assert all(torch.equal(v,dense.state_dict()[k]) for k,v in before.items())


@pytest.mark.parametrize('kind',['gcn','graphsage'])
@pytest.mark.parametrize('depth',[2,3,4])
@pytest.mark.parametrize('target',[.3,.5])
def test_geometry_matches_independent_all_descendants(kind,depth,target):
    model=network(kind,depth);b=Budget(model,target);expected=goals(kind,(8,)*(depth-1),target,.01)
    actual=[w for w in itertools.product(range(4,9),repeat=depth-1) if b.goal(w)]
    assert actual==expected
    assert b.horizon==4*(depth-1)
    for w in itertools.product(range(4,9),repeat=depth-1):
        assert b.count(w)==count(kind,w)
        assert b.reachable(w,b.horizon)==any(all(a<=c for a,c in zip(g,w)) for g in expected)
    with pytest.raises(ValueError,match='horizon'):
        b.reachable(b.original,0)


@pytest.mark.parametrize('kind',['gcn','graphsage'])
@pytest.mark.parametrize('depth',[2,3,4])
@pytest.mark.parametrize('method',METHODS)
def test_all_seven_methods_physical_reproducible_band_and_accounting(kind,depth,method):
    m=network(kind,depth);data=graph();data.y[8:]=-1
    calls=[]
    def evaluate(candidate):
        assert (data.y[8:]==-1).all()
        calls.append(1)
        with torch.no_grad():return float((candidate(data)[:4].argmax(-1)==data.y[:4]).float().mean())
    b=Budget(m,.3)
    if method.startswith('static'):
        criterion=method.removeprefix('static_')
        p=static_plan(m,criterion,b,101,data,torch.tensor([4,5,6,7]))
        final=apply_static_plan(m,p);ids=p.kept_indices
        assert not calls
    else:
        r=run(m,evaluate,method,.3,episodes=2);final=r.model;ids=r.kept_indices
        assert len(calls)==r.statistics['validation_evaluations']
        assert r.statistics['complete'] and r.statistics['terminal_reason'] in ('target_band','closest_feasible')
        assert sum(x['terminal_reason'] is not None for x in r.trace)==1
        assert len(r.trace)<=b.horizon and all(x['action']!='stop' for x in r.trace)
        r2=run(m,evaluate,method,.3,episodes=2)
        assert r.trace==r2.trace and r.kept_indices==r2.kept_indices and r.q_table==r2.q_table
        assert all(torch.equal(v,r2.model.state_dict()[k]) for k,v in final.state_dict().items())
    widths=tuple(map(len,ids))
    assert b.goal(widths)
    assert sum(p.numel() for p in final.parameters())==count(kind,widths)<b.p0
    assert final(data).shape==(12,3)


@pytest.mark.parametrize('kind',['gcn','graphsage'])
def test_static_prefix_objective_matches_exhaustive_unique_union(kind):
    m=network(kind,4);b=Budget(m,.3)
    p=static_plan(m,'magnitude',b,101)
    scores=compute_scores(m,'magnitude');ranks=channel_rankings(m,scores)
    choices=[]
    for w in goals(kind,(8,8,8),.3,.01):
        ids=[sorted(r[:k]) for r,k in zip(ranks,w)]
        choices.append((removed_parameter_score(m,scores,ids),w))
    expected=min(choices)
    assert tuple(p.widths)==expected[1]
    assert p.removed_score==pytest.approx(expected[0],abs=1e-10)


@pytest.mark.parametrize('kind',['gcn','graphsage'])
@pytest.mark.parametrize('depth',[2,3,4])
def test_real_forward_depth2_independent_path_oracle_replans(kind,depth):
    m=network(kind,depth);data=graph()
    def evaluate(model):
        with torch.no_grad():return float((model(data).argmax(-1)==data.y).float().mean())
    target=.5;goal=goals(kind,(8,)*(depth-1),target,.01);p0=count(kind,(8,)*(depth-1));denseacc=evaluate(m)
    def legal(model):
        widths=model.hidden_channel_dims
        result=[]
        for i,w in enumerate(widths):
            for k in (1,2,4,8):
                nxt=[x-(k if j==i else 0) for j,x in enumerate(widths)]
                if w-k>=4 and any(all(x<=y for x,y in zip(g,nxt)) for g in goal):result.append((i,k))
        return result
    def transition(parent,action):
        i,k=action;values=torch.zeros(parent.hidden_channel_dims[i],dtype=torch.float64)
        for name,param in parent.named_parameters():
            layer=int(name.split('.')[1]);value=param.detach().abs()
            if layer==i:values+=(value.sum(1) if value.ndim==2 else value).double()
            elif layer==i+1 and value.ndim==2:values+=value.sum(0).double()
        rank=sorted(range(len(values)),key=lambda c:(-float(values[c]),c))
        child=prune_hidden_channels_local_v2(parent,i,sorted(rank[:-k]),4)
        s=1-count(kind,child.hidden_channel_dims)/p0;old=1-count(kind,parent.hidden_channel_dims)/p0
        terminal=tuple(child.hidden_channel_dims) in goal
        reward=.6*(s-old)-.4*max(0.,denseacc-evaluate(child))
        if terminal:reward-=2*max(0.,abs(s-target)-.01)
        return child,reward,terminal
    env=BudgetEnv(m,evaluate,Budget(m,target));cache=None
    while not env.terminal:
        expected=[]
        parent=env.model;parent_state=env.state();before=copy.deepcopy(parent.state_dict())
        for a in legal(parent):
            c,r1,terminal=transition(parent,a)
            if terminal:expected.append(([a],r1))
            else:
                for b in legal(c):
                    _,r2,_=transition(c,b);expected.append(([a,b],r1+.9*r2))
        best=max(expected,key=lambda p:p[1])
        child,cache,decision,new,hits=plan(env,2,cache)
        assert [p['score'] for p in decision['candidate_paths']]==pytest.approx([r for _,r in expected])
        assert decision['selected_action']==f'prune:{best[0][0][0]}:{best[0][0][1]}'
        assert len(child.trace)==len(env.trace)+1 and env.state()==parent_state
        assert all(torch.equal(v,parent.state_dict()[k]) for k,v in before.items())
        env=child


@pytest.mark.parametrize('kind',['gcn','graphsage'])
def test_depth1_is_new_greedy_reference_exact_ties_and_terminal_penalty(kind):
    m=network(kind,3)
    for p in m.parameters():p.data.zero_()
    b=Budget(m,.333,tolerance=0)
    assert b.fallback
    a=run(m,lambda m:.8,'adaptive_greedy',.333,tolerance=0)
    c=run(m,lambda m:.8,'lookahead_depth2',.333,tolerance=0,depth=1)
    assert a.trace==c.trace and a.kept_indices==c.kept_indices
    assert a.statistics['validation_evaluations']==c.statistics['validation_evaluations']
    assert a.trace[-1]['terminal_penalty']<0
    assert all(x['terminal_penalty']==0 for x in a.trace[:-1])
    env=BudgetEnv(m,lambda m:.8,b)
    while not env.terminal:env.step(env.legal_actions()[0])
    with pytest.raises(ValueError):env.step((0,1))


@pytest.mark.parametrize('kind',['gcn','graphsage'])
def test_q_target_conditioning_retained_ids_no_stop_and_bellman(kind):
    m=network(kind,3,width=12)
    a=run(m,lambda m:.8,'tabular_q',.3,episodes=2)
    b=run(m,lambda m:.8,'tabular_q',.5,episodes=2)
    assert a.kept_indices!=b.kept_indices
    assert a.training_trace[0]['state'][1]!=b.training_trace[0]['state'][1]
    first=a.training_trace[0]
    assert first['q_before']==0 and first['q_after']==pytest.approx(.1*first['reward'])
    env=BudgetEnv(m,lambda m:.8,Budget(m,.3));other=copy.copy(env)
    other.kept=[list(reversed(ids)) for ids in env.kept]
    assert env.state()!=other.state()
    assert all('stop' not in x['legal_actions'] for x in a.training_trace)

"""Exact inference reuse must preserve every decision and all Q updates."""
import copy

import pytest
import torch

from gnn_pruning.experiments.frozen_validation import FrozenAccuracy
from gnn_pruning.pruning.budget_v5 import run
from test_budget_v5 import graph, network


@pytest.mark.parametrize('kind',['gcn','graphsage'])
@pytest.mark.parametrize('depth',[2,3,4])
@pytest.mark.parametrize('method',['sequential_random','adaptive_greedy','lookahead_depth2','tabular_q'])
def test_cached_and_uncached_actions_masks_rewards_tables_and_tensors_exact(kind,depth,method):
    dense=network(kind,depth,width=12);data=graph();data.y[8:]=-1
    calls=[]
    def forward(model):
        assert (data.y[8:]==-1).all();calls.append(1)
        with torch.no_grad():return float((model(data)[:8].argmax(-1)==data.y[:8]).float().mean())
    plain=run(dense,forward,method,.5,episodes=3)
    calls.clear();memo=FrozenAccuracy(forward,'pinned-graph-and-split')
    reused=run(dense,memo,method,.5,episodes=3)
    assert plain.trace==reused.trace and plain.decisions==reused.decisions
    assert plain.training_trace==reused.training_trace and plain.q_table==reused.q_table
    assert plain.kept_indices==reused.kept_indices
    assert all(torch.equal(v,reused.model.state_dict()[k]) for k,v in plain.model.state_dict().items())
    assert len(calls)==memo.forward_calls<=memo.requests==reused.statistics['validation_evaluations']
    assert sum(not r['reused'] for r in memo.events)==memo.forward_calls
    if method=='lookahead_depth2':assert memo.forward_calls<memo.requests


def test_cache_requires_exact_tensor_identity_not_same_widths_or_aggregate_state():
    model=network('gcn',3);counter=[]
    def forward(m):counter.append(1);return float(next(m.parameters()).sum())
    memo=FrozenAccuracy(forward,'input-one')
    original=memo(model);assert memo(copy.deepcopy(model))==original and len(counter)==1
    changed=copy.deepcopy(model)
    with torch.no_grad():next(changed.parameters()).flatten()[0].add_(1.)
    assert memo(changed)!=original and len(counter)==2
    fresh=FrozenAccuracy(forward,'input-two');fresh(model)
    assert fresh.forward_calls==1 and len(counter)==3

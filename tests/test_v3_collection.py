"""Independent collector checks, including corrupted evidence rejection."""
import copy
import importlib.util
from pathlib import Path
import pytest
import torch

PATH=Path(__file__).resolve().parents[1]/'research/experiments/local_structured_v3_sprint_20261027/scripts/collect_results.py'
SPEC=importlib.util.spec_from_file_location('v3_collection',PATH)
collector=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(collector)

@pytest.mark.parametrize('kind,p0',[('gcn',47175),('graphsage',94279)])
def test_exact_geometry_and_bfs_oracle(kind,p0):
    assert collector.count(kind,(32,32))==p0
    for target in (.3,.5):
        for start in ((32,32),(23,4),(16,32),(4,4)):
            frontier={start};visited=set(frontier)
            for horizon in range(4):
                expected=any(abs(100*(p0-collector.count(kind,w))-round(target*100)*p0)<=p0 for w in visited)
                assert collector.reachable(kind,target,start,horizon)==expected
                next_frontier=set()
                for widths in frontier:
                    for layer in (0,1):
                        for k in (1,2,4,8):
                            after=list(widths);after[layer]-=k
                            if min(after)>=4:next_frontier.add(tuple(after))
                frontier=next_frontier-visited;visited|=next_frontier

def fixture():
    cfg=dict(variant='q_conservative_v3',model='gcn',target=.3,run_id='synthetic')
    dense=float(torch.tensor(.9,dtype=torch.float32));s=1-collector.count('gcn',(31,32))/47175
    ids=[list(range(31)),list(range(32))];penalty=-2*(.3-s-.01)
    first=dict(action='prune:0:1',successful_deletions=1,widths_before=[32,32],widths_after=[31,32],sparsity=s,
               signed_target_deviation=s-.3,validation_accuracy=dense,accuracy_drop=0.,reward=.6*s,
               terminal_penalty=0.,terminal_reason=None,original_kept_ids=ids)
    last=dict(first,action='stop',widths_before=[31,32],reward=penalty,terminal_penalty=penalty,terminal_reason='agent_stop')
    training=[];qfirst=-.4*sum(.9**j for j in range(20))-2*.69
    unseen19=-.4*sum(.9**j for j in range(19))-2*.69;qstop=unseen19
    for i in range(100):
        training.extend([dict(event='reset',episode=i,widths=[32,32]),dict(first,episode=i),dict(last,episode=i)])
        qfirst+=.1*(first['reward']+.9*max(qstop,unseen19)-qfirst)
        qstop+=.1*(penalty-qstop)
    report=dict(dense_validation={'accuracy':.9},final_validation={'accuracy':dense},original_kept_ids=ids,
                statistics=dict(terminal_reason='agent_stop',q_entries=2,validation_evaluations=202))
    q=[dict(state=['local_v2',[32,32],20,89,False],actions={'prune:0:1':qfirst}),
       dict(state=['local_v2',[31,32],19,89,False],actions={'stop':qstop})]
    return cfg,report,dict(training=training,deployment=[first,last]),q

def test_independent_bellman_replay_accepts_known_sequence():
    audit,episodes=collector.audit_trace(*fixture())
    assert audit['episodes']==100 and audit['q_entries_verified']==2 and len(episodes)==100

def test_q_row_serialization_order_is_not_a_platform_difference():
    *_,q=fixture()
    assert collector.normalize_q(q)==collector.normalize_q(list(reversed(q)))

@pytest.mark.parametrize('corruption',['reward','q','penalty','reset','ids','illegal_stop','terminal'])
def test_corrupted_evidence_is_rejected(corruption):
    cfg,report,trace,q=copy.deepcopy(fixture())
    if corruption=='reward':trace['training'][1]['reward']+=.1
    elif corruption=='q':q[0]['actions']['prune:0:1']+=.1
    elif corruption=='penalty':trace['deployment'][0]['terminal_penalty']=-.2
    elif corruption=='reset':trace['training'][0]['widths']=[31,32]
    elif corruption=='ids':trace['deployment'][0]['original_kept_ids'][0][0]=31
    elif corruption=='illegal_stop':trace['deployment'][0]['action']='stop'
    else:trace['deployment'][-1]['terminal_reason']=None
    with pytest.raises(AssertionError):collector.audit_trace(cfg,report,trace,q)

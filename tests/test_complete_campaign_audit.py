"""Offline audit adversarial tests; no training, pruning strategy or downloads."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch_geometric.data import Data

from gnn_pruning.models import build_model
from gnn_pruning.experiments import complete_audit as a
from gnn_pruning.experiments.complete_audit_trace import projected_state,count_parameters,replay
from gnn_pruning.experiments.complete_collection import inventory_audit,safe_relative


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def seal(directory):
    put(directory/'completed.json',dict(files={p.name:a.digest(p) for p in directory.iterdir() if p.name!='completed.json'}))


@pytest.fixture(params=['gcn','graphsage'])
def artifact(tmp_path,request):
    torch.manual_seed(91);kind=request.param
    dense=build_model(kind,in_channels=5,out_channels=3,hidden_channels=8,num_layers=3,dropout=.5).eval()
    data=Data(x=torch.randn(20,5),edge_index=torch.tensor([list(range(20)),[(i+1)%20 for i in range(20)]]),y=torch.arange(20)%3)
    split=dict(train=list(range(12)),val=list(range(12,16)),test=list(range(16,20)))
    state=dense.state_dict();keeps=[list(range(6)),list(range(7))]
    final=build_model(kind,in_channels=5,out_channels=3,hidden_channels=[6,7],num_layers=3,dropout=.5).eval()
    final.load_state_dict(projected_state(state,keeps,5,3),strict=True)
    before=sum(p.numel() for p in dense.parameters());after=sum(p.numel() for p in final.parameters());sparsity=1-after/before
    proto=dict(source_commit='a'*40,adaptive=dict(tolerance=.01,min_channels=4,max_steps=20,penalty_lambda=2.),benchmark=dict(warmup_passes=5,timed_passes=20))
    run=dict(id='run',method='static_random',implementation='random_local_v2',target=sparsity,policy_seed=101)
    condition=dict(id='condition',dataset='cora',model=kind,seed=101,pilot=True,runs=[run])
    d=dict(checkpoint=dict(model_state_dict=state,model_config=dict(hidden_channels=8),model_name=kind,graph_tensor_sha256='g'*64),
        model=dense,metrics=a.metrics(dense,data,split),split=split,checkpoint_hash='b'*64,split_hash='s'*64,
        parameters=before,reused=False,training_seconds=1.,checkpoint_bytes=100,tensor_hash=a.tensor_identity(state),origin_source=proto['source_commit'])
    directory=tmp_path/'runs/run';directory.mkdir(parents=True)
    architecture=final.export_architecture_config();architecture.pop('name')
    cp=dict(model_name=kind,model_config=architecture,model_state_dict=final.state_dict(),original_kept_ids=keeps,source_checkpoint_sha256=d['checkpoint_hash'])
    torch.save(cp,directory/'pruned.pt');m=a.metrics(final,data,split)
    report=dict(run=run,condition={k:v for k,v in condition.items() if k!='runs'},source_commit=proto['source_commit'],
        checkpoint_sha256=d['checkpoint_hash'],split_sha256=d['split_hash'],graph_tensor_sha256='g'*64,test_labels_exposed=False,
        retained_original_ids=keeps,final_widths=[6,7],parameters_before=before,parameters_after=after,
        checkpoint_bytes=(directory/'pruned.pt').stat().st_size,parameter_bytes=sum(p.numel()*p.element_size() for p in final.parameters()),
        achieved_compression=sparsity,signed_target_deviation=0.,overshoot=0.,outside_symmetric_band=False,
        validation=m['val'],test=m['test'],trace=[],training=[],q_table=[],decisions=[],search_seconds=.2,observed_validation_evaluations=0,
        inference=dict(inference_time_mean_sec=.001,inference_time_std_sec=.0001,inference_time_mean_ms=1.,inference_time_std_ms=.1,inference_warmup_passes=5,inference_timed_passes=20),
        statistics=dict(validation_evaluations=0,terminal_reason='static_target_band',static_plan=dict(criterion='random',kept_indices=keeps,widths=[6,7],parameters_after=after,
            achieved_sparsity=sparsity,candidate_count=24,eligible_candidate_count=1,fallback=False)))
    put(directory/'report.json',report);seal(directory)
    return SimpleNamespace(root=tmp_path,c=condition,run=run,protocol=proto,data=data,dense=d,directory=directory,report=report)


def audit(x,duplicates=()):
    return a.audit_run(x.root,x.c,x.run,x.protocol,x.data,x.dense,duplicates)


def test_valid_real_forward_artifact(artifact):
    row=audit(artifact);assert row['verified'] and row['audit_status']=='PASS'
    assert row['final_metrics_recomputed']=='PASS' and row['tensor_projection']=='PASS'


def test_missing_checkpoint_visible(artifact):
    (artifact.directory/'pruned.pt').unlink();r=audit(artifact)
    assert not r['verified'] and r['audit_status']=='FAIL'


def test_corrupt_output_hash_rejected(artifact):
    with (artifact.directory/'pruned.pt').open('ab') as stream:stream.write(b'corrupted')
    assert audit(artifact)['issues']==['COMPLETED_OUTPUT_HASH_MISMATCH']


def test_duplicate_identity_not_counted(artifact):
    assert audit(artifact,{'run':['run','duplicate']})['issues']==['DUPLICATE_RESULT_IDENTITY']


@pytest.mark.parametrize('mutation,code',[
    ('seed','CONDITION_IDENTITY_MISMATCH'),('metric','PRUNED_METRIC_val_accuracy'),
    ('compression','EXACT_COMPRESSION'),('criterion','STATIC_CRITERION_IDENTITY'),('source','RUN_SOURCE_MISMATCH'),('leak','TEST_LABEL_EXPOSURE_REPORTED')])
def test_self_consistent_hashes_do_not_hide_semantic_corruption(artifact,mutation,code):
    r=deepcopy(artifact.report)
    if mutation=='seed':r['condition']['seed']=202
    elif mutation=='metric':r['validation']['accuracy']+=.1
    elif mutation=='compression':r['achieved_compression']+=.1
    elif mutation=='criterion':r['statistics']['static_plan']['criterion']='snip'
    elif mutation=='source':r['source_commit']='f'*40
    elif mutation=='leak':r['test_labels_exposed']=True
    put(artifact.directory/'report.json',r);seal(artifact.directory)
    assert audit(artifact)['issues']==[code]


def test_finite_unrelated_tensor_change_rejected(artifact):
    p=artifact.directory/'pruned.pt';cp=torch.load(p,weights_only=True)
    name=next(k for k in cp['model_state_dict'] if k.startswith('convs.2') and cp['model_state_dict'][k].ndim==1)
    cp['model_state_dict'][name]+=0.001;torch.save(cp,p)
    artifact.report['checkpoint_bytes']=p.stat().st_size;put(artifact.directory/'report.json',artifact.report);seal(artifact.directory)
    assert audit(artifact)['issues']==['DENSE_TENSOR_PROJECTION_MISMATCH']


def test_nonfinite_checkpoint_rejected(artifact):
    p=artifact.directory/'pruned.pt';cp=torch.load(p,weights_only=True);next(iter(cp['model_state_dict'].values())).flatten()[0]=float('nan');torch.save(cp,p)
    seal(artifact.directory)
    assert audit(artifact)['issues']==['NONFINITE_CHECKPOINT']


def test_inventory_missing_corrupt_unexpected_and_duplicates(tmp_path):
    (tmp_path/'ok').write_bytes(b'a');(tmp_path/'bad').write_bytes(b'wrong');(tmp_path/'extra').write_bytes(b'a')
    inv=[dict(path='ok',bytes=1,sha256=hashlib.sha256(b'a').hexdigest()),dict(path='bad',bytes=1,sha256=hashlib.sha256(b'b').hexdigest()),dict(path='absent',bytes=1,sha256='0'*64)]
    statuses={r['path']:r['status'] for r in inventory_audit(tmp_path,inv)}
    assert statuses==dict(ok='VERIFIED',bad='CORRUPT',absent='MISSING',extra='UNEXPECTED')
    with pytest.raises(ValueError,match='Duplicate'):inventory_audit(tmp_path,inv+inv[:1])
    for path in ('../escape','/absolute'):
        with pytest.raises(ValueError):safe_relative(path)


def test_summary_preserves_failed_expected_seed():
    rows=[dict(dataset='cora',architecture='gcn',method='tabular_q',requested_compression=.3,dense_training_seed=seed,
        collected=seed!=303,verified=seed==101,audit_status='PASS' if seed==101 else 'FAIL',outside_symmetric_band=False,physical_pruning=True,terminal_reason='max_steps',pruned_val_accuracy=.8)
        for seed in (101,202,303)]
    result=a.summarize(rows)[0]
    assert result['n_expected']==3 and result['n_collected']==2 and result['n_verified']==1 and result['n_fail']==2
    assert result['pruned_val_accuracy_mean']==.8 and result['pruned_val_accuracy_sd'] is None


def test_accounting_does_not_equate_acceptance_with_completion(tmp_path):
    sub=dict(jobs=[dict(job_id=i,job=dict(id='node'+str(i),stage='prune'),resources=dict(slots=4,wall_minutes=20)) for i in (1,2,3,4)])
    text='Job <1>,\nDone successfully. The CPU time used is 2.0 seconds;\nJob <2>,\nExited with exit code 1;\nJob <3>,\nExited TERM_OWNER;\n'
    rows=a.parse_accounting(text,sub,tmp_path)
    assert [r['status'] for r in rows]==['DONE','FAILED','CANCELLED','ACCOUNTING_MISSING']
    assert not rows[0]['stdout_success']


def test_global_missing_evidence_is_not_claimed_verified(tmp_path):
    source=tmp_path/'source';output=tmp_path/'audit';root=tmp_path/'raw';root.mkdir()
    condition=dict(id='x',dataset='cora',model='gcn',seed=101,runs=[dict(id='x-random',method='static_random',implementation='random_local_v2',policy_seed=101,target=.3)])
    put(source/'research/experiments/polus_complete_pruning_20261031/manifest.json',dict(conditions=[condition]))
    result=a.audit_safe(root,source,[],'',output)
    assert result['verified_runs']==0 and result['audit_status_counts']=={'NOT_VERIFIED':1} and not result['go_for_descriptive_analysis']
    with pytest.raises(FileExistsError):a.audit_safe(root,source,[],'',output)


def test_equal_width_different_original_mask_rejected(artifact):
    report=deepcopy(artifact.report);keeps=deepcopy(report['retained_original_ids']);keeps[0][-1]=7
    report['retained_original_ids']=keeps;report['statistics']['static_plan']['kept_indices']=keeps
    path=artifact.directory/'pruned.pt';cp=torch.load(path,weights_only=True);cp['original_kept_ids']=keeps;torch.save(cp,path)
    report['checkpoint_bytes']=path.stat().st_size;put(artifact.directory/'report.json',report);seal(artifact.directory)
    assert audit(artifact)['issues']==['DENSE_TENSOR_PROJECTION_MISMATCH']


def manual_terminal_trace():
    """Independent two-action, one-step zero-weight fixture; no production search."""
    import math
    import numpy as np
    widths=[5,5];features=1;classes=2;p0=count_parameters(features,classes,widths,'gcn')
    state={'convs.0.lin.weight':torch.zeros(5,1),'convs.0.bias':torch.zeros(5),
           'convs.1.lin.weight':torch.zeros(5,5),'convs.1.bias':torch.zeros(5),
           'convs.2.lin.weight':torch.zeros(2,5),'convs.2.bias':torch.zeros(2)}
    target=1-count_parameters(1,2,[4,5],'gcn')/p0;accuracy=float(np.float32(.8))
    ctx=dict(state=state,features=1,classes=2,kind='gcn',dense_widths=widths,target=target,tolerance=.01,minimum=4,maximum=1,penalty=2.,dense_accuracy=accuracy)
    baseline=['local_v2',[5,5],1,math.floor(accuracy/.01),False];entries=[];transitions=[];paths=[]
    for layer in (0,1):
        keeps=[list(range(5)),list(range(5))];keeps[layer]=list(range(4));after=list(map(len,keeps))
        compression=1-count_parameters(1,2,after,'gcn')/p0;penalty=-2*max(0.,abs(compression-target)-.01);reward=.6*compression+penalty
        key=f'prune:{layer}:1'
        entry=dict(action=key,successful_deletions=1,widths_before=[5,5],widths_after=after,sparsity=compression,signed_target_deviation=compression-target,
            validation_accuracy=accuracy,accuracy_drop=0.,reward=reward,terminal_penalty=penalty,terminal_reason='max_steps',original_kept_ids=keeps)
        entries.append(entry)
        transitions.append(dict(path=[key],from_state=baseline,state=['local_v2',after,0,math.floor(accuracy/.01),True],parameters_after=count_parameters(1,2,after,'gcn'),transition=entry,evaluation_reused=False))
        paths.append(dict(actions=[key],rewards=[reward],score=reward,terminal=True,terminal_reason='max_steps'))
    return ctx,baseline,entries,transitions,paths


@pytest.mark.parametrize('mutation',[None,'reward','q_value'])
def test_q_offline_bellman_and_reward_audit(mutation):
    from gnn_pruning.experiments.complete_audit_trace import audit_q
    ctx,state,entries,_,_=manual_terminal_trace();entry=entries[0]
    report=dict(run=dict(policy_seed=101),training=[dict(event='reset',episode=0,widths=[5,5]),dict(episode=0,**entry)],trace=[entry],
        q_table=[dict(state=state,actions={'prune:0:1':.1*entry['reward']})],
        statistics=dict(episodes=1,q_entries=1,unique_states=2,legal_state_action_slots=2,q_table_sparsity=.5))
    if mutation=='reward':report['training'][1]['reward']+=.1
    if mutation=='q_value':report['q_table'][0]['actions']['prune:0:1']+=.1
    protocol=dict(q_learning=dict(episodes=1,epsilon=.2,alpha=.1,gamma=.9))
    if mutation:
        with pytest.raises(ValueError,match='REWARD|Q_BELLMAN_VALUE'):audit_q(report,ctx,protocol)
    else:assert audit_q(report,ctx,protocol)[0]==4


@pytest.mark.parametrize('mutation',[None,'return','terminal_extension','count'])
def test_lookahead_recorded_paths_and_honest_counts(mutation):
    from gnn_pruning.experiments.complete_audit_trace import audit_lookahead
    ctx,state,entries,nodes,paths=manual_terminal_trace()
    decision=dict(state=state,original_kept_ids=[list(range(5)),list(range(5))],legal_actions=['prune:0:1','prune:1:1'],selected_action='prune:0:1',
        best_planned_path=['prune:0:1'],selected_score=paths[0]['score'],candidate_paths=paths,candidate_transitions=nodes,
        candidate_evaluations=2,reused_evaluations=0,explored_transitions=2,planning_seconds=.1)
    report=dict(trace=[entries[0]],decisions=[decision],statistics=dict(depth=2,gamma=.9,candidate_evaluations=2,reused_evaluations=0,explored_candidate_transitions=2,candidate_paths=2))
    if mutation=='return':paths[0]['score']+=.1
    elif mutation=='count':decision['candidate_evaluations']=3
    elif mutation=='terminal_extension':
        bad=deepcopy(nodes[0]);bad['path']=['prune:0:1','prune:1:1'];nodes.append(bad)
    if mutation:
        with pytest.raises(ValueError,match='LOOKAHEAD_RETURN|LOOKAHEAD_EXTENDS_TERMINAL|LOOKAHEAD_EVALUATION_ACCOUNTING'):audit_lookahead(report,ctx)
    else:assert audit_lookahead(report,ctx)[0]==3


def test_terminal_statistics_must_match_recorded_transition():
    from gnn_pruning.experiments.complete_audit_trace import audit_behavior
    ctx,_,entries,_,_=manual_terminal_trace()
    report=dict(run=dict(method='tabular_q'),statistics=dict(terminal_reason='agent_stop'),trace=[entries[0]])
    with pytest.raises(ValueError,match='TERMINAL_STATS_TRACE_MISMATCH'):audit_behavior(report,ctx,{})

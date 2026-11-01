"""Campaign scope, frozen inference-only runner and fail-closed launch tests."""
import json
from pathlib import Path
from types import SimpleNamespace
import time

import pytest
import torch

from gnn_pruning.experiments import flickr_pruning as f, flickr_pruning_lsf as l
from gnn_pruning.experiments.flickr_dense import metrics, sha, write, tensors_hash
from gnn_pruning.pruning.budget_v5 import METHODS, Budget
from test_budget_v5 import graph, network


def protocol():
    return dict(architectures=list(f.ARCHITECTURES),methods=list(METHODS),targets=[.3,.5],
        policy_seed=101,dense_seed=101,split_seed=101,source_commit='a'*40,
        budget=dict(tolerance=.01,minimum=4),c4_host='verified-c4',
        q_learning=dict(episodes=2,epsilon=.2,alpha=.1,gamma=.9))


def test_full56_no_substitution_no_dependencies_four_threads():
    p=protocol();rows=f.matrix(p);jobs=l.jobs(p)
    assert len(rows)==len(jobs)==len({r['id'] for r in rows})==56
    assert {r['implementation'] for r in rows}=={m+'_structural_budget_v5' for m in METHODS}
    assert all(not j['dependencies'] and j['resources']['slots']==4 for j in jobs)
    for j in jobs:
        cmd=l.command(j,Path('/fresh'))
        assert '-w' not in cmd and '-x' in cmd and '180' in cmd
    p['targets']=[.3]
    with pytest.raises(ValueError):f.matrix(p)


@pytest.mark.parametrize('kind',['gcn','graphsage'])
@pytest.mark.parametrize('depth',[2,3,4])
@pytest.mark.parametrize('method',METHODS)
def test_real_forward_runner_checkpoint_masks_metrics_and_no_test(tmp_path,monkeypatch,kind,depth,method):
    p=protocol();model=network(kind,depth);data=graph();data.y[8:]=-1
    split=dict(train=list(range(4,8)),val=list(range(4)),test=list(range(8,12)))
    condition=next(r for r in f.matrix(p) if r['method']==method and r['target']==.3)
    p['dense']={condition['architecture']:dict(condition=dict(model=kind))}
    original=dict(checkpoint_sha256='pinned-original',metrics=metrics(model,data,split),tensor_sha256=tensors_hash(model.state_dict()))
    manifest=dict(split_sha256='split',training_tensor_sha256='masked-graph')
    monkeypatch.setattr(f,'frozen',lambda *a:(model,data,split,manifest,original))
    report=f.execute(tmp_path,p,condition)
    assert report['strict_reload'] and report['retained_projection'] and not report['test_labels_accessed']
    assert report['parameters_after']<report['parameters_before']
    assert report['statistics']['complete']
    assert report['validation_evaluation_requests']==report['observed_validation_evaluations']+report['statistics']['evaluation_cache_hits']
    if method in ('adaptive_greedy','lookahead_depth2'):
        decisions=json.loads((tmp_path/'runs'/condition['id']/'decisions.json').read_text())
        assert sum(d['candidate_evaluations'] for d in decisions)==report['statistics']['candidate_evaluations']
    assert len(report['validation_per_class_f1'])==3
    out=tmp_path/'runs'/condition['id']
    for name,expected in json.loads((out/'completed.json').read_text()).items():assert sha(out/name)==expected
    with pytest.raises(FileExistsError):f.execute(tmp_path,p,condition)


def test_256_three_hidden_budget_exceeds_old_horizon_and_is_exact():
    m=network('graphsage',4,256);b=Budget(m,.5)
    assert b.horizon==756 and b.goal_count>0 and not b.fallback
    assert b.reachable((256,256,256),756)


@pytest.mark.parametrize('stdout,code,expected',[
    ('Job <123> is submitted to queue <normal>.\n',0,'ACCEPTED'),
    ('Job <123> is submitted to queue <normal>.\n',1,'UNRESOLVED'),
    ('connection lost',0,'UNRESOLVED')])
def test_intent_single_attempt_duplicate_ambiguity(tmp_path,stdout,code,expected):
    root=tmp_path/'root';root.mkdir();source=tmp_path/'source';source.mkdir();task=tmp_path/'task';task.mkdir()
    (task/'activate-polus.sh').write_text('verified');(source/'safe.py').write_text('# pinned')
    write(root/'source-files.json',{'safe.py':sha(source/'safe.py')})
    p=protocol();write(root/'protocol.json',p);plan=l.jobs(p,technical=True)[:2];calls=[]
    def invoke(*a,**kw):calls.append(a);return SimpleNamespace(stdout=stdout,stderr='',returncode=code)
    receipts=l.submit(plan,root,source,task,root/'protocol.json',sha(task/'activate-polus.sh'),invoke)
    assert receipts[0]['status']==expected and len(calls)==(2 if expected=='ACCEPTED' else 1)
    with pytest.raises(FileExistsError):l.submit(plan,root,source,task,root/'protocol.json',sha(task/'activate-polus.sh'),invoke)


def test_research_gate_fails_before_any_scheduler_call(tmp_path):
    p=protocol();write(tmp_path/'protocol.json',p)
    write(tmp_path/'capacity.json',dict(status='FAIL',source_commit='a'*40,checked_unix=time.time()))
    with pytest.raises(ValueError,match='resource gate'):
        l.submit(l.jobs(p),tmp_path,tmp_path,tmp_path,tmp_path/'protocol.json','x')


def test_launcher_cannot_train_or_touch_test_labels(tmp_path):
    p=protocol();path=tmp_path/'protocol.json';write(path,p)
    script=l.launcher(l.jobs(p)[0],tmp_path,tmp_path,tmp_path,path,'hash')
    assert 'gnn_pruning.experiments.flickr_pruning ' in script
    assert 'flickr_dense ' not in script and 'heldout' not in script and 'fit(' not in script

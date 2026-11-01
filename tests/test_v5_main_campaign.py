"""Fixed scope, frozen-label isolation, native-engine bridge and independent geometry."""
import copy
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from gnn_pruning.experiments import v5_campaign as c
from gnn_pruning.experiments.flickr_dense import sha, tensors_hash
from gnn_pruning.pruning.budget_v5 import Budget, METHODS, run
from test_budget_v5 import network, graph


def test_authorized_matrix_exact_and_no_historical_substitutions():
    p = c.protocol(); rows = c.matrix(p)
    assert len(rows) == len({r['id'] for r in rows}) == 252
    assert len({r['dense_id'] for r in rows}) == 18
    assert all(sum(r['dense_id'] == k for r in rows) == 14 for k in {r['dense_id'] for r in rows})
    assert {r['implementation'] for r in rows} == {m+'_structural_budget_v5' for m in METHODS}
    assert all(r['policy_seed'] == r['dense_seed'] == r['split_seed'] for r in rows)
    for field, value in [('datasets', ['flickr']), ('q_learning', dict(episodes=3)), ('methods', list(METHODS)+['adaptive_layerwise'])]:
        changed = copy.deepcopy(p); changed[field] = value
        with pytest.raises(ValueError): c.matrix(changed)


@pytest.mark.parametrize('kind', ['gcn', 'graphsage'])
@pytest.mark.parametrize('features,classes', [(1433,7), (3703,6), (500,3)])
@pytest.mark.parametrize('target', [.3, .5])
def test_citation_geometry_all_widths_independent(kind, features, classes, target):
    from gnn_pruning.models import build_model
    from fractions import Fraction
    model = build_model(kind, in_channels=features, hidden_channels=32, out_channels=classes, num_layers=3, dropout=.5)
    b = Budget(model, target); factor = 2 if kind == 'graphsage' else 1
    count = lambda a,z: factor*(features*a+a*z+z*classes)+a+z+classes
    assert b.p0 == count(32,32) == sum(p.numel() for p in model.parameters())
    distances = {(a,z): abs(Fraction(b.p0-count(a,z), b.p0)-Fraction(str(target)))
                 for a,z in itertools.product(range(4,33), repeat=2) if (a,z) != (32,32)}
    goals = {w for w,g in distances.items() if g <= Fraction(1,100)}
    if not goals: goals = {w for w,g in distances.items() if g == min(distances.values())}
    assert {w for w in distances if b.goal(w)} == goals
    for w in distances:
        assert b.reachable(w, 56) == any(a <= w[0] and z <= w[1] for a,z in goals)


@pytest.mark.parametrize('kind', ['gcn', 'graphsage'])
@pytest.mark.parametrize('method', METHODS)
def test_bridge_preserves_engine_and_cache_validation_only(kind, method):
    p = c.protocol(); row = next(r for r in c.matrix(p) if r['model'] == kind and r['method'] == method)
    model = network(kind, 3); data = graph(); data.y[8:] = -1
    split = dict(train=list(range(4)), val=list(range(4,8)), test=list(range(8,12)))
    p['dense'] = {row['dense_id']:dict(graph='masked')}; p['graphs'] = {'masked':dict(identity='masked')}
    before = tensors_hash(model.state_dict())
    final, kept, details, wall, cpu, budget = c.search(model, data, split, row, p, technical=True)
    stats = details['statistics']
    assert (data.y[split['test']] == -1).all()
    assert tensors_hash(model.state_dict()) == before
    assert stats['validation_callback_requests'] == stats['actual_validation_forwards'] + stats['exact_cache_hits']
    assert len(details['evaluation_cache']) == stats['validation_callback_requests']
    assert sum(v.numel() for v in final.parameters()) == budget.count(tuple(map(len,kept)))
    if not method.startswith('static_'):
        def accuracy(m):
            with torch.no_grad(): return float((m(data)[split['val']].argmax(-1) == data.y[split['val']]).float().mean())
        direct = run(model, accuracy, method, row['target'], seed=row['policy_seed'], episodes=3,
                     max_deployment_steps=2 if method == 'tabular_q' else None)
        assert details['deployment'] == direct.trace and details['training'] == direct.training_trace
        assert kept == direct.kept_indices and tensors_hash(final.state_dict()) == tensors_hash(direct.model.state_dict())
        assert details['q_table'] == [dict(state=s,actions=a) for s,a in direct.q_table.items()]
    else:
        assert stats['actual_validation_forwards'] == 0
        assert stats['scoring_gradient_forwards'] == int(method == 'static_snip')
    assert wall > 0 and cpu > 0


def test_test_labels_fail_before_adaptive_evaluation():
    p = c.protocol(); row = next(r for r in c.matrix(p) if r['method'] == 'tabular_q')
    p['dense'] = {row['dense_id']:dict(graph='unmasked')};p['graphs'] = {'unmasked':{}}
    with pytest.raises(ValueError, match='Test labels'):
        c.search(network('gcn',3), graph(), dict(train=[0],val=[1],test=[2]), row, p, True)


def test_batch_isolated_processes_and_no_retry(tmp_path, monkeypatch):
    p=c.protocol(); calls=[]
    def invoke(args, **kwargs):
        calls.append(args); return SimpleNamespace(returncode=int(len(calls)==3))
    monkeypatch.setattr(c.subprocess,'run',invoke)
    with pytest.raises(RuntimeError):c.batch(tmp_path,p,'cora-gcn-s101',1)
    assert len(calls)==14 and all('--technical' not in a for a in calls)
    assert all(a[a.index('--threads')+1]=='1' for a in calls)
    with pytest.raises(FileExistsError):c.batch(tmp_path,p,'cora-gcn-s101',1)


def test_production_capacity_fail_closed_exact_coverage(tmp_path):
    import time
    from gnn_pruning.experiments import v5_campaign_lsf as l
    from gnn_pruning.experiments.flickr_dense import digest
    p=c.protocol();p['source_commit']='a'*40
    p['dense']={r['dense_id']:{} for r in c.matrix(p)}
    gate=dict(status='PASS',source_commit=p['source_commit'],protocol_sha256=digest(p),checked_unix=time.time(),
              dense_verified=18,technical_verified=168,threads=1,
              conditions={r['id']:dict(safe_worker_seconds=10,memory_mb=1024) for r in c.matrix(p)})
    jobs=l.production_jobs(p,gate)
    assert len(jobs)==18 and sum(len(j['conditions']) for j in jobs)==252
    assert all(j['resources']['host']==l.C4 and j['threads']==1 for j in jobs)
    for mutation in ('status','stale','missing','source','unsafe'):
        bad=copy.deepcopy(gate)
        if mutation=='status':bad['status']='FAIL'
        if mutation=='stale':bad['checked_unix']-=3601
        if mutation=='missing':bad['conditions'].pop(next(iter(bad['conditions'])))
        if mutation=='source':bad['source_commit']='b'*40
        if mutation=='unsafe':next(iter(bad['conditions'].values()))['safe_worker_seconds']=20000
        with pytest.raises(ValueError):l.production_jobs(p,bad)


@pytest.mark.parametrize('response,code,accepted',[
    ('Job <891> is submitted to queue <normal>.\n',0,True),
    ('Job <891> is submitted to queue <normal>.\n',1,False),
    ('timeout, possibly submitted',0,False)])
def test_lsf_single_attempt_intent_before_call(tmp_path,response,code,accepted):
    from gnn_pruning.experiments import v5_campaign_lsf as l
    from gnn_pruning.experiments.flickr_dense import write
    root=tmp_path/'root';root.mkdir();source=tmp_path/'src';source.mkdir();task=tmp_path/'task';task.mkdir()
    (task/'activate-polus.sh').write_text('# verified');(source/'pinned').write_text('immutable')
    p=c.protocol();p['source_commit']='a'*40;write(root/'protocol.json',p)
    write(root/'source-files.json',{'pinned':sha(source/'pinned')});calls=[]
    def invoke(args,**kw):
        assert (root/'submissions'/l.technical_jobs()[len(calls)]['id']/'intent.json').is_file()
        calls.append(args);return SimpleNamespace(stdout=response,stderr='',returncode=code)
    receipts=l.submit(l.technical_jobs(),root,source,task,sha(task/'activate-polus.sh'),invoke)
    assert receipts[0]['accepted']==accepted
    assert len(calls)==(2 if accepted else 1)
    # Repeated same mocked job ID makes the second response unresolved.
    if accepted:assert receipts[1]['status']=='UNRESOLVED'
    with pytest.raises(FileExistsError):l.submit(l.technical_jobs(),root,source,task,sha(task/'activate-polus.sh'),invoke)


def test_launcher_explicit_threads_and_no_training(tmp_path):
    from gnn_pruning.experiments import v5_campaign_lsf as l
    (tmp_path/'protocol.json').write_text('{}')
    for job in l.technical_jobs():
        text=l.launcher(job,tmp_path,tmp_path,tmp_path,'hash')
        assert f'OMP_NUM_THREADS={job["threads"]}' in text
        assert 'v5_campaign' in text and 'DenseTrainer' not in text and '.fit(' not in text
        assert '-x' in l.command(job,tmp_path)


@pytest.mark.parametrize('kind',['gcn','graphsage'])
@pytest.mark.parametrize('method',METHODS)
def test_independent_recorded_trace_audit_with_real_forwards(tmp_path,kind,method):
    from gnn_pruning.experiments.v5_campaign_audit import behavior
    from gnn_pruning.experiments.flickr_dense import write
    p=c.protocol();row=next(r for r in c.matrix(p) if r['model']==kind and r['method']==method)
    model=network(kind,3,32);data=graph();data.y[8:]=-1
    split=dict(train=list(range(4)),val=list(range(4,8)),test=list(range(8,12)))
    p['dense']={row['dense_id']:dict(graph='masked')};p['graphs']={'masked':dict(identity='frozen')}
    _,_,details,_,_,_=c.search(model,data,split,row,p,technical=True)
    for name,value in details.items():write(tmp_path/(name+'.json'),value)
    cfg=model.export_architecture_config();cfg.pop('name')
    cp=dict(model_config=cfg,model_name=kind,model_state_dict=model.state_dict())
    result=behavior(tmp_path,cp,row,details['statistics'],True)
    assert result['training_steps']==len(details['training'])
    if method=='tabular_q':
        records=json.loads((tmp_path/'training.json').read_text());records[0]['q_after']+=.01
        (tmp_path/'training.json').write_text(json.dumps(records))
        with pytest.raises(AssertionError):behavior(tmp_path,cp,row,details['statistics'],True)

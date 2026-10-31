"""Focused Dense-only sweep/provenance/submission tests on tiny synthetic graphs."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import time

import pytest
import torch

from gnn_pruning.config import load_yaml
from gnn_pruning.data import generate_exact_ratio_split
from gnn_pruning.experiments import flickr_dense as d, flickr_lsf as l
from gnn_pruning.models import build_model

PROTOCOL=Path('research/experiments/flickr_dense_sweep_20261031/protocol.yaml')


def protocol():
    p=load_yaml(PROTOCOL);p['source_commit']='a'*40
    return p


def inputs(tmp_path):
    root=tmp_path/'inputs';root.mkdir()
    split=generate_exact_ratio_split(30,seed=101).to_dict()
    torch.manual_seed(77)
    tensors=dict(x=torch.randn(30,5),edge_index=torch.stack([torch.arange(30),torch.roll(torch.arange(30),1)]),
                 y=torch.arange(30)%3)
    tensors['y'][split['test']]=-1
    torch.save(tensors,root/'graph.pt');d.write(root/'split.json',split)
    d.write(root/'manifest.json',dict(classes=3,features=5,graph_file_sha256=d.sha(root/'graph.pt'),
           training_tensor_sha256=d.tensors_hash(tensors),split_sha256=d.digest(split)))
    return root


def test_exact_authorized_32_matrix_and_one_protocol():
    rows=d.matrix(protocol())
    assert len(rows)==len({r['id'] for r in rows})==32
    assert {(r['model'],r['layers'],r['width']) for r in rows}=={
        (m,n,w) for m in ('gcn','graphsage') for n in (2,3,4,5) for w in (32,64,128,256)}
    assert {r['seed'] for r in rows}=={101}


@pytest.mark.parametrize('key,value',[('seed',42),('layers',[2,3]),('widths',[32]),('dataset','cora')])
def test_matrix_cannot_shrink_or_change_scope(key,value):
    p=protocol();p[key]=value
    with pytest.raises(ValueError):d.matrix(p)


def test_research_budget_cannot_be_shortened():
    p=protocol();p['training']['max_epochs']=20
    with pytest.raises(ValueError):d.matrix(p)


def test_all_depths_and_widths_apply_with_independent_parameter_oracle():
    p=protocol()
    for row in d.matrix(p):
        arch=d.architecture(row,dict(features=500,classes=7),p)
        model=build_model(row['model'],**arch)
        n,w=row['layers'],row['width']
        linear=500*w+(n-2)*w*w+w*7
        expected=(2 if row['model']=='graphsage' else 1)*linear+(n-1)*w+7
        assert sum(x.numel() for x in model.parameters())==expected
        assert len(model.convs)==n and model.hidden_channel_dims==[w]*(n-1)


def test_frozen_split_is_disjoint_exact_and_repeatable(tmp_path):
    root=inputs(tmp_path);data,split,manifest=d.load_inputs(root)
    assert split==generate_exact_ratio_split(30,seed=101).to_dict()
    assert len(set().union(*(set(v) for v in split.values())))==30
    assert bool((data.y[split['test']]==-1).all())
    assert d.digest(split)==manifest['split_sha256']


@pytest.mark.parametrize('kind',['bytes','split','test_labels'])
def test_bad_or_unmasked_inputs_are_rejected(tmp_path,kind):
    root=inputs(tmp_path)
    if kind=='bytes':
        with (root/'graph.pt').open('ab') as stream:stream.write(b'x')
    elif kind=='split':
        split=json.loads((root/'split.json').read_text());split['train'][0]=split['val'][0]
        (root/'split.json').write_text(json.dumps(split))
    else:
        value=torch.load(root/'graph.pt',weights_only=True);value['y'].fill_(0);torch.save(value,root/'graph.pt')
        m=json.loads((root/'manifest.json').read_text());m.update(graph_file_sha256=d.sha(root/'graph.pt'),training_tensor_sha256=d.tensors_hash(value))
        (root/'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError):d.load_inputs(root)


@pytest.mark.parametrize('model,layers',[('gcn',2),('gcn',5),('graphsage',2),('graphsage',5)])
def test_tiny_real_dense_fit_reload_logging_and_no_test_access(tmp_path,model,layers):
    root=inputs(tmp_path);p=protocol();p['threads']=1;p['input_manifest_sha256']=d.sha(root/'manifest.json')
    condition=dict(id='tiny-'+model+str(layers),model=model,layers=layers,width=4,seed=101)
    report=d.run(root,tmp_path/'first',condition,p,technical=True,epochs=2,lr=.001)
    assert report['strict_reload_verified'] and not report['test_metrics_evaluated']
    assert report['result']['epochs_ran']==2 and report['technical']
    assert set(report['metrics'])=={'train','val'}
    assert report['training_wall_seconds']>0 and report['peak_process_rss_bytes']>0
    assert len((tmp_path/'first/history.jsonl').read_text().splitlines())==2
    second=d.run(root,tmp_path/'second',condition,p,technical=True,epochs=2,lr=.001)
    assert second['tensor_sha256']==report['tensor_sha256'] and second['metrics']==report['metrics']
    with pytest.raises(FileExistsError):d.run(root,tmp_path/'first',condition,p,technical=True,epochs=2,lr=.001)
    cp=torch.load(tmp_path/'first/checkpoint.pt',weights_only=True)
    config=json.loads((tmp_path/'first/resolved_config.json').read_text());manifest=json.loads((root/'manifest.json').read_text())
    wrong=dict(condition,width=8)
    with pytest.raises(ValueError):d.verify_checkpoint(cp,wrong,config,manifest)


def test_research_overrides_rejected_before_creating_output(tmp_path):
    p=protocol();condition=d.matrix(p)[0]
    with pytest.raises(ValueError):d.run(tmp_path,tmp_path/'out',condition,p,epochs=5)
    assert not (tmp_path/'out').exists()


def test_failure_keeps_partial_learning_curve(tmp_path,monkeypatch):
    root=inputs(tmp_path);p=protocol();p['threads']=1;p['input_manifest_sha256']=d.sha(root/'manifest.json')
    def fail(self,*args,progress_callback=None,**kwargs):
        progress_callback(dict(epoch=1.,best_epoch=1.,train_loss=1.,val_loss=1.,train_acc=.3,val_acc=.3,elapsed_sec=.01))
        raise RuntimeError('simulated termination')
    monkeypatch.setattr(d.DenseTrainer,'fit',fail)
    with pytest.raises(RuntimeError):d.run(root,tmp_path/'out',dict(id='tiny',model='gcn',layers=2,width=4,seed=101),p,True,2,.001)
    assert (tmp_path/'out/best.partial.pt').exists()
    assert json.loads((tmp_path/'out/failed.json').read_text())['completed_epochs']==1
    assert not (tmp_path/'out/completed.json').exists()


def capacity():
    return dict(status='PASS',common_lr=.001,checked_unix=time.time(),source_commit='a'*40,
                full_performance_hosts=['fast1','fast2'],reduced_frequency_host='slow',
                conditions={r['id']:{h:dict(safe_500_epoch_seconds=3600,memory_mb=16384)
                                    for h in ('fast1','fast2','slow')} for r in d.matrix(protocol())})


def test_all32_independent_jobs_with_bounded_host_eligibility():
    jobs=l.research_jobs(protocol(),capacity())
    assert len(jobs)==32 and all(j['dependencies']==[] for j in jobs)
    for job in jobs:
        r=next(r for r in d.matrix(protocol()) if r['id']==job['condition'])
        if r['layers']>=4 and r['width']>=128:assert 'slow' not in job['resources']['eligible_hosts']
        else:assert 'slow+10' in job['resources']['host_preferences']
        command=l.command(job,Path('/fresh'))
        assert '-x' in command and '-w' not in command and '180' in command


def test_insufficient_wall_capacity_blocks_instead_of_shortening_epochs():
    c=capacity()
    for h in ('fast1','fast2','slow'):c['conditions'][d.matrix(protocol())[0]['id']][h]['safe_500_epoch_seconds']=11000
    with pytest.raises(ValueError,match='unchanged500'):l.research_jobs(protocol(),c)


@pytest.mark.parametrize('response,code,expected',[
    ('Job <123> is submitted to queue <normal>.\n',0,'ACCEPTED'),
    ('Job <123> is submitted to queue <normal>.\n',1,'UNRESOLVED'),
    ('connection lost after request',0,'UNRESOLVED')])
def test_single_attempt_receipts_and_duplicate_prevention(tmp_path,response,code,expected):
    root=tmp_path/'root';root.mkdir();source=tmp_path/'source';source.mkdir();task=tmp_path/'task';task.mkdir()
    (task/'activate-polus.sh').write_text('verified')
    (source/'safe.py').write_text('# pinned\n');d.write(root/'source-files.json',{'safe.py':d.sha(source/'safe.py')})
    path=root/'protocol.json';d.write(path,protocol());calls=[]
    jobs=l.research_jobs(protocol(),capacity())[:2]
    def invoke(*a,**kw):calls.append(a);return SimpleNamespace(stdout=response,stderr='',returncode=code)
    receipts=l.submit(jobs,root,source,task,path,d.sha(task/'activate-polus.sh'),runner=invoke)
    assert receipts[0]['status']==expected
    assert (root/'submissions'/jobs[0]['id']/'intent.json').exists()
    assert (root/'submissions'/jobs[0]['id']/'response.json').exists()
    assert len(calls)==(2 if expected=='ACCEPTED' else 1)
    if expected=='ACCEPTED':assert receipts[1]['status']=='UNRESOLVED'  # repeated scheduler ID is ambiguous
    with pytest.raises(FileExistsError):l.submit(jobs,root,source,task,path,d.sha(task/'activate-polus.sh'),runner=invoke)


def test_fixed_preflight_is_separate_and_short():
    trials=l.preflight_trials()
    assert len(trials)==6 and all(t['epochs']<=12 for t in trials)
    assert {t['lr'] for t in trials}=={.001,.01}


def test_incomplete_analysis_preserves_audit_without_ranking(tmp_path):
    from gnn_pruning.experiments.flickr_analysis import analyze
    (tmp_path/'raw').mkdir()
    with pytest.raises(ValueError,match='Incomplete'):
        analyze(tmp_path/'raw',protocol(),{},tmp_path/'analysis')
    audit=json.loads((tmp_path/'analysis/COMPLETENESS.json').read_text())
    assert len(audit['missing'])==32 and audit['verified']==0
    assert not (tmp_path/'analysis/RESULTS.csv').exists()
    assert not (tmp_path/'analysis/ANALYSIS.md').exists()


def test_convergence_flags_are_operational_and_do_not_claim_success():
    history=[dict(train_loss=1.,val_loss=v,train_acc=.5,val_acc=.5) for v in (1.,3.,2.)]
    flags=d.convergence(history,dict(best_epoch=0),500)
    assert flags['stopped_by']=='early_stopping_patience'
    assert flags['large_early_loss_excursion'] and flags['validation_loss_ended_above_initial']
    assert not flags['convergence_established']


def test_source_has_no_pruning_or_heldout_evaluator_imports():
    import ast
    tree=ast.parse(Path(d.__file__).read_text())
    imported=[node.module for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)]
    assert not any(name and (name.startswith('gnn_pruning.pruning.') or 'workflow' in name) for name in imported)
    assert 'heldout-labels' not in Path(d.__file__).read_text()

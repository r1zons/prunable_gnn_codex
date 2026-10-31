"""Technical orchestration tests; no dataset download or campaign execution."""
import copy
import json
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from gnn_pruning.config import load_yaml
from gnn_pruning.experiments import complete_campaign as c
from gnn_pruning.models import build_model

PROTOCOL = Path(__file__).parents[1]/'research/experiments/polus_complete_pruning_20261031/protocol.yaml'


def protocol():
    return load_yaml(PROTOCOL)


def test_fixed_matrix_topological_gates():
    p = protocol(); conditions = c.matrix(p); jobs = c.dependency_plan(p)
    assert len(conditions) == 18 and sum(len(x['runs']) for x in conditions) == 252
    assert len({r['id'] for x in conditions for r in x['runs']}) == 252
    assert len(jobs) == 55
    seen = set()
    for job in jobs:
        assert set(job['dependencies']) <= seen
        seen.add(job['id'])
    assert sum(x['pilot'] for x in conditions) == 2
    for job in jobs:
        if job['stage'] == 'train' and 'cora-' not in job['condition']:
            assert job['dependencies'] == ['pilot-gate']
        if job['stage'] == 'prune':
            assert job['dependencies'] == ['validate-'+job['condition']]
    p['datasets'].append('questions')
    with pytest.raises(ValueError):c.matrix(p)


@pytest.mark.parametrize('kind',['gcn','graphsage'])
@pytest.mark.parametrize('method',list(c.METHODS))
def test_all_methods_real_forward_isolation_projection_and_counts(kind,method):
    torch.manual_seed(72)
    model = build_model(kind, in_channels=5, out_channels=3, hidden_channels=8, num_layers=3, dropout=.5).eval()
    data = Data(x=torch.randn(20,5),edge_index=torch.tensor([list(range(20)),[(i+1)%20 for i in range(20)]]),y=torch.arange(20)%3)
    split = dict(train=list(range(12)),val=list(range(12,16)),test=list(range(16,20)))
    before = c.tensor_hash(model.state_dict()); graph_before = c.tensor_hash(dict(x=data.x,y=data.y,edge_index=data.edge_index))
    p = protocol();p['adaptive']['max_steps'] = 2;p['q_learning']['episodes'] = 2
    run = dict(method=method,target=.3,policy_seed=101)
    final, keeps, evidence = c.search(model,data,split,run,p)
    c.verify_projection(model,final,keeps)
    assert c.tensor_hash(model.state_dict()) == before
    assert c.tensor_hash(dict(x=data.x,y=data.y,edge_index=data.edge_index)) == graph_before
    assert not evidence['test_labels_exposed']
    assert evidence['observed_validation_evaluations'] == evidence['statistics']['validation_evaluations']
    changed = data.clone();changed.y[split['test']] = 2-changed.y[split['test']]
    final2, keeps2, evidence2 = c.search(model,changed,split,run,p)
    assert keeps == keeps2 and c.tensor_hash(final.state_dict()) == c.tensor_hash(final2.state_dict())
    assert evidence['trace'] == evidence2['trace']


def test_artifact_hash_gate_and_no_overwrite(tmp_path):
    c.write(tmp_path/'report.json',dict(strict_reload=True))
    c.write(tmp_path/'completed.json',dict(files={'report.json':c.sha(tmp_path/'report.json')}))
    assert c.verify_completed(tmp_path)['strict_reload']
    with pytest.raises(FileExistsError):c.write(tmp_path/'report.json',{})
    (tmp_path/'report.json').write_text('{}')
    with pytest.raises(ValueError):c.verify_completed(tmp_path)


def test_pilot_gate_refuses_incomplete(tmp_path):
    with pytest.raises(FileNotFoundError):c.pilot_gate(tmp_path,dict(protocol(),source_commit='test'))


def test_full_train_validate_prune_lifecycle_tiny(tmp_path):
    p = protocol();p['architecture']['hidden_channels']=8;p['training']['max_epochs']=2;p['source_commit']='a'*40
    torch.manual_seed(8)
    tensors=dict(x=torch.randn(20,5),edge_index=torch.tensor([list(range(20)),[(i+1)%20 for i in range(20)]]),y=torch.arange(20)%3)
    (tmp_path/'inputs').mkdir();torch.save(tensors,tmp_path/'inputs/cora.pt')
    c.write(tmp_path/'inputs/manifest.json',dict(cora=dict(file='cora.pt',sha256=c.sha(tmp_path/'inputs/cora.pt'),tensor_sha256=c.tensor_hash(tensors))))
    p['input_manifest_sha256']=c.sha(tmp_path/'inputs/manifest.json')
    condition=c.matrix(p)[0]
    c.train(tmp_path,condition,p)
    c.validate(tmp_path,condition,p)
    p['adaptive']['max_steps']=2;p['q_learning']['episodes']=1
    c.prune(tmp_path,condition,p)
    for run in condition['runs']:
        r=c.verify_completed(tmp_path/'runs'/run['id'])
        assert r['strict_reload'] and r['retained_tensor_projection']
    # Failed integrity must not release a pruning batch.
    (tmp_path/'dense'/condition['id']/'checkpoint.pt').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='Validated checkpoint'):c.prune(tmp_path,condition,p)


def submission_setup(tmp_path):
    from gnn_pruning.experiments import campaign_submission as sub
    import time
    root=tmp_path/'root';root.mkdir();source=tmp_path/'source';source.mkdir();taskroot=tmp_path/'task';taskroot.mkdir()
    (taskroot/'activate-polus.sh').write_text('activation')
    p=protocol();p['source_commit']='a'*40
    c.write(root/'protocol.json',p);c.write(root/'source-files.json',{})
    preflight=dict(status='PASS',lsf_limit_unit='MB',native_inputs_verified=True,capacity_safe=True,source_commit='a'*40,
                   checked_unix=time.time(),maximum_aggregate_wall_minutes=845)
    return sub,root,source,taskroot,preflight


def test_submission_full_dag_exact_acceptances_and_duplicates(tmp_path):
    from types import SimpleNamespace
    sub,root,source,taskroot,p=submission_setup(tmp_path)
    commands=[]
    def fake(command,**kw):
        commands.append(command);return SimpleNamespace(stdout=f'Job <{100+len(commands)}> is submitted to queue <normal>.\n',stderr='',returncode=0)
    out=sub.submit(root,source,root/'protocol.json',taskroot,c.sha(taskroot/'activate-polus.sh'),p,runner=fake)
    assert len(out['jobs'])==55 and all(j['accepted'] for j in out['jobs'])
    assert out['aggregate_wall_minutes']==845
    assert '-w' not in commands[0]
    assert commands[0][commands[0].index('-M')+1] == '8192'
    assert 'done(101)' in commands[1]
    with pytest.raises(RuntimeError,match='claim'):
        sub.submit(root,source,root/'protocol.json',taskroot,c.sha(taskroot/'activate-polus.sh'),p,runner=fake)


def test_ambiguous_submission_never_retried(tmp_path):
    from types import SimpleNamespace
    sub,root,source,taskroot,p=submission_setup(tmp_path)
    called=[]
    def fake(*args,**kw):
        called.append(1);return SimpleNamespace(stdout='network disconnected',stderr='',returncode=1)
    out=sub.submit(root,source,root/'protocol.json',taskroot,c.sha(taskroot/'activate-polus.sh'),p,runner=fake)
    assert len(called)==1 and out['jobs'][0]['status']=='UNRESOLVED'
    assert (root/'submissions/train-cora-gcn-s101/intent.json').exists()

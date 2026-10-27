"""Frozen-input, platform identity and unattended-launch regression checks."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
import torch

from gnn_pruning.config import dump_yaml
from gnn_pruning.models import build_model
from gnn_pruning.experiments import v3_sprint as sprint
from gnn_pruning.pruning.workflow_v2 import tensor_hash, _json_hash


def config(tmp_path):
    return dict(run_id='test',model='gcn',variant='fixed_probe_v3',target=.3,training_seed=101,
        policy_seed=101,episodes=2,purpose='smoke',max_steps=20,penalty_lambda=2.,gamma=.9,
        alpha=.1,epsilon=.2,minimum=4,tolerance=.01,source_commit='0'*40,platform='local',
        threads=2,measure_inference=False,timeout_seconds=900,output_dir=str(tmp_path/'output'))


def test_platform_paths_do_not_change_algorithm_identity(tmp_path):
    cfg=config(tmp_path);inputs=dict(graph_tensor_sha256='a'*64,split={'sha256':'b'*64},models={'gcn':{'sha256':'c'*64}})
    other=dict(cfg,platform='polus',threads=4,output_dir='/elsewhere',run_id='other')
    assert sprint.algorithm_identity(cfg,inputs)==sprint.algorithm_identity(other,inputs)
    assert sprint.algorithm_identity(cfg,inputs)!=sprint.algorithm_identity(dict(other,policy_seed=5001),inputs)


@pytest.mark.parametrize('field,value',[('gamma',.8),('max_steps',3),('training_seed',5001),('episodes',100)])
def test_sprint_rejects_wrong_budget_or_scientific_settings(tmp_path,field,value):
    cfg=config(tmp_path);cfg[field]=value
    with pytest.raises(ValueError):sprint.validate_task(cfg)


@pytest.mark.parametrize('kind',['gcn','graphsage'])
def test_tensor_snapshot_masks_forbidden_labels_and_real_physical_reload(tmp_path,monkeypatch,kind):
    m=build_model(kind,in_channels=1433,hidden_channels=32,out_channels=7,num_layers=3,dropout=.5).eval()
    graph=dict(x=torch.zeros(2708,1433),edge_index=torch.tensor([[0,1],[1,0]]),y=torch.arange(2708)%7)
    split=dict(train=list(range(1624)),val=list(range(1624,2166)),test=list(range(2166,2708)))
    torch.save(graph,tmp_path/'graph.pt');dump_yaml(split,tmp_path/'splits.yaml')
    torch.save(dict(model_name=kind,model_config=dict(in_channels=1433,hidden_channels=32,out_channels=7,num_layers=3,dropout=.5),model_state_dict=m.state_dict(),run_signature=dict(dataset='cora',model_name=kind,seed=101,split_hash=_json_hash(split))),tmp_path/'dense.pt')
    manifest=dict(training_seed=101,graph_tensor_sha256=tensor_hash(graph),graph=dict(file='graph.pt',sha256=sprint.file_hash(tmp_path/'graph.pt')),split=dict(file='splits.yaml',sha256=sprint.file_hash(tmp_path/'splits.yaml')),models={kind:dict(file='dense.pt',sha256=sprint.file_hash(tmp_path/'dense.pt'),tensor_sha256=tensor_hash(m.state_dict()))})
    path=tmp_path/'inputs.json';path.write_text(json.dumps(manifest))
    _,data,idx,_,_=sprint.load_frozen(path,kind)
    assert torch.all(data.y[split['train']]==-1) and torch.all(data.y[split['test']]==-1)
    assert len(idx)==542 and torch.all(data.y[idx]>=0)
    monkeypatch.setattr(sprint,'check_source',lambda _:None)
    cfg=dict(config(tmp_path),model=kind,input_manifest=str(path))
    record=sprint.run_task(cfg);report=json.loads((Path(record['output'])/'metrics.json').read_text())
    assert record['status']=='completed' and report['hidden_widths']==[27,30]
    assert 'test' not in report and report['provenance']['label_access']=='validation_only'


def test_batch_invokes_real_module_and_marks_failed_tasks(tmp_path,monkeypatch):
    cfg=config(tmp_path);cfgpath=tmp_path/'config.json';cfgpath.write_text(json.dumps(cfg))
    manifest=dict(artifact_root=str(tmp_path/'batch'),workers=1,source_commit='0'*40,tasks=[dict(config=str(cfgpath))])
    monkeypatch.setattr(sprint,'validate_manifest',lambda _:manifest)
    commands=[]
    class Failed:
        pid=999999
        def __init__(self,args,**kwargs):commands.append(args)
        def wait(self,**kwargs):return 1
    monkeypatch.setattr(sprint.subprocess,'Popen',Failed)
    with pytest.raises(RuntimeError,match='At least one task failed'):sprint.run_batch('unused')
    assert commands[0][2:4]==['-m','gnn_pruning.experiments.v3_sprint']
    status=json.loads((tmp_path/'batch/batch.finished.json').read_text())
    assert not status['all_completed'] and status['failed_tasks']==1


def test_detached_child_survives_launcher_process_and_duplicate_claim_is_refused(tmp_path):
    module=Path(__file__).resolve().parents[1]/'research/experiments/local_structured_v3_sprint_20261027/scripts/launch_local.py'
    marker=tmp_path/'child-marker';receipt=tmp_path/'receipt.json'
    child_code='import time,pathlib,os;time.sleep(.35);pathlib.Path('+repr(str(marker))+').write_text(str(os.getppid()))'
    parent_code='''import importlib.util,os,sys
from pathlib import Path
spec=importlib.util.spec_from_file_location('launch',sys.argv[1]);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.launch([sys.executable,'-c',sys.argv[2]],Path(sys.argv[3]),Path(sys.argv[4]),Path.cwd(),dict(os.environ))
'''
    parent=subprocess.Popen([sys.executable,'-c',parent_code,str(module),child_code,str(receipt),str(tmp_path/'child.log')]);parent.wait(timeout=5)
    end=time.monotonic()+4
    while not marker.exists() and time.monotonic()<end:time.sleep(.05)
    assert marker.exists() and int(marker.read_text())!=parent.pid
    r=json.loads(receipt.read_text());assert r['status']=='ACCEPTED' and r['session_id']==r['pid']
    spec=importlib.util.spec_from_file_location('launch',module);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    with pytest.raises(FileExistsError):m.launch([],receipt,tmp_path/'duplicate.log',tmp_path,{})

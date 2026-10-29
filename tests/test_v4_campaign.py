"""Frozen-input execution, complete matrix, cost-only extension and LSF safety."""
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import pytest
import torch
from gnn_pruning.experiments import v4_campaign as campaign
from gnn_pruning.pruning.experimental_v4 import PLANS,Q_VARIANTS,GREEDY_VARIANTS

SCRIPTS=Path(__file__).resolve().parents[1]/'research/experiments/polus_v4_diagnostic_20261029/scripts'
def module(name):
    spec=importlib.util.spec_from_file_location(name,SCRIPTS/(name+'.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def config(tmp,variant='q_v4_gamma0'):
    return dict(run_id='toy',variant=variant,model='gcn',target=.3,training_seed=101,policy_seed=101,
        episodes=2,plan=asdict(PLANS[variant]),purpose='smoke',max_steps=20,minimum=4,tolerance=.01,
        alpha=.1,penalty_lambda=2.,threads=2,timeout_seconds=900,platform='local_preflight',
        source_commit='0'*40,output_dir=str(tmp/'outputs'),input_manifest=str(tmp/'inputs.json'))

@pytest.mark.parametrize('extension,count',[(False,92),(True,148)])
def test_primary_and_uniform_extension_matrices(tmp_path,extension,count):
    result=module('prepare_campaign').prepare(tmp_path/'campaign',tmp_path/'inputs.json','a'*40,extension=extension)
    manifest=json.loads(Path(result['manifests']['all']).read_text());configs=[json.loads(Path(t['config']).read_text()) for t in manifest['tasks']]
    assert len(configs)==count and result['greedy_tasks']==8
    for c in configs:campaign.validate_task(c)
    for variant in Q_VARIANTS:
        assert len([c for c in configs if c['variant']==variant])==(20 if extension else 12)
    for variant in GREEDY_VARIANTS:
        assert len([c for c in configs if c['variant']==variant])==4
    assert all(c['variant']!='q_v4_reference' for c in configs)


def test_capacity_decision_ignores_accuracy_and_is_all_or_none():
    prepare=module('prepare_campaign');samples=[]
    for kind in ('gcn','graphsage'):
        for target in (.3,.5):
            for variant in (*Q_VARIANTS,*GREEDY_VARIANTS):
                samples.append(dict(model=kind,target=target,variant=variant,episodes=0 if PLANS[variant].greedy else 30,
                                    search_seconds=2.,worker_wall_seconds=5.,accuracy=.1))
    a=prepare.capacity_decision(samples)
    b=prepare.capacity_decision([dict(s,accuracy=1.) for s in samples])
    assert a==b and a['extension_selected'] and a['campaign_tasks']==148
    slow=[dict(s,search_seconds=150.,worker_wall_seconds=160.) for s in samples]
    assert not prepare.capacity_decision(slow)['primary_fits']
    assert not prepare.capacity_decision(samples,30)['primary_fits']


@pytest.mark.parametrize('field,value',[('tolerance',.02),('minimum',3),('max_steps',21),('training_seed',5001),('episodes',300)])
def test_scientific_setting_mutations_are_rejected(tmp_path,field,value):
    cfg=config(tmp_path);cfg[field]=value
    with pytest.raises(ValueError):campaign.validate_task(cfg)


def test_failed_child_is_not_reported_completed_or_retried(tmp_path,monkeypatch):
    cfg=config(tmp_path);path=tmp_path/'config.json';path.write_text(json.dumps(cfg))
    manifest=dict(source_commit='0'*40,artifact_root=str(tmp_path/'batch'),wall_budget_seconds=30,tasks=[dict(config=str(path))])
    monkeypatch.setattr(campaign,'validate_manifest',lambda _:manifest)
    calls=[]
    class Failed:
        pid=999999
        def __init__(self,cmd,**kwargs):calls.append(cmd)
        def wait(self,**kwargs):return 1
    monkeypatch.setattr(campaign.subprocess,'Popen',Failed)
    with pytest.raises(RuntimeError,match='Incomplete'):campaign.run_batch('unused')
    result=json.loads((tmp_path/'batch/batch.finished.json').read_text())
    assert result['failed_tasks']==1 and not result['all_completed'] and len(calls)==1
    with pytest.raises(FileExistsError):campaign.run_batch('unused')


def test_ambiguous_bsub_acceptance_is_preserved_and_never_retried(tmp_path,monkeypatch):
    submit=module('submit_polus');source=tmp_path/'source';destination=source/'research/experiments/polus_v4_diagnostic_20261029/scripts'
    destination.mkdir(parents=True);(destination/'v4-cpu.lsf').write_text((SCRIPTS/'v4-cpu.lsf').read_text())
    taskroot=tmp_path/'taskroot';taskroot.mkdir();activation=taskroot/'activate-polus.sh';activation.write_text('existing activation')
    import hashlib
    digest=hashlib.sha256(activation.read_bytes()).hexdigest();configs=tmp_path/'configs';configs.mkdir();(configs/'manifest-all.json').write_text('{}')
    cfg=config(tmp_path);cfgpath=tmp_path/'cfg.json';cfgpath.write_text(json.dumps(cfg))
    manifest=dict(source_commit='a'*40,input_manifest='inputs',tasks=[dict(config=str(cfgpath))])
    monkeypatch.setattr(submit,'validate_manifest',lambda _:manifest)
    calls=[]
    class Response:
        returncode=0;stdout='connection lost before acknowledgement';stderr=''
    monkeypatch.setattr(submit.subprocess,'run',lambda *a,**kw:(calls.append(a) or Response()))
    root=tmp_path/'results';result=submit.submit(root,source,configs,taskroot,digest,'capacity')
    assert len(calls)==1 and not result['jobs'][0]['accepted']
    assert (root/'submissions/capacity-all/status').read_text()=='UNRESOLVED\n'
    with pytest.raises(RuntimeError,match='Existing claim'):submit.submit(root,source,configs,taskroot,digest,'capacity')


def test_real_frozen_smoke_and_original_state_are_validation_only(tmp_path,monkeypatch):
    # Reuse the established V3 synthetic frozen-input helper contract, without loading a dataset.
    from gnn_pruning.models import build_model
    from gnn_pruning.config import dump_yaml
    from gnn_pruning.experiments.v3_sprint import file_hash
    from gnn_pruning.pruning.workflow_v2 import tensor_hash,_json_hash
    model=build_model('gcn',in_channels=1433,hidden_channels=32,out_channels=7,num_layers=3,dropout=.5).eval()
    graph=dict(x=torch.zeros(2708,1433),edge_index=torch.tensor([[0,1],[1,0]]),y=torch.arange(2708)%7)
    split=dict(train=list(range(1624)),val=list(range(1624,2166)),test=list(range(2166,2708)))
    torch.save(graph,tmp_path/'graph.pt');dump_yaml(split,tmp_path/'splits.yaml')
    torch.save(dict(model_name='gcn',model_config=dict(in_channels=1433,hidden_channels=32,out_channels=7,num_layers=3,dropout=.5),
        model_state_dict=model.state_dict(),run_signature=dict(dataset='cora',model_name='gcn',seed=101,split_hash=_json_hash(split))),tmp_path/'dense.pt')
    manifest=dict(training_seed=101,graph_tensor_sha256=tensor_hash(graph),graph=dict(file='graph.pt',sha256=file_hash(tmp_path/'graph.pt')),
        split=dict(file='splits.yaml',sha256=file_hash(tmp_path/'splits.yaml')),models={'gcn':dict(file='dense.pt',sha256=file_hash(tmp_path/'dense.pt'),tensor_sha256=tensor_hash(model.state_dict()))})
    (tmp_path/'inputs.json').write_text(json.dumps(manifest));monkeypatch.setattr(campaign,'check_source',lambda _:None)
    record=campaign.run_task(config(tmp_path));report=json.loads((Path(record['output'])/'metrics.json').read_text())
    assert record['status']=='completed' and report['target_hit'] and report['provenance']['label_access']=='validation_only'
    _,data,idx,_,_=campaign.load_frozen(tmp_path/'inputs.json','gcn')
    assert (data.y[split['train']]==-1).all() and (data.y[split['test']]==-1).all()

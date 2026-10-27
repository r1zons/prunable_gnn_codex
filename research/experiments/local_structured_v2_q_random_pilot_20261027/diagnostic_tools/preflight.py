"""Read-only provenance checks and baseline snapshot for the authorized smoke extension."""
import csv, hashlib, json, os, platform, subprocess, sys, tarfile, zipfile
from pathlib import Path
import numpy as np
import pytest
import torch
import torch_geometric
import yaml
from gnn_pruning.data import load_dataset
from gnn_pruning.models import build_model
from gnn_pruning.pruning.workflow_v2 import tensor_hash, _json_hash, _validate_split, _validate_dense_architecture
from gnn_pruning.config import resolve_config
from sklearn.metrics import f1_score

repo=Path.cwd(); artifacts=repo.parent/'prunable_gnn_artifacts'; out=artifacts/'local-structured-v2-q-random-pilot-20261027'; old=artifacts/'local-structured-v2-smoke-20261027'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(name, obj):
    (out/name).write_text(json.dumps(obj,indent=2,allow_nan=False))
torch.set_num_threads(2)
head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(); assert head=='f110c5d1c57bd34afb5ad76e17415ec09ce64e9f'
assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
runtime=dict(python=sys.version,executable=sys.executable,torch=str(torch.__version__),pyg=torch_geometric.__version__,pytest=pytest.__version__,numpy=np.__version__,platform=platform.platform(),threads=torch.get_num_threads(),cuda_available=torch.cuda.is_available(),git_sha=head,git_branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),environment={k:os.environ.get(k) for k in ('PYTHONPATH','PYTHONDONTWRITEBYTECODE','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')})
write('runtime.json',runtime)
prev=json.loads((repo/'research/experiments/local_structured_v2_smoke_20261027/graph_provenance.json').read_text())
assert json.loads((old/'graph_provenance.json').read_text())==prev
for r in json.loads((old/'verification.json').read_text())['runs']:
    for name,digest in r['hashes'].items(): assert sha(Path(r['artifacts'])/name)==digest
archives={
'results':(artifacts/'polus-pilot-20261024/archives/pilot-cpu-101-c4-v2-results.tar.gz','2b97d6923efa6cd8836830fdb7e13872c336da1700b514ab538118d47763fa62'),
'evidence':(artifacts/'polus-pilot-20261024/archives/pilot-cpu-101-c4-v2-evidence.tar.gz','66a905939510edb50e1a64ae814069c91a052fb362e61c6d7fe3cd88ea83aa3f')}
for p,digest in archives.values(): assert sha(p)==digest
source=artifacts/'polus-environment-20261023/polus-research.zip';assert sha(source)==prev['source_archive_sha256']
with zipfile.ZipFile(source) as f: raw_manifest=json.loads(f.read('polus-research/server-evidence/raw-manifest.json'))
assert raw_manifest==json.loads(Path(prev['raw_manifest']).read_text())
for r in prev['raw_files']:assert sha(old/'verified_data'/r['file'])==r['sha256']
# Reuse verified processing without changing or regenerating any previous cache.
processed=old/'verified_data/planetoid/Cora/processed/data.pt';assert processed.is_file()
data=load_dataset('cora',root=str(old/'verified_data'))[0]
assert tensor_hash({k:getattr(data,k) for k in ('x','y','edge_index')})==prev['graph_tensor_sha256']
assert data.x.shape==(2708,1433) and data.edge_index.shape==(2,10556) and data.y.shape==(2708,)
assert torch.all(data.edge_index[0]*2708+data.edge_index[1] == torch.sort(data.edge_index[0]*2708+data.edge_index[1]).values)
# Link each model independently to the verified raw-input preparation and executed job.
selected={};inventory=[]
with tarfile.open(archives['results'][0]) as result_tar, tarfile.open(archives['evidence'][0]) as evidence_tar:
    for model,jobid in [('gcn',1598695),('graphsage',1598696)]:
        condition=f'cora_{model}_l3_h32';job=artifacts/f'polus-pilot-20261024/raw/results/pilot-cpu-101-c4-v2/jobs/{condition}'
        archived_manifest=f'results/pilot-cpu-101-c4-v2/jobs/{condition}/artifact_manifest.csv'
        assert (job/'artifact_manifest.csv').read_bytes()==result_tar.extractfile(archived_manifest).read()
        manifest={r['relative_path']:r for r in csv.DictReader((job/'artifact_manifest.csv').open())}
        d=job/'seed_101'/condition
        files=['dense_checkpoint.pt','splits.yaml','resolved_config.yaml','metrics_train.json']
        for file in files:assert sha(d/file)==manifest[str((d/file).relative_to(job))]['sha256']
        for rel in ('environment.json',f'prepared-configs/{condition}.yaml',f'prepared-configs/{condition}.provenance.json'):
            assert sha(job/rel)==manifest[rel]['sha256']
        log_rel=f'logs/lsf/pilot-cpu.{jobid}.out';log=artifacts/'polus-pilot-20261024/raw'/log_rel
        assert log.read_bytes()==evidence_tar.extractfile(log_rel).read()
        assert condition in log.read_text() and 'Pilot preparation complete' in log.read_text()
        cfg=yaml.safe_load((d/'resolved_config.yaml').read_text());assert cfg['data']['name']=='cora' and f'data-by-condition/pilot-cpu-101-c4-v2/{condition}' in cfg['data']['root']
        arch=dict(in_channels=1433,hidden_channels=32,out_channels=7,num_layers=3,dropout=0.5)
        c=torch.load(d/'dense_checkpoint.pt',map_location='cpu',weights_only=True);assert c['model_name']==model and c['model_config']==arch
        signature=c['run_signature'];split=yaml.safe_load((d/'splits.yaml').read_text());_validate_split(split,2708)
        assert signature['dataset']=='cora' and signature['model_name']==model and signature['seed']==101 and signature['split_hash']==_json_hash(split)
        model_obj=build_model(model,**arch).eval();model_obj.load_state_dict(c['model_state_dict'],strict=True)
        if model=='graphsage':
            assert all(conv.aggr=='mean' and conv.root_weight and not conv.project and not conv.normalize for conv in model_obj.convs)
        with torch.no_grad():logits=model_obj(data)
        assert logits.shape==(2708,7) and torch.isfinite(logits).all()
        idx=torch.tensor(split['val']);true=data.y[idx].numpy();pred=logits.argmax(1)[idx].numpy();metrics=dict(accuracy=float((true==pred).mean()),macro_f1=float(f1_score(true,pred,average='macro',zero_division=0)))
        historical=json.loads((d/'metrics_train.json').read_text())['val']
        for key in metrics:assert abs(metrics[key]-historical[key])<1e-12
        entry=dict(model=model,group='POLUS-CPU-101-C4-V2',seed=101,checkpoint=str(d/'dense_checkpoint.pt'),checkpoint_sha256=sha(d/'dense_checkpoint.pt'),split=str(d/'splits.yaml'),split_sha256=sha(d/'splits.yaml'),split_indices_sha256=_json_hash(split),model_config=arch,run_signature=signature,dense_tensor_sha256=tensor_hash(model_obj.state_dict()),dense_parameters=sum(p.numel() for p in model_obj.parameters()),dense_validation_metrics=metrics,historical_validation_metrics=historical,training_environment=json.loads((job/'environment.json').read_text()),historical_job_id=jobid,job_log=str(log),prepared_config=str(job/f'prepared-configs/{condition}.yaml'),prepared_config_sha256=sha(job/f'prepared-configs/{condition}.yaml'),job_manifest=str(job/'artifact_manifest.csv'),job_manifest_sha256=sha(job/'artifact_manifest.csv'))
        if model=='gcn':assert entry['checkpoint_sha256']==prev['selected']['checkpoint_sha256']
        selected[model]=entry
# Inventory local artifacts without approving their unrecorded training-cache identity.
local_root=artifacts/'local-cpu-20261025/raw';local_manifest={r['relative_path']:r for r in csv.DictReader((local_root/'audit/artifact_manifest.csv').open())}
for model in ('gcn','graphsage'):
    for seed in (101,202,303):
        d=local_root/f'results/seed_{seed}/cora_{model}_l3_h32'
        for name in ('dense_checkpoint.pt','splits.yaml'):assert sha(d/name)==local_manifest[str((d/name).relative_to(local_root))]['sha256']
        inventory.append(dict(model=model,seed=seed,group='LOCAL-CPU-20261025',checkpoint=str(d/'dense_checkpoint.pt'),checkpoint_sha256=sha(d/'dense_checkpoint.pt'),split=str(d/'splits.yaml'),split_sha256=sha(d/'splits.yaml'),selected=False,reason='Historical graph identity not independently hash-attested. Do not infer it from the current local cache.'))
provenance=dict(protocol='LOCAL-STRUCTURED-V2-R2',previous_evidence_commit=head,previous_graph_provenance_file=str(old/'graph_provenance.json'),previous_graph_provenance_sha256=sha(old/'graph_provenance.json'),graph_tensor_sha256=prev['graph_tensor_sha256'],data_root=str(old/'verified_data'),processed_cache=str(processed),processed_cache_sha256=sha(processed),raw_files=prev['raw_files'],input_archives={k:dict(path=str(p),sha256=h) for k,(p,h) in archives.items()},source_archive=dict(path=str(source),sha256=sha(source)),selected=selected,local_checkpoint_inventory=inventory,limitation='No contemporaneous processed tensor hash was recorded at dense training. Reuses manifest-backed reconstruction of original PyG2.4 row ordering; each selected checkpoint is independently linked via archived job/config/input preparation lineage. Not an arbitrary current-cache approval.')
for record in csv.DictReader((repo/'research/experiments/local_structured_v2_cora_matrix_20261027/artifact_manifest.csv').open()):
    assert sha(record['path'])==record['sha256']
write('provenance.json',provenance)
# Protect all previously existing raw artifacts, runs and original cached inputs.
protected={}
for root in (artifacts,repo/'runs',repo/'data'):
    for p in sorted(root.rglob('*')):
        if p.is_file() and out not in p.parents:protected[str(p)]=sha(p)
write('protected_files_before.json',protected)
write('preflight_result.json',dict(status='PASS',selected_models=list(selected),graph_sha256=provenance['graph_tensor_sha256'],protected_files=len(protected),previous_gcn30_not_rerun=True))
print(json.dumps(dict(status='PASS',selected=selected,protected_files=len(protected)),indent=2),flush=True)

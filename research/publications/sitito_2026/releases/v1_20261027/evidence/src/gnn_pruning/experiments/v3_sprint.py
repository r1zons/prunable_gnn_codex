"""Pinned frozen-input V3 tasks and restart-safe bounded subprocess orchestration."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
import uuid

import torch
import torch_geometric
from torch_geometric.data import Data

from gnn_pruning.config import load_yaml
from gnn_pruning.models import build_model
from gnn_pruning.evaluation.metrics import classification_metrics
from gnn_pruning.pruning.experimental_v3 import VARIANTS, WidthReachability, run_variant
from gnn_pruning.pruning.workflow_v2 import tensor_hash, _json_hash, _validate_split
from gnn_pruning.surgery.local_v2 import prune_hidden_channels_local_v2

REPO = Path(__file__).resolve().parents[3]


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, payload):
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False))
    temporary.replace(path)


def check_source(commit):
    """Pin executable source, allowing later receipt-only commits on this branch."""
    if not re.fullmatch('[0-9a-f]{40}', commit):
        raise ValueError('Expected a full source commit SHA.')
    subprocess.run(['git', '-C', str(REPO), 'cat-file', '-e', commit+'^{commit}'], check=True)
    subprocess.run(['git', '-C', str(REPO), 'diff', '--quiet', commit, '--', 'src'], check=True)
    if subprocess.check_output(['git', '-C', str(REPO), 'ls-files', '--others', '--exclude-standard', 'src'], text=True).strip():
        raise ValueError('Untracked executable source is not pinned.')


def load_frozen(input_manifest, model_name):
    """Verify full graph identity first, then expose validation labels only."""
    path = Path(input_manifest);manifest = json.loads(path.read_text());root = path.parent
    entry = manifest['models'][model_name]
    if 'source_provenance_ref' in manifest:
        if file_hash(REPO/manifest['source_provenance_ref'])!=manifest['source_provenance_sha256']:
            raise ValueError('Recorded original graph provenance changed.')
        original=json.loads((REPO/manifest['source_provenance_ref']).read_text())
        if manifest['graph_tensor_sha256']!=original['graph_tensor_sha256']:
            raise ValueError('Original graph provenance pin differs.')
        selected=original['selected'][model_name]
        if (entry['sha256']!=selected['checkpoint_sha256']
                or entry['tensor_sha256']!=selected['dense_tensor_sha256']
                or manifest['split']['sha256']!=selected['split_sha256']):
            raise ValueError('Original checkpoint/split provenance pins differ.')
    for item in (manifest['graph'], manifest['split'], entry):
        if file_hash(root/item['file']) != item['sha256']:
            raise ValueError('Frozen input file hash mismatch.')
    checkpoint = torch.load(root/entry['file'], map_location='cpu', weights_only=True)
    split = load_yaml(root/manifest['split']['file'])
    signature = checkpoint['run_signature']
    if (checkpoint['model_name'] != model_name or signature['model_name'] != model_name
            or signature['dataset'] != 'cora' or signature['seed'] != manifest['training_seed']
            or signature['split_hash'] != _json_hash(split)):
        raise ValueError('Frozen checkpoint identity/split mismatch.')
    expected = dict(in_channels=1433, hidden_channels=32, out_channels=7, num_layers=3, dropout=.5)
    if checkpoint['model_config'] != expected:
        raise ValueError('Expected frozen Cora three-layer hidden32/32 architecture.')
    tensors = torch.load(root/manifest['graph']['file'], map_location='cpu', weights_only=True)
    dtype = next(v.dtype for v in checkpoint['model_state_dict'].values() if v.is_floating_point())
    tensors['x'] = tensors['x'].to(dtype=dtype)
    if tensor_hash(tensors) != manifest['graph_tensor_sha256']:
        raise ValueError('Historical graph tensor identity differs.')
    _validate_split(split, len(tensors['y']))
    idx = torch.tensor(split['val'], dtype=torch.long)
    labels = torch.full_like(tensors['y'], -1);labels[idx] = tensors['y'][idx]
    data = Data(x=tensors['x'], edge_index=tensors['edge_index'], y=labels)
    del tensors
    model = build_model(model_name, **expected).to(dtype=dtype).eval()
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    if tensor_hash(model.state_dict()) != entry['tensor_sha256']:
        raise ValueError('Dense tensor signature mismatch.')
    return model, data, idx, checkpoint, manifest


def metrics(model, data, indices):
    model.eval()
    with torch.no_grad():
        logits = model(data)
    if logits.shape != (2708, 7) or not torch.isfinite(logits).all():
        raise ValueError('Nonfinite or incompatible Cora forward output.')
    values = classification_metrics(data.y[indices].numpy(), logits.argmax(1)[indices].numpy())
    return values, logits


def benchmark(model, data, warmup=5, passes=20):
    """CPU full-graph inference timing, separately from search and reporting."""
    samples = []
    with torch.no_grad():
        for _ in range(warmup):model(data)
        for _ in range(passes):
            start = time.perf_counter();model(data);samples.append(time.perf_counter()-start)
    mean = sum(samples)/passes
    return dict(mean_seconds=mean, std_seconds=(sum((x-mean)**2 for x in samples)/passes)**.5,
                warmup=warmup, passes=passes)


def algorithm_identity(cfg, inputs):
    """Same hash across hardware/paths when algorithm, RNG and frozen inputs match."""
    keys = ('model','variant','target','training_seed','policy_seed','episodes','max_steps',
            'penalty_lambda','gamma','alpha','epsilon','minimum','tolerance','source_commit')
    values = {k: cfg[k] for k in keys}
    values['inputs'] = dict(graph=inputs['graph_tensor_sha256'],split=inputs['split']['sha256'],
                           checkpoint=inputs['models'][cfg['model']]['sha256'])
    return _json_hash(values)


def validate_task(cfg):
    if cfg['variant'] not in (*VARIANTS, 'fixed_probe_v3'):
        raise ValueError('Unknown variant.')
    if cfg['training_seed'] != 101 or cfg['policy_seed'] not in (101,5001,5002):
        raise ValueError('Unexpected training/policy seed.')
    if cfg['target'] not in (.3,.5) or cfg['threads'] not in (1,2,4):
        raise ValueError('Unapproved target/thread count.')
    if cfg['episodes'] != (2 if cfg.get('purpose')=='smoke' else 100) or cfg['max_steps'] != 20:
        raise ValueError('Invalid diagnostic budget.')
    if (cfg['penalty_lambda'],cfg['gamma'],cfg['alpha'],cfg['epsilon'],cfg['minimum'],cfg['tolerance']) != (2.,.9,.1,.2,4,.01):
        raise ValueError('Scientific settings differ from the approved sprint.')


def run_task(cfg):
    """A fresh attempt; previous attempts/results are never overwritten."""
    validate_task(cfg);check_source(cfg['source_commit'])
    started = time.perf_counter()
    model,data,indices,source,inputs = load_frozen(cfg['input_manifest'],cfg['model'])
    identity = algorithm_identity(cfg,inputs);dense_hash = tensor_hash(model.state_dict())
    dense_metrics,dense_logits = metrics(model,data,indices)
    output = Path(cfg['output_dir'])/('attempt-'+uuid.uuid4().hex[:12]);output.mkdir(parents=True,exist_ok=False)
    calls = 0
    def evaluate(candidate):
        nonlocal calls
        calls += 1
        with torch.no_grad():predicted = candidate(data).argmax(1)[indices]
        return float((predicted==data.y[indices]).float().mean())
    start = time.perf_counter()
    if cfg['variant'] == 'fixed_probe_v3':
        final = model;trace = [];ids = [list(range(32)),list(range(32))]
        # Explicitly fixed keeps: remove highest current IDs, independent of saliency.
        for layer,k in ((0,1),(1,2),(0,4)):
            keep = list(range(final.hidden_channel_dims[layer]-k))
            final = prune_hidden_channels_local_v2(final,layer,keep)
            ids[layer] = [ids[layer][i] for i in keep]
            trace.append(dict(layer=layer,k=k,keep_current=keep,original_kept_ids=[list(x) for x in ids]))
        training,qtable,statistics = [],[],dict(terminal_reason='fixed_sequence_complete',validation_evaluations=0)
    else:
        result = run_variant(model,evaluate,variant=cfg['variant'],target=cfg['target'],
            penalty_lambda=cfg['penalty_lambda'],episodes=cfg['episodes'],max_steps=cfg['max_steps'],
            gamma=cfg['gamma'],alpha=cfg['alpha'],epsilon=cfg['epsilon'],seed=cfg['policy_seed'],
            min_channels=cfg['minimum'],tolerance=cfg['tolerance'])
        final,ids,trace,training,statistics = result.model,result.kept_indices,result.trace,result.training_trace,result.statistics
        qtable = [dict(state=s,actions=row) for s,row in result.q_table.items()]
        if calls != statistics['validation_evaluations']:
            raise RuntimeError('Validation evaluation counter mismatch.')
    search_seconds = time.perf_counter()-start
    final_metrics,final_logits = metrics(final,data,indices)
    # Independent physical projection from original dense channel identities.
    for name,value in final.state_dict().items():
        layer = int(name.split('.')[1]);expected = source['model_state_dict'][name]
        if layer<2:expected=expected[ids[layer]]
        if layer>0 and expected.ndim==2:expected=expected[:,ids[layer-1]]
        if not torch.equal(value,expected):raise RuntimeError('Physical checkpoint differs from original kept IDs.')
    if tensor_hash(model.state_dict()) != dense_hash:
        raise RuntimeError('Source dense model was mutated.')
    architecture=final.export_architecture_config();architecture.pop('name')
    checkpoint=output/'pruned.pt'
    provenance=dict(source_commit=cfg['source_commit'],algorithm_identity_sha256=identity,
        graph_tensor_sha256=inputs['graph_tensor_sha256'],split_sha256=inputs['split']['sha256'],
        checkpoint_sha256=inputs['models'][cfg['model']]['sha256'],training_seed=101,
        policy_seed=cfg['policy_seed'],platform=cfg['platform'],label_access='validation_only')
    torch.save(dict(model_name=cfg['model'],model_config=architecture,model_state_dict=final.state_dict(),
                    original_kept_ids=ids,variant=cfg['variant'],provenance=provenance),checkpoint)
    reloaded=torch.load(checkpoint,map_location='cpu',weights_only=True)
    rebuilt=build_model(reloaded['model_name'],**reloaded['model_config']).to(dtype=data.x.dtype).eval()
    rebuilt.load_state_dict(reloaded['model_state_dict'],strict=True)
    _,reloaded_logits=metrics(rebuilt,data,indices)
    if not torch.equal(reloaded_logits,final_logits):raise RuntimeError('Reload changed final logits.')
    p0=sum(p.numel() for p in model.parameters());after=sum(p.numel() for p in final.parameters());achieved=1-after/p0
    space=WidthReachability.from_model(model,cfg['target'],cfg['tolerance'],cfg['minimum'])
    if after != space.count(final.hidden_channel_dims):raise RuntimeError('Parameter report mismatch.')
    report=dict(run_id=cfg['run_id'],variant=cfg['variant'],model=cfg['model'],provenance=provenance,
        hidden_widths=final.hidden_channel_dims,original_kept_ids=ids,parameters_before=p0,parameters_after=after,
        achieved_sparsity=achieved,target=cfg['target'],signed_target_deviation=achieved-cfg['target'],
        target_gap=abs(achieved-cfg['target']),target_violation=not space.in_band(final.hidden_channel_dims),
        dense_validation=dense_metrics,final_validation=final_metrics,statistics=statistics,search_seconds=search_seconds,
        inference=benchmark(final,data) if cfg['measure_inference'] else None,
        runtime=dict(host=socket.gethostname(),machine=os.uname().machine,python=sys.version,
                     torch=str(torch.__version__),pyg=torch_geometric.__version__,threads=torch.get_num_threads(),
                     interop_threads=torch.get_num_interop_threads(),backend=torch.__config__.parallel_info()),
        total_seconds=time.perf_counter()-started)
    if cfg['variant']=='fixed_probe_v3':
        torch.save(dense_logits,output/'dense_logits.pt');torch.save(final_logits,output/'fixed_pruned_logits.pt')
        report['dense_inference']=benchmark(model,data)
    write_json(output/'metrics.json',report);write_json(output/'config.json',cfg)
    write_json(output/'trace.json',dict(deployment=trace,training=training));write_json(output/'q_table.json',qtable)
    record=dict(status='completed',run_id=cfg['run_id'],source_commit=cfg['source_commit'],
                algorithm_identity_sha256=identity,config_sha256=_json_hash(cfg),output=str(output),
                hashes={p.name:file_hash(p) for p in output.iterdir() if p.is_file()})
    write_json(output/'completed.json',record)
    return record


def validate_manifest(path):
    manifest=json.loads(Path(path).read_text());check_source(manifest['source_commit'])
    if manifest['workers'] not in (1,2):raise ValueError('Bounded parallelism required.')
    if 'source_provenance_ref' not in json.loads(Path(manifest['input_manifest']).read_text()):
        raise ValueError('Scientific batch requires original recorded graph provenance.')
    tasks=manifest['tasks'];seen=set()
    for task in tasks:
        cfg=json.loads(Path(task['config']).read_text());validate_task(cfg)
        if cfg['run_id'] in seen:raise ValueError('Duplicate run ID.')
        seen.add(cfg['run_id'])
        if cfg['source_commit']!=manifest['source_commit'] or file_hash(task['config'])!=task['config_sha256']:
            raise ValueError('Config/source pin mismatch.')
    for kind in sorted({json.loads(Path(t['config']).read_text())['model'] for t in tasks}):
        model,_,_,_,_=load_frozen(manifest['input_manifest'],kind)
        for target in (.3,.5):
            if not WidthReachability.from_model(model,target,.01,4).can_reach((32,32),20):
                raise ValueError('Approved Cora target is initially infeasible.')
    return manifest


def _alive(pid):
    try:os.kill(int(pid),0);return True
    except ProcessLookupError:return False


def run_batch(path,resume=False):
    manifest=validate_manifest(path);root=Path(manifest['artifact_root']);root.mkdir(parents=True,exist_ok=True)
    lock=root/'batch.lock'
    if lock.exists():
        previous=json.loads((lock/'owner.json').read_text())
        if not resume or previous['host']!=socket.gethostname() or _alive(previous['pid']):
            raise RuntimeError('Existing live/ambiguous batch claim; do not duplicate it.')
        lock.rename(root/('batch.lock.stale-'+uuid.uuid4().hex[:12]))
    lock.mkdir();write_json(lock/'owner.json',dict(pid=os.getpid(),host=socket.gethostname(),lsf_job=os.environ.get('LSB_JOBID')))
    deadline=time.monotonic()+manifest.get('wall_budget_seconds',10800)
    def one(task):
        cfg=json.loads(Path(task['config']).read_text());status=root/'status'/(cfg['run_id']+'.json')
        if time.monotonic()>=deadline:
            write_json(status,dict(status='wall_budget_exhausted'));return False
        if status.exists():
            old=json.loads(status.read_text())
            if old.get('status')=='completed':
                if old['source_commit']!=cfg['source_commit']:raise RuntimeError('Completed source differs.')
                if old.get('config_sha256')!=_json_hash(cfg):raise RuntimeError('Completed configuration differs.')
                for name,digest in old['hashes'].items():
                    if file_hash(Path(old['output'])/name)!=digest:raise RuntimeError('Completed evidence changed.')
                return True
            if not resume:raise RuntimeError('Prior task attempt exists; explicit safe resume required.')
            if old.get('host')==socket.gethostname() and old.get('child_pid') and _alive(old['child_pid']):
                raise RuntimeError('Prior scientific subprocess may still be alive.')
        log=root/'logs'/(cfg['run_id']+'-'+uuid.uuid4().hex[:8]+'.log');log.parent.mkdir(exist_ok=True)
        env=dict(os.environ,PYTHONPATH=str(REPO/'src'),PYTHONDONTWRITEBYTECODE='1',
                 OMP_NUM_THREADS=str(cfg['threads']),MKL_NUM_THREADS=str(cfg['threads']),OPENBLAS_NUM_THREADS=str(cfg['threads']))
        with log.open('x') as stream:
            child=subprocess.Popen([sys.executable,'-u','-m','gnn_pruning.experiments.v3_sprint','task','--config',task['config']],cwd=REPO,env=env,stdout=stream,stderr=subprocess.STDOUT)
            write_json(status,dict(status='running',child_pid=child.pid,host=socket.gethostname(),log=str(log),config=task['config']))
            try:code=child.wait(timeout=min(cfg['timeout_seconds'],max(1,deadline-time.monotonic())))
            except subprocess.TimeoutExpired:
                child.terminate()
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:child.kill();child.wait()
                code='timeout'
        if code:write_json(status,dict(status='failed',return_code=code,log=str(log),host=socket.gethostname()));return False
        attempts=sorted(Path(cfg['output_dir']).glob('attempt-*/completed.json'),key=lambda p:p.stat().st_mtime)
        record=json.loads(attempts[-1].read_text());record['log']=str(log);write_json(status,record);return True
    # Established local launcher pattern: bounded threads orchestrate independent Python processes.
    with concurrent.futures.ThreadPoolExecutor(max_workers=manifest['workers']) as pool:
        outcomes=list(pool.map(one,manifest['tasks']))
    write_json(root/'batch.finished.json',dict(finished=True,all_completed=all(outcomes),completed_tasks=sum(outcomes),failed_tasks=len(outcomes)-sum(outcomes),source_commit=manifest['source_commit']))
    if not all(outcomes):raise RuntimeError('At least one task failed; inspect preserved status and attempts.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('task','batch','validate'))
    parser.add_argument('--config');parser.add_argument('--manifest');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.mode=='task':
        cfg=json.loads(Path(args.config).read_text());torch.set_num_threads(cfg['threads']);torch.set_num_interop_threads(1)
        print(json.dumps(run_task(cfg)))
    elif args.mode=='validate':print(json.dumps(dict(validated=len(validate_manifest(args.manifest)['tasks']))))
    else:run_batch(args.manifest,args.resume)


if __name__=='__main__':main()

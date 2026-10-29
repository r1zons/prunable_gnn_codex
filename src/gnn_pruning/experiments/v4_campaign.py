"""Pinned V4 task/batch execution, adapted from V3 without modifying its files."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import uuid
import torch
import torch_geometric
from gnn_pruning.models import build_model
from gnn_pruning.pruning.experimental_v4 import (PLANS, Q_VARIANTS, GREEDY_VARIANTS,
    REFERENCE, MenuReachability, run_variant)
from gnn_pruning.pruning.experimental_v3 import WidthReachability, run_variant as run_v3
from gnn_pruning.pruning.workflow_v2 import _json_hash, tensor_hash
from .v3_sprint import load_frozen, metrics, file_hash, write_json, check_source, REPO

EVIDENCE='research/experiments/polus_v4_diagnostic_20261029'


def scientific_identity(cfg, inputs):
    """Explicit new-source identity; reused V3 references retain their old provenance."""
    values={k:cfg[k] for k in ('model','variant','target','policy_seed','training_seed','source_commit','purpose','episodes')}
    values.update(plan=asdict(PLANS[cfg['variant']]),max_steps=20,minimum=4,tolerance=.01,
                  state_version='local_v2',alpha=.1,penalty_lambda=2.,
                  graph=inputs['graph_tensor_sha256'],split=inputs['split']['sha256'],
                  checkpoint=inputs['models'][cfg['model']]['sha256'])
    return _json_hash(values)


def validate_task(cfg):
    """Fail on any setting outside the approved matrix or explicitly truncated gate."""
    if cfg['variant'] not in (*Q_VARIANTS,*GREEDY_VARIANTS):raise ValueError('Not a new campaign variant.')
    if cfg['training_seed']!=101 or cfg['policy_seed'] not in (101,5001,5002,5003,5004):raise ValueError('Seed mismatch.')
    if cfg['target'] not in (.3,.5) or cfg['model'] not in ('gcn','graphsage'):raise ValueError('Condition mismatch.')
    if (cfg['max_steps'],cfg['minimum'],cfg['tolerance'],cfg['alpha'],cfg['penalty_lambda'])!=(20,4,.01,.1,2.):raise ValueError('Fixed settings changed.')
    if cfg['plan']!=asdict(PLANS[cfg['variant']]):raise ValueError('Single-factor settings changed.')
    plan=PLANS[cfg['variant']]
    if plan.greedy:
        if cfg['policy_seed']!=101 or cfg['episodes']!=0:raise ValueError('Greedy must be deterministic, run once.')
    elif cfg['purpose']=='campaign':
        if cfg['episodes']!=plan.episodes:raise ValueError('Full episode budget required.')
    elif cfg['purpose'] in ('smoke','capacity'):
        if not 1<=cfg['episodes']<=30 or cfg['policy_seed']!=101:raise ValueError('Unbounded preflight.')
    else:raise ValueError('Unknown purpose.')
    if cfg['threads'] not in (2,4):raise ValueError('Unapproved threads.')
    if not 0<cfg['timeout_seconds']<=1800:raise ValueError('Unbounded task timeout.')


def validate_manifest(path):
    """All manifest and frozen-input gates precede execution/submission."""
    manifest=json.loads(Path(path).read_text());check_source(manifest['source_commit'])
    subprocess.run(['git','-C',str(REPO),'diff','--quiet',manifest['source_commit'],'--',EVIDENCE+'/scripts'],check=True)
    if manifest['workers']!=1 or not 0<manifest['wall_budget_seconds']<=4200:raise ValueError('Job resource bound changed.')
    configs=[];seen=set()
    for task in manifest['tasks']:
        cfg=json.loads(Path(task['config']).read_text());validate_task(cfg)
        if cfg['run_id'] in seen or cfg['source_commit']!=manifest['source_commit']:raise ValueError('Duplicate/source mismatch.')
        if file_hash(task['config'])!=task['config_sha256']:raise ValueError('Config file changed.')
        if cfg['input_manifest']!=manifest['input_manifest']:raise ValueError('Input path differs.')
        seen.add(cfg['run_id']);configs.append(cfg)
    for kind in {c['model'] for c in configs}:
        model,_,_,_,inputs=load_frozen(manifest['input_manifest'],kind)
        if inputs.get('source_provenance_ref') is None:raise ValueError('Original graph provenance required.')
        for cfg in [c for c in configs if c['model']==kind]:
            geometry=WidthReachability.from_model(model,cfg['target'],.01,4)
            if not MenuReachability(geometry,PLANS[cfg['variant']].menu).can_reach((32,32),20):raise ValueError('Initial target infeasible.')
    return manifest


def run_task(cfg):
    """Fresh immutable attempt with physical reload/projection and validation only."""
    validate_task(cfg);check_source(cfg['source_commit']);started=time.perf_counter()
    model,data,indices,source,inputs=load_frozen(cfg['input_manifest'],cfg['model'])
    dense_hash=tensor_hash(model.state_dict());dense_metrics,_=metrics(model,data,indices)
    output=Path(cfg['output_dir'])/('attempt-'+uuid.uuid4().hex[:12]);output.mkdir(parents=True,exist_ok=False)
    write_json(output/'config.json',cfg);calls=0
    def evaluate(candidate):
        nonlocal calls
        calls+=1
        with torch.no_grad():pred=candidate(data).argmax(1)[indices]
        return float((pred==data.y[indices]).float().mean())
    start=time.perf_counter()
    result=run_variant(model,evaluate,cfg['variant'],cfg['target'],seed=cfg['policy_seed'],
        episodes_override=cfg['episodes'] if cfg['purpose']!='campaign' and not PLANS[cfg['variant']].greedy else None)
    search_seconds=time.perf_counter()-start
    if calls!=result.statistics['validation_evaluations']:raise RuntimeError('Evaluation counter differs.')
    final=result.model;ids=result.kept_indices
    final_metrics,logits=metrics(final,data,indices)
    for name,value in final.state_dict().items():
        layer=int(name.split('.')[1]);original=source['model_state_dict'][name]
        if layer<2:original=original[ids[layer]]
        if layer>0 and original.ndim==2:original=original[:,ids[layer-1]]
        if not torch.equal(value,original):raise RuntimeError('Physical weights not a dense projection.')
    if tensor_hash(model.state_dict())!=dense_hash:raise RuntimeError('Dense input mutated.')
    geometry=WidthReachability.from_model(model,cfg['target'],.01,4)
    after=sum(p.numel() for p in final.parameters());before=sum(p.numel() for p in model.parameters())
    if after!=geometry.count(final.hidden_channel_dims) or not after<before or not geometry.in_band(final.hidden_channel_dims):
        raise RuntimeError('Physical count or exact band invariant failed.')
    identity=scientific_identity(cfg,inputs)
    provenance=dict(source_commit=cfg['source_commit'],algorithm_identity_sha256=identity,
        graph_tensor_sha256=inputs['graph_tensor_sha256'],split_sha256=inputs['split']['sha256'],
        checkpoint_sha256=inputs['models'][cfg['model']]['sha256'],training_seed=101,policy_seed=cfg['policy_seed'],
        platform=cfg['platform'],label_access='validation_only',purpose=cfg['purpose'])
    architecture=final.export_architecture_config();architecture.pop('name')
    checkpoint=output/'pruned.pt'
    torch.save(dict(model_name=cfg['model'],model_config=architecture,model_state_dict=final.state_dict(),
                    original_kept_ids=ids,variant=cfg['variant'],provenance=provenance),checkpoint)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    rebuilt=build_model(saved['model_name'],**saved['model_config']).eval();rebuilt.load_state_dict(saved['model_state_dict'],strict=True)
    _,reloaded_logits=metrics(rebuilt,data,indices)
    if not torch.equal(logits,reloaded_logits):raise RuntimeError('Reload changed forward.')
    write_json(output/'trace.json',dict(training=result.training_trace,deployment=result.trace))
    write_json(output/'decisions.json',result.decisions)
    write_json(output/'q_table.json',[dict(state=s,actions=row) for s,row in result.q_table.items()])
    for episode,snapshot in result.snapshots.items():write_json(output/f'policy_snapshot_{episode:03}.json',snapshot)
    report=dict(run_id=cfg['run_id'],variant=cfg['variant'],model=cfg['model'],provenance=provenance,
        target=cfg['target'],tolerance=.01,achieved_sparsity=1-after/before,target_hit=True,
        hidden_widths=final.hidden_channel_dims,original_kept_ids=ids,parameters_before=before,parameters_after=after,
        parameter_bytes=sum(p.numel()*p.element_size() for p in final.parameters()),checkpoint_bytes=checkpoint.stat().st_size,
        dense_validation=dense_metrics,final_validation=final_metrics,statistics=result.statistics,
        search_seconds=search_seconds,total_seconds=time.perf_counter()-started,
        runtime=dict(host=socket.gethostname(),machine=os.uname().machine,python=sys.version,torch=str(torch.__version__),
                     pyg=torch_geometric.__version__,threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),backend=torch.__config__.parallel_info()))
    write_json(output/'metrics.json',report)
    record=dict(status='completed',run_id=cfg['run_id'],source_commit=cfg['source_commit'],algorithm_identity_sha256=identity,
        config_sha256=_json_hash(cfg),output=str(output),hashes={p.name:file_hash(p) for p in output.iterdir() if p.is_file()})
    write_json(output/'completed.json',record)
    return record


def run_batch(path):
    """One scientific process at a time; collisions/ambiguity fail, never auto-resume."""
    manifest=validate_manifest(path);root=Path(manifest['artifact_root']);root.mkdir(parents=True,exist_ok=True)
    lock=root/'batch.lock';lock.mkdir()
    write_json(lock/'owner.json',dict(pid=os.getpid(),host=socket.gethostname(),lsf_job=os.environ.get('LSB_JOBID')))
    deadline=time.monotonic()+manifest['wall_budget_seconds'];outcomes=[]
    for task in manifest['tasks']:
        cfg=json.loads(Path(task['config']).read_text());remaining=deadline-time.monotonic()
        if remaining<2:
            outcomes.append(dict(run_id=cfg['run_id'],status='not_started_budget_exhausted'));continue
        status=root/'status'/(cfg['run_id']+'.json')
        if status.exists():raise RuntimeError('Prior task claim; inspect instead of duplicating.')
        log=root/'logs'/(cfg['run_id']+'.log');log.parent.mkdir(exist_ok=True)
        env=dict(os.environ,PYTHONPATH=str(REPO/'src'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS=str(cfg['threads']),MKL_NUM_THREADS=str(cfg['threads']),OPENBLAS_NUM_THREADS=str(cfg['threads']))
        with log.open('x') as stream:
            worker_started=time.perf_counter()
            child=subprocess.Popen([sys.executable,'-u','-m','gnn_pruning.experiments.v4_campaign','task','--config',task['config']],cwd=REPO,env=env,stdout=stream,stderr=subprocess.STDOUT)
            write_json(status,dict(status='running',child_pid=child.pid,host=socket.gethostname(),log=str(log)))
            try:code=child.wait(timeout=min(cfg['timeout_seconds'],remaining))
            except subprocess.TimeoutExpired:
                child.terminate()
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:child.kill();child.wait()
                code='timeout'
        if code:record=dict(run_id=cfg['run_id'],status='failed',return_code=code,log=str(log))
        else:
            attempts=list(Path(cfg['output_dir']).glob('attempt-*/completed.json'))
            if len(attempts)!=1:raise RuntimeError('Ambiguous completed attempts.')
            record=json.loads(attempts[0].read_text());record['log']=str(log)
        record['worker_wall_seconds']=time.perf_counter()-worker_started
        write_json(status,record);outcomes.append(record)
    completed=sum(o['status']=='completed' for o in outcomes)
    write_json(root/'batch.finished.json',dict(finished=True,all_completed=completed==len(outcomes),completed_tasks=completed,
        failed_tasks=len(outcomes)-completed,source_commit=manifest['source_commit'],outcomes=outcomes))
    if completed!=len(outcomes):raise RuntimeError('Incomplete batch; preserved evidence, no retries.')


def compatibility(inputs):
    """Bounded real-Cora V4 reference/V3 equality at two episodes; no full rerun."""
    checks=[]
    for kind in ('gcn','graphsage'):
        model,data,idx,_,_=load_frozen(inputs,kind)
        def evaluate(candidate):
            with torch.no_grad():return float((candidate(data).argmax(1)[idx]==data.y[idx]).float().mean())
        for target in (.3,.5):
            a=run_v3(model,evaluate,'q_reachable_v3',target,2.,episodes=2,seed=101)
            b=run_variant(model,evaluate,REFERENCE,target,episodes_override=2,seed=101)
            if a.trace!=b.trace or a.training_trace!=b.training_trace or a.q_table!=b.q_table:raise RuntimeError('V3 reference regression.')
            for name,value in a.model.state_dict().items():
                if not torch.equal(value,b.model.state_dict()[name]):raise RuntimeError('Reference weights differ.')
            checks.append(dict(model=kind,target=target,episodes=2,status='PASS'))
    return dict(status='PASS',checks=checks)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('task','batch','validate','compatibility'))
    p.add_argument('--config');p.add_argument('--manifest');p.add_argument('--inputs');p.add_argument('--threads',type=int,default=4);a=p.parse_args()
    if a.mode in ('task','compatibility'):
        threads=json.loads(Path(a.config).read_text())['threads'] if a.config else a.threads
        torch.set_num_threads(threads);torch.set_num_interop_threads(1)
    if a.mode=='task':print(json.dumps(run_task(json.loads(Path(a.config).read_text()))))
    elif a.mode=='batch':run_batch(a.manifest)
    elif a.mode=='validate':print(json.dumps(dict(validated=len(validate_manifest(a.manifest)['tasks']))))
    else:print(json.dumps(compatibility(a.inputs)))


if __name__=='__main__':main()

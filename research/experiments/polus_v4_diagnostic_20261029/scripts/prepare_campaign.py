"""Prepare collision-proof V4 matrices and make a cost-only extension decision."""
from dataclasses import asdict
import argparse
import hashlib
import json
import math
from pathlib import Path
from gnn_pruning.pruning.experimental_v4 import PLANS,Q_VARIANTS,GREEDY_VARIANTS

PREFLIGHT_WALL_MINUTES=15
CAMPAIGN_WALL_MINUTES=70
AGGREGATE_LIMIT_MINUTES=300


def prepare(root,inputs,source,stage='campaign',extension=False,threads=4,job_wall_minutes=None):
    """92 primary tasks or148 with uniform two-seed extension;36 capacity tasks."""
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False)
    purpose='capacity' if stage=='capacity' else 'smoke' if stage=='smoke' else 'campaign'
    seeds=(101,) if purpose!='campaign' else (101,5001,5002,5003,5004) if extension else (101,5001,5002)
    groups={};tasks=[]
    for kind in ('gcn','graphsage'):
        for target in (.3,.5):
            name=f'{kind}-t{round(100*target)}';groups[name]=[]
            for variant in (*Q_VARIANTS,*GREEDY_VARIANTS):
                plan=PLANS[variant]
                for seed in ((101,) if plan.greedy else seeds):
                    runid=f'polus-v4-{kind}-t{round(100*target)}-{variant}-p{seed}'
                    episodes=0 if plan.greedy else (30 if purpose=='capacity' else 2 if purpose=='smoke' else plan.episodes)
                    cfg=dict(run_id=runid,model=kind,variant=variant,target=target,training_seed=101,policy_seed=seed,
                        episodes=episodes,plan=asdict(plan),purpose=purpose,max_steps=20,minimum=4,tolerance=.01,alpha=.1,penalty_lambda=2.,
                        source_commit=source,input_manifest=str(Path(inputs).resolve()),output_dir=str(root/'outputs'/runid),
                        platform='polus' if threads==4 else 'local_preflight',threads=threads,timeout_seconds=1800)
                    path=root/'configs'/(runid+'.json');path.parent.mkdir(exist_ok=True);path.write_text(json.dumps(cfg,indent=2))
                    task=dict(config=str(path),config_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                    tasks.append(task);groups[name].append(task)
    manifests={}
    for name,subset in [('all',tasks),*groups.items()]:
        minutes=(job_wall_minutes.get(name,max(job_wall_minutes.values())) if job_wall_minutes else CAMPAIGN_WALL_MINUTES) if purpose=='campaign' else PREFLIGHT_WALL_MINUTES
        wall=minutes*60-(120 if purpose=='campaign' else 90)
        path=root/f'manifest-{name}.json'
        payload=dict(source_commit=source,input_manifest=str(Path(inputs).resolve()),workers=1,
                     artifact_root=str(root/'execution'/name),wall_budget_seconds=wall,lsf_wall_minutes=minutes,tasks=subset)
        path.write_text(json.dumps(payload,indent=2));manifests[name]=str(path)
    return dict(root=str(root),tasks=len(tasks),q_tasks=sum(not PLANS[json.loads(Path(t['config']).read_text())['variant']].greedy for t in tasks),
                greedy_tasks=8,extension=extension,manifests=manifests)


def capacity_decision(samples,accepted_preflight_wall_minutes=15):
    """No accuracy/quality inputs: conservative projections from all measured cells."""
    if len(samples)!=36:raise ValueError('Every model/target/variant capacity sample is required.')
    estimates={};groups={}
    for sample in samples:
        plan=PLANS[sample['variant']];key=(sample['model'],sample['target'],sample['variant'])
        if key in estimates:raise ValueError('Duplicate capacity cell.')
        episodes=0 if plan.greedy else plan.episodes
        # Include observed interpreter/serialization overhead, with15 seconds floor.
        search=sample['search_seconds'] if plan.greedy else sample['search_seconds']*episodes/sample['episodes']
        overhead=max(15.,sample['worker_wall_seconds']-sample['search_seconds'])
        estimates[key]=dict(unpadded_seconds=search+overhead,planning_seconds=2*(search+overhead))
    for kind in ('gcn','graphsage'):
        for target in (.3,.5):
            costs=[(PLANS[v].greedy,estimates[kind,target,v]) for v in (*Q_VARIANTS,*GREEDY_VARIANTS)]
            primary=sum((1 if greedy else 3)*e['planning_seconds'] for greedy,e in costs)
            extended=sum((1 if greedy else 5)*e['planning_seconds'] for greedy,e in costs)
            primary_raw=sum((1 if greedy else 3)*e['unpadded_seconds'] for greedy,e in costs)
            extended_raw=sum((1 if greedy else 5)*e['unpadded_seconds'] for greedy,e in costs)
            groups[f'{kind}-t{round(100*target)}']=dict(primary_planning_seconds=primary,extended_planning_seconds=extended,
                primary_unpadded_seconds=primary_raw,extended_unpadded_seconds=extended_raw,
                primary_lsf_wall_minutes=math.ceil(primary/(.8*60)+2),
                extended_lsf_wall_minutes=math.ceil(extended/(.8*60)+2))
    primary_bound=accepted_preflight_wall_minutes+sum(g['primary_lsf_wall_minutes'] for g in groups.values())
    extended_bound=accepted_preflight_wall_minutes+sum(g['extended_lsf_wall_minutes'] for g in groups.values())
    primary_fits=primary_bound<=300 and all(g['primary_lsf_wall_minutes']<=180 for g in groups.values())
    extension_fits=primary_fits and extended_bound<=300 and all(g['extended_lsf_wall_minutes']<=180 for g in groups.values())
    for group in groups.values():group['selected_lsf_wall_minutes']=group['extended_lsf_wall_minutes'] if extension_fits else group['primary_lsf_wall_minutes']
    hard_bound=extended_bound if extension_fits else primary_bound
    return dict(status='PASS' if primary_fits else 'BLOCKED_CAPACITY',primary_fits=primary_fits,extension_selected=extension_fits,
        primary_policy_seeds=[101,5001,5002],optional_predeclared_seeds=[5003,5004],extension_variants=list(Q_VARIANTS),
        extension_rule='Both seeds added to every one of the seven new Q variants in every model/target cell, or none.',
        campaign_tasks=148 if extension_fits else 92,hard_aggregate_wall_limit_minutes=hard_bound,
        preflight_reserved_wall_minutes=accepted_preflight_wall_minutes,campaign_jobs=4,
        campaign_wall_minutes={k:g['selected_lsf_wall_minutes'] for k,g in groups.items()},
        primary_reserved_minutes=primary_bound,extended_reserved_minutes=extended_bound,
        maximum_budget_minutes=300,planning_safety_factor=2,job_headroom_fraction=.2,groups=groups,
        inputs_used='Search time, actual worker wall time, planned episode budgets and resource bounds only; no accuracy.',
        target_note='4–5h is a planning target, never artificial idle time or authorization for extra scientific conditions.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--inputs',required=True);p.add_argument('--source',required=True)
    p.add_argument('--stage',choices=('campaign','capacity','smoke'),default='campaign');p.add_argument('--extension',action='store_true')
    p.add_argument('--threads',type=int,choices=(2,4),default=4);a=p.parse_args()
    print(json.dumps(prepare(a.root,a.inputs,a.source,a.stage,a.extension,a.threads),indent=2))

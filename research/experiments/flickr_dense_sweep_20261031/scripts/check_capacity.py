"""Review all three native technical jobs; select resources without accuracy filtering."""
import argparse
import json
import math
from pathlib import Path
import statistics
import time

from gnn_pruning.config import load_yaml
from gnn_pruning.experiments.flickr_dense import matrix, write
from gnn_pruning.experiments.flickr_lsf import preflight_trials, research_jobs


def work(model, layers, width):
    dims=[500]+[width]*(layers-1)+[7]
    linear=89250*sum(a*b for a,b in zip(dims,dims[1:]))
    return linear+899756*sum(dims[1:]) if model=='gcn' else 2*linear+899756*sum(dims[:-1])


def check(root: Path, protocol: dict, output: Path):
    reports={};clock={};evidence=[]
    for job in sorted((root/'preflight').iterdir()):
        group={};frequencies=[]
        for trial in preflight_trials():
            path=job/trial['id']
            completed=json.loads((path/'completed.json').read_text())
            from gnn_pruning.experiments.flickr_dense import sha
            if any(sha(path/name)!=value for name,value in completed.items()):raise ValueError('Technical artifact hash differs.')
            r=json.loads((path/'report.json').read_text());env=json.loads((path/'environment_after.json').read_text())
            if not r['technical'] or r['source_commit']!=protocol['source_commit'] or not r['strict_reload_verified']:
                raise ValueError('Native technical compatibility failed.')
            if not r['convergence']['finite'] or r['test_metrics_evaluated']:
                raise ValueError('Nonfinite or test-contaminated technical run.')
            group[trial['id']]=r
            freq=env['proc_cpu_clock_mhz_median']
            if freq:frequencies.append(freq)
            evidence.append(dict(job=job.name,trial=trial['id'],actual_host=r['actual_host'],epoch_median=r['epoch_seconds_median'],
                                 epoch_max=r['epoch_seconds_max'],peak_rss_bytes=r['peak_process_rss_bytes'],clock_mhz=freq,
                                 source_commit=r['source_commit']))
        hosts={r['actual_host'] for r in group.values()}
        if len(hosts)!=1 or not frequencies:raise ValueError('Missing unambiguous actual-host/CPU-clock evidence.')
        host=hosts.pop();reports[host]=group;clock[host]=statistics.median(frequencies)
    if len(reports)!=3:raise ValueError('All three compute nodes require completed preflight.')
    c4=next(h for h in reports if h.startswith('polus-c4-'))
    conditions={}
    for condition in matrix(protocol):
        predictions={}
        for host,group in reports.items():
            model=condition['model'];small=group[model+'-small-lr0p001'];maximum=group[model+'-maximum']
            actual=work(model,condition['layers'],condition['width'])
            # Affine interpolation retains fixed small-graph/aggregation overhead;
            # at the measured maximum use its actual cost, not a small-case extrapolation.
            small_work,max_work=work(model,2,64),work(model,5,256)
            slope=max(0.,maximum['epoch_seconds_max']-small['epoch_seconds_max'])/(max_work-small_work)
            epoch=max(small['epoch_seconds_max']*min(1.,actual/small_work),
                      small['epoch_seconds_max']+slope*(actual-small_work))
            memory=math.ceil((max(small['peak_process_rss_bytes'],maximum['peak_process_rss_bytes'])*3+2*1024**3)/1024**2/1024)*1024
            predictions[host]=dict(projected_epoch_seconds=epoch,safe_500_epoch_seconds=epoch*500*1.75+120,
                                   memory_mb=max(8192,memory),projection='native small/maximum affine cost; exact maximum anchored to its observed max;1.75 wall margin+120s')
        conditions[condition['id']]=predictions
    capacity=dict(status='PASS',source_commit=protocol['source_commit'],checked_unix=time.time(),common_lr=.001,
                  verified_hosts=sorted(reports),c4_host=c4,measured_clock_mhz=clock,
                  scheduler_cpu_factor_limitation='LSF advertises250 and native interfaces4023MHz for all; reduced-frequency cause not independently established.',
                  scheduling_rule='Researcher clarified: large jobs C4; remaining jobs no host binding; no frequency classification or smart host allocation.',
                  lr_rule='Common0.001 predeclared from historical instability;0.01 comparisons diagnostic only, no per-architecture/test tuning.',
                  evidence=evidence,conditions=conditions,accuracy_used_for_feasibility=False)
    jobs=research_jobs(protocol,capacity)
    from gnn_pruning.experiments.flickr_dense import sha
    capacity['capacity_tool_sha256']=sha(Path(__file__))
    write(output,capacity);write(output.with_name('research-jobs.json'),jobs)
    print(json.dumps(dict(status='PASS',hosts=clock,research_jobs=len(jobs),max_memory_mb=max(j['resources']['memory_mb'] for j in jobs))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('root','protocol','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();check(a.root,load_yaml(a.protocol),a.output)

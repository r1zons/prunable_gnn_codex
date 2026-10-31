"""Single-attempt Flickr LSF submission. Scheduler owns execution after acceptance."""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import re
import shlex
import subprocess
import time

from gnn_pruning.config import load_yaml
from .flickr_dense import digest, matrix, sha, write


def preflight_trials() -> list[dict]:
    """A fixed technical LR/resource check, never part of the research matrix."""
    trials=[]
    for model in ('gcn','graphsage'):
        for lr in (.01,.001):
            trials.append(dict(condition=f'flickr-{model}-l2-h64-s101',epochs=12,lr=lr,
                               id=f'{model}-small-lr{str(lr).replace(".","p")}'))
        trials.append(dict(condition=f'flickr-{model}-l5-h256-s101',epochs=6,lr=.001,id=model+'-maximum'))
    return trials


def launcher(root: Path, source: Path, taskroot: Path, protocol: Path,
             activation_sha: str, job: dict) -> str:
    """Pin environment/source/input protocol before any node computation."""
    q=shlex.quote
    lines=['#!/bin/bash','set -euo pipefail','export LC_ALL=C LANG=C',
           'cd '+q(str(taskroot)),
           'echo '+q(activation_sha+'  activate-polus.sh')+' | sha256sum -c -',
           'source ./activate-polus.sh','export PYTHONPATH='+q(str(source/'src')),
           'export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4',
           'cd '+q(str(source)),
           'sha256sum --quiet -c '+q(str(root/'source-files.sha256')),
           'echo '+q(sha(protocol)+'  '+str(protocol))+' | sha256sum -c -',
           'echo ACTUAL_HOST=$(hostname) LSB_JOBID=$LSB_JOBID',
           'cat /proc/cpuinfo | head -30',
           'lshosts -l "$(hostname)" || true']
    trials=preflight_trials() if job['stage']=='preflight' else [dict(condition=job['condition'],id=job['condition'])]
    for trial in trials:
        output=root/('preflight/'+job['id']+'/'+trial['id'] if job['stage']=='preflight' else 'runs/'+trial['id'])
        command=['python','-u','-m','gnn_pruning.experiments.flickr_dense','--protocol',str(protocol),
                 '--inputs',str(root/'inputs'),'--output',str(output),'--condition',trial['condition']]
        if job['stage']=='preflight':
            command += ['--technical','--epochs',str(trial['epochs']),'--lr',str(trial['lr'])]
        lines.append(' '.join(map(q,command)))
    return '\n'.join(lines)+'\n'


def command(job: dict, root: Path) -> list[str]:
    """Build explicit normal-queue, single-host, exclusive resource requests."""
    r=job['resources']
    if not 1 <= r['wall_minutes'] <= 180 or r['slots']!=4 or r['memory_mb']<8192:
        raise ValueError('Unsafe LSF resource request.')
    hosts=['-m',r['host_preferences']] if r.get('host_preferences') else []
    return ['bsub','-q','normal','-n','4','-W',str(r['wall_minutes']),'-x',*hosts,'-R','span[hosts=1]',
            '-R',f'rusage[mem={r["memory_mb"]}]','-M',str(r['memory_mb']),
            '-J','flickr-'+job['id'],'-o',str(root/'logs'/(job['id']+'.%J.out')),
            '-e',str(root/'logs'/(job['id']+'.%J.err'))]


def research_jobs(protocol: dict, capacity: dict) -> list[dict]:
    """Keep all32 conditions. Eligibility depends on measured cost, never accuracy."""
    if capacity['status']!='PASS' or capacity['common_lr']!=protocol['training']['lr']:
        raise ValueError('Native resource/common-LR acceptance gate missing.')
    if time.time()-capacity['checked_unix'] > 3600 or capacity['source_commit']!=protocol['source_commit']:
        raise ValueError('Stale or different-source capacity evidence.')
    hosts=capacity['verified_hosts'];c4=capacity['c4_host']
    if len(set(hosts))!=3 or c4 not in hosts:
        raise ValueError('Three verified compute hosts, including C4, required.')
    jobs=[]
    for condition in matrix(protocol):
        estimates=capacity['conditions'][condition['id']]
        expensive=condition['width']==256 or (condition['layers']>=4 and condition['width']>=128)
        eligible=[c4] if expensive else hosts
        if any(estimates[h]['safe_500_epoch_seconds'] > 175*60 for h in eligible):
            raise ValueError('No host safely fits unchanged500-epoch budget: '+condition['id'])
        # Researcher's explicit clarification: large on C4, all others unbound.
        preferences=c4 if expensive else None
        memory=max(estimates[h]['memory_mb'] for h in eligible)
        jobs.append(dict(id=condition['id'],stage='research',condition=condition['id'],dependencies=[],
                         resources=dict(slots=4,memory_mb=memory,wall_minutes=180,
                                        host_preferences=preferences,eligible_hosts=eligible)))
    return jobs


def submit(jobs: list[dict],root: Path,source: Path,taskroot: Path,protocol_path: Path,
           activation_sha: str,runner=subprocess.run) -> list[dict]:
    """Persist INTENT then one bsub attempt; never retry ambiguous submissions."""
    protocol=load_yaml(protocol_path)
    if not re.fullmatch('[0-9a-f]{40}',protocol['source_commit']):
        raise ValueError('Exact source commit required.')
    if sha(taskroot/'activate-polus.sh')!=activation_sha:
        raise ValueError('Verified activation script changed.')
    for name,expected in json.loads((root/'source-files.json').read_text()).items():
        if sha(source/name)!=expected:raise ValueError('Executable snapshot changed: '+name)
    if any((root/'submissions'/job['id']).exists() for job in jobs):
        raise FileExistsError('Prior submission claim exists. Inspect receipts; never silently retry.')
    (root/'logs').mkdir(exist_ok=True);receipts=[]
    for job in jobs:
        claim=root/'submissions'/job['id'];claim.mkdir(parents=True,exist_ok=False)
        script=launcher(root,source,taskroot,protocol_path,activation_sha,job)
        (claim/'launcher.sh').write_text(script)
        intent=dict(status='INTENT',job=job,source_commit=protocol['source_commit'],
                    protocol_sha256=sha(protocol_path),launcher_sha256=sha(claim/'launcher.sh'),
                    submission_tool_sha256=sha(Path(__file__)),created_unix=time.time())
        write(claim/'intent.json',intent)
        try:
            response=runner(command(job,root),input=script,text=True,capture_output=True,timeout=30)
            stdout,stderr,code=response.stdout,response.stderr,response.returncode
        except (OSError,subprocess.TimeoutExpired) as exc:
            stdout,stderr,code='',str(exc),None
        write(claim/'response.json',dict(stdout=stdout,stderr=stderr,returncode=code))
        match=re.fullmatch(r'\s*Job <(\d+)> is submitted to queue <normal>\.\s*',stdout)
        accepted=code==0 and match is not None and int(match[1]) not in [r['job_id'] for r in receipts]
        receipt=dict(intent,status='ACCEPTED' if accepted else 'UNRESOLVED',accepted=accepted,
                     job_id=int(match[1]) if accepted else None)
        write(claim/'receipt.json',receipt);receipts.append(receipt)
        print(json.dumps(dict(identity=job['id'],status=receipt['status'],job_id=receipt['job_id'])),flush=True)
        if not accepted:break
    return receipts


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','source','taskroot','protocol','jobs'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--activation-sha',required=True);args=parser.parse_args()
    receipts=submit(json.loads(args.jobs.read_text()),args.root,args.source,args.taskroot,args.protocol,args.activation_sha)
    write(args.root/('acceptance-'+args.jobs.stem+'.json'),receipts)
    if not all(r['accepted'] for r in receipts):raise SystemExit(2)

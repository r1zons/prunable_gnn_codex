"""Bounded thread-scaling jobs; isolated from the unsubmitted56 research tasks."""
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
import time

from .flickr_dense import sha, write

MODULE='gnn_pruning.experiments.flickr_thread_scaling'


def jobs(config):
    """Only two tiny topology jobs and one serial eight-process C4 benchmark."""
    if set(config['thread_counts'])!={4,8,12,20} or config['targets']!=[.3,.5]:
        raise ValueError('Authorized resource matrix differs.')
    return [dict(id='topology-'+name,stage='topology',host=config['hosts'][name],slots=1,
                 wall_minutes=3,memory_mb=8192) for name in ['c2','c3']] + [
            dict(id='c4-thread-scaling',stage='benchmark',host=config['host'],slots=20,
                 wall_minutes=175,memory_mb=32768)]


def command(job,root):
    """Normal/exclusive/single-host allocation; explicitly request all20 slots."""
    if (job['stage']=='benchmark' and (job['slots']!=20 or job['wall_minutes']!=175)
        or job['stage']=='topology' and (job['slots']!=1 or job['wall_minutes']!=3)
        or job['stage'] not in ['benchmark','topology']):
        raise ValueError('Unexpected/unbounded resource request.')
    return ['bsub','-q','normal','-n',str(job['slots']),'-W',str(job['wall_minutes']),
            '-x','-m',job['host'],'-R','span[hosts=1]','-R',f'rusage[mem={job["memory_mb"]}]',
            '-M',str(job['memory_mb']),'-J','flickr-cpu-'+job['id'],
            '-o',str(root/'logs'/(job['id']+'.%J.out')),'-e',str(root/'logs'/(job['id']+'.%J.err'))]


def launcher(job,root,source,taskroot,config_path,protocol_path,inputs,activation):
    """Hash-pinned existing CPU environment; no candidate CUDA prefix or training."""
    q=shlex.quote
    args=['python','-u','-m',MODULE,'topology' if job['stage']=='topology' else 'orchestrate',
          '--root',str(root/('topology-'+job['host'].split('.')[0]) if job['stage']=='topology' else root)]
    if job['stage']=='benchmark':
        args+=['--inputs',str(inputs),'--protocol',str(protocol_path),'--config',str(config_path)]
    return '\n'.join(['#!/bin/bash','set -euo pipefail','export LC_ALL=C LANG=C',
        'cd '+q(str(taskroot)),
        'echo '+q(activation+'  activate-polus.sh')+' | sha256sum -c -',
        'source ./activate-polus.sh',
        'export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1',
        'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1',
        'export PYTHONPATH='+q(str(source/'src')),
        'cd '+q(str(source)),
        'sha256sum --quiet -c '+q(str(root/'source-files.sha256')),
        'echo '+q(sha(config_path)+'  '+str(config_path))+' | sha256sum -c -',
        'echo '+q(sha(protocol_path)+'  '+str(protocol_path))+' | sha256sum -c -',
        ' '.join(map(q,args))])+'\n'


def submit(root,source,taskroot,config_path,protocol_path,inputs,activation,runner=subprocess.run):
    """Prior INTENT/ACCEPTED protocol; one attempt, no retry on ambiguous response."""
    config=json.loads(config_path.read_text());plan=jobs(config)
    if not re.fullmatch('[0-9a-f]{40}',config['benchmark_source_commit']):
        raise ValueError('Pinned executable source required.')
    if sha(taskroot/'activate-polus.sh')!=activation:
        raise ValueError('Working CPU activation identity changed.')
    for name,expected in json.loads((root/'source-files.json').read_text()).items():
        if sha(source/name)!=expected:raise ValueError('Source pin differs: '+name)
    if any((root/'submissions'/j['id']).exists() for j in plan):
        raise FileExistsError('Prior intent exists: inspect receipts, never silently retry.')
    (root/'logs').mkdir(exist_ok=True);receipts=[]
    for job in plan:
        claim=root/'submissions'/job['id'];claim.mkdir(parents=True,exist_ok=False)
        script=launcher(job,root,source,taskroot,config_path,protocol_path,inputs,activation)
        (claim/'launcher.sh').write_text(script)
        intent=dict(status='INTENT',job=job,source_commit=config['benchmark_source_commit'],
                    config_sha256=sha(config_path),protocol_sha256=sha(protocol_path),
                    launcher_sha256=sha(claim/'launcher.sh'),created_unix=time.time())
        write(claim/'intent.json',intent)
        try:
            p=runner(command(job,root),input=script,text=True,capture_output=True,timeout=30)
            stdout,stderr,code=p.stdout,p.stderr,p.returncode
        except (OSError,subprocess.TimeoutExpired) as exc:
            stdout,stderr,code='',str(exc),None
        write(claim/'response.json',dict(stdout=stdout,stderr=stderr,returncode=code))
        match=re.fullmatch(r'\s*Job <(\d+)> is submitted to queue <normal>\.\s*',stdout)
        accepted=code==0 and match is not None and int(match[1]) not in [r['job_id'] for r in receipts]
        receipt=dict(intent,status='ACCEPTED' if accepted else 'UNRESOLVED',accepted=accepted,
                     job_id=int(match[1]) if accepted else None)
        write(claim/'receipt.json',receipt);receipts.append(receipt)
        print(json.dumps(dict(id=job['id'],status=receipt['status'],job_id=receipt['job_id'])),flush=True)
        if not accepted:break
    write(root/'submission-receipts.json',receipts)
    return receipts


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['root','source','taskroot','config','protocol','inputs']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--activation-sha',required=True);a=parser.parse_args()
    receipts=submit(a.root,a.source,a.taskroot,a.config,a.protocol,a.inputs,a.activation_sha)
    if len(receipts)!=3 or not all(r['accepted'] for r in receipts):raise SystemExit(2)

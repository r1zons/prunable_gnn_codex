"""Single-attempt LSF submission with immutable INTENT/ACCEPTED evidence."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
import time

from gnn_pruning.config import load_yaml
from .complete_campaign import dependency_plan, matrix, sha, write

C4 = 'polus-c4-ib.bmc.hpc.cs.msu.ru'


def resources(job):
    """Conservative absolute wall limits, four slots, 8 GiB memory per job."""
    stage = job['stage']
    wall = {'train':10,'validate':5,'gate':5}.get(stage)
    if stage == 'prune':
        wall = {'cora':20,'citeseer':30,'pubmed':45}[job['condition'].split('-')[0]]
    return dict(wall_minutes=wall, slots=4, memory_limit_mb=8192, exclusive=True, queue='normal', host=C4)


def validate_source(source, manifest):
    """Require every executable snapshot file to match its pinned content hash."""
    for name, digest in manifest.items():
        path = source/name
        if not path.is_file() or sha(path) != digest:
            raise ValueError('Pinned executable source file mismatch: '+name)


def launcher(job, root, source, protocol_path, taskroot, activation_sha):
    """Generate a fail-closed node launcher; historical environments stay intact."""
    q = shlex.quote
    invoke = ['python','-m','gnn_pruning.experiments.complete_campaign',
              '--root',str(root),'--protocol',str(protocol_path),'--stage',job['stage']]
    if 'condition' in job:
        invoke += ['--condition',job['condition']]
    return '\n'.join(['#!/bin/bash','set -euo pipefail',
        'export LC_ALL=C LANG=C', 'cd '+q(str(taskroot)),
        'echo '+q(activation_sha+'  activate-polus.sh')+' | sha256sum -c -',
        'source ./activate-polus.sh',
        'export PYTHONPATH='+q(str(source/'src')),
        'export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4',
        'cd '+q(str(source)),
        'sha256sum --quiet -c '+q(str(root/'source-files.sha256')),
        'echo '+q(sha(protocol_path)+'  '+str(protocol_path))+' | sha256sum -c -',
        ' '.join(map(q,invoke)), ''])


def submit(root, source, protocol_path, taskroot, activation_sha, preflight, dry_run=False, runner=subprocess.run):
    """Submit a topological DAG once; an ambiguous response stops all submission."""
    root,source,protocol_path,taskroot = map(lambda p:Path(p).resolve(),(root,source,protocol_path,taskroot))
    protocol = load_yaml(protocol_path)
    jobs = dependency_plan(protocol)
    if not re.fullmatch('[0-9a-f]{40}',protocol.get('source_commit','')):
        raise ValueError('Exact executable source commit required.')
    if sha(taskroot/'activate-polus.sh') != activation_sha:
        raise ValueError('Native activation changed.')
    if preflight.get('lsf_limit_unit') != 'MB':
        raise ValueError('Verified LSF memory limit unit MB required.')
    if preflight['status'] != 'PASS' or not preflight['native_inputs_verified'] or not preflight['capacity_safe']:
        raise ValueError('Native environment, input and capacity gate required.')
    if preflight['source_commit'] != protocol['source_commit'] or time.time()-preflight['checked_unix'] > 1800:
        raise ValueError('Preflight is stale or refers to another source.')
    validate_source(source, json.loads((root/'source-files.json').read_text()))
    if any((root/'submissions'/job['id']).exists() for job in jobs):
        raise RuntimeError('Existing submission claim: inspect receipts, never silently retry.')
    total = sum(resources(job)['wall_minutes'] for job in jobs)
    if total > preflight['maximum_aggregate_wall_minutes']:
        raise ValueError('Aggregate reserved wall budget exceeded.')
    result = dict(source_commit=protocol['source_commit'], protocol_sha256=sha(protocol_path),
        conditions=len(matrix(protocol)), pruning_tasks=252, planned_jobs=len(jobs),
        aggregate_wall_minutes=total, allocated_slot_hours=total*4/60,
        queue_waiting='excluded from active wall reservation',jobs=[])
    if dry_run:
        return dict(result,dry_run=True,dag=jobs)
    (root/'logs').mkdir(exist_ok=True)
    accepted = {}
    for job in jobs:
        claim = root/'submissions'/job['id']; claim.mkdir(parents=True, exist_ok=False)
        allocation = resources(job)
        dependencies = [accepted[key] for key in job['dependencies']]
        script = launcher(job,root,source,protocol_path,taskroot,activation_sha)
        (claim/'launcher.sh').write_text(script)
        intent = dict(job=job,resources=allocation,dependency_job_ids=dependencies,
            source_commit=protocol['source_commit'],protocol_sha256=sha(protocol_path),
            launcher_sha256=sha(claim/'launcher.sh'),created_unix=time.time(),status='INTENT')
        write(claim/'intent.json',intent)
        command = ['bsub','-q','normal','-n','4','-W',str(allocation['wall_minutes']),
            '-m',C4,'-x','-R','span[hosts=1]','-R','rusage[mem=8192]',
            '-M','8192','-J','complete-'+job['id'],
            '-o',str(root/'logs'/(job['id']+'.%J.out')),
            '-e',str(root/'logs'/(job['id']+'.%J.err'))]
        if dependencies:
            command += ['-w',' && '.join(f'done({key})' for key in dependencies)]
        try:
            response = runner(command,input=script,text=True,capture_output=True,timeout=30)
            stdout,stderr,code = response.stdout,response.stderr,response.returncode
        except (OSError,subprocess.TimeoutExpired) as exc:
            stdout,stderr,code = '',str(exc),None
        write(claim/'response.json',dict(stdout=stdout,stderr=stderr,returncode=code))
        match = re.fullmatch(r'\s*Job <(\d+)> is submitted to queue <normal>\.\s*',stdout)
        ok = code == 0 and match is not None and int(match[1]) not in accepted.values()
        receipt = dict(intent,status='ACCEPTED' if ok else 'UNRESOLVED',job_id=int(match[1]) if ok else None,
            accepted=ok,returncode=code)
        write(claim/'receipt.json',receipt)
        result['jobs'].append(receipt)
        write(root/'submissions'/f'ledger-{len(result["jobs"]):03}.json',result)
        print(json.dumps(dict(node=job['id'],status=receipt['status'],job_id=receipt['job_id'])),flush=True)
        if not ok:
            break
        accepted[job['id']] = receipt['job_id']
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','source','protocol','taskroot','preflight'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--activation-sha',required=True);p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();r=submit(a.root,a.source,a.protocol,a.taskroot,a.activation_sha,json.loads(a.preflight.read_text()),a.dry_run)
    write(a.root/('submission-dry-run.json' if a.dry_run else 'submission-result.json'),r)
    print(json.dumps(dict(accepted=sum(j['accepted'] for j in r['jobs']),planned=r['planned_jobs'],dry_run=a.dry_run)),flush=True)

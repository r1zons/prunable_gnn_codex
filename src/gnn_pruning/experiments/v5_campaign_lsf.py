"""Single-attempt V5 LSF submission; native measurement gates all production jobs."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import re
import shlex
import subprocess
import time

from .flickr_dense import digest, sha, write
from .v5_campaign import matrix

C4 = 'polus-c4-ib.bmc.hpc.cs.msu.ru'


def technical_jobs():
    """Same host, separate fresh workers, fixed one/four-thread resource comparison."""
    return [dict(id=f'preflight-t{n}', stage='preflight', threads=n,
                 resources=dict(slots=4, memory_mb=8192, wall_minutes=90, host=C4)) for n in (1,4)]


def production_jobs(p, gate):
    """Require fresh complete native capacity evidence; fourteen conditions per job."""
    rows = matrix(p)
    if (gate.get('status') != 'PASS' or gate.get('source_commit') != p['source_commit']
        or gate.get('protocol_sha256') != digest(p) or not 0 <= time.time()-gate['checked_unix'] <= 3600
        or set(gate.get('conditions', {})) != {c['id'] for c in rows}
        or gate.get('dense_verified') != 18 or gate.get('technical_verified') != 168):
        raise ValueError('Fresh complete source-matched native gate required.')
    result = []
    for key in sorted(p['dense']):
        selected = [c for c in rows if c['dense_id'] == key]
        estimates = [gate['conditions'][c['id']] for c in selected]
        threads = gate['threads']
        seconds = sum(e['safe_worker_seconds'] for e in estimates) + 120
        memory = max(e['memory_mb'] for e in estimates)
        if (len(selected) != 14 or threads not in (1,4) or not math.isfinite(seconds)
            or not 0 < seconds <= 170*60 or not 0 < memory <= 8192):
            raise ValueError('Unsafe native resource forecast: '+key)
        result.append(dict(id='research-'+key, dense_id=key, stage='research', threads=threads,
            conditions=[c['id'] for c in selected], safe_seconds=seconds,
            resources=dict(slots=4, memory_mb=8192, wall_minutes=min(180, max(10,math.ceil(seconds/60)+5)), host=C4)))
    if len(result) != 18:
        raise ValueError('Eighteen verified frozen Dense inputs required.')
    return result


def command(job, root):
    """Normal queue enforces exclusive allocation; no hidden concurrent workers."""
    r = job['resources']
    if (r['slots'] != 4 or not 1 <= r['wall_minutes'] <= 180 or r['memory_mb'] != 8192
        or r['host'] != C4 or job['threads'] not in (1,4)):
        raise ValueError('Invalid bounded resource allocation.')
    return ['bsub','-q','normal','-n','4','-x','-m',r['host'],'-W',str(r['wall_minutes']),
            '-R','span[hosts=1]','-R',f'rusage[mem={r["memory_mb"]}]','-M',str(r['memory_mb']),
            '-J','v5-'+job['id'],'-o',str(root/'logs'/(job['id']+'.%J.out')),
            '-e',str(root/'logs'/(job['id']+'.%J.err'))]


def launcher(job, root, source, taskroot, activation):
    """Pin protocol, source, activation and inputs before unchanged V5 execution."""
    q = shlex.quote; n = job['threads']
    lines = ['#!/bin/bash','set -euo pipefail','export LC_ALL=C LANG=C',
             'cd '+q(str(taskroot)), 'echo '+q(activation+'  activate-polus.sh')+' | sha256sum -c -',
             'source ./activate-polus.sh',
             f'export OMP_NUM_THREADS={n} MKL_NUM_THREADS={n} OPENBLAS_NUM_THREADS={n} NUMEXPR_NUM_THREADS={n}',
             'export OMP_DYNAMIC=FALSE OMP_MAX_ACTIVE_LEVELS=1 PYTHONDONTWRITEBYTECODE=1',
             'export PYTHONPATH='+q(str(source/'src')), 'cd '+q(str(source)),
             'sha256sum --quiet -c '+q(str(root/'source-files.sha256')),
             'echo '+q(sha(root/'protocol.json')+'  '+str(root/'protocol.json'))+' | sha256sum -c -',
             'lscpu', 'bhosts -l "$(hostname)"', 'lshosts -l "$(hostname)"']
    if job['stage'] == 'preflight':
        # Verify all eighteen frozen inputs natively before representative searches.
        lines.append('python -u -c '+q("from pathlib import Path; import json, torch; from gnn_pruning.experiments.v5_campaign import frozen; from gnn_pruning.experiments.flickr_dense import write; "
                    f"r=Path({str(root)!r}); p=json.loads((r/'protocol.json').read_text()); torch.set_num_threads({n}); "
                    "[print(k, 'DENSE PASS', flush=True) for k in p['dense'] if frozen(r,p,k)]; "
                    f"write(r/'dense-native-t{n}.json',dict(status='PASS',source_commit=p['source_commit'],verified={{k:v['checkpoint_sha256'] for k,v in p['dense'].items()}}))"))
        keys = [f'{d}-{m}-s101' for d in ('cora','citeseer','pubmed') for m in ('gcn','graphsage')]
    else:
        keys = [job['dense_id']]
    for key in keys:
        args = ['python','-u','-m','gnn_pruning.experiments.v5_campaign','--root',str(root),
                '--protocol',str(root/'protocol.json'),'--dense-id',key,'--threads',str(n)]
        if job['stage'] == 'preflight': args.append('--technical')
        lines.append(' '.join(map(q,args)))
    return '\n'.join(lines)+'\n'


def submit(plan, root, source, taskroot, activation, runner=subprocess.run):
    """Persist INTENT before bsub. Ambiguity or earlier intent always stops retries."""
    p = json.loads((root/'protocol.json').read_text())
    if not re.fullmatch('[0-9a-f]{40}',p['source_commit']):
        raise ValueError('Pinned commit required.')
    if any(j['stage'] == 'research' for j in plan):
        expected = production_jobs(p, json.loads((root/'capacity.json').read_text()))
        if plan != expected: raise ValueError('No partial or changed production plan.')
    elif plan != technical_jobs():
        raise ValueError('Unknown native preflight plan.')
    if sha(taskroot/'activate-polus.sh') != activation:
        raise ValueError('Activation script changed.')
    for name, pin in json.loads((root/'source-files.json').read_text()).items():
        if sha(source/name) != pin: raise ValueError('Source snapshot changed.')
    if any((root/'submissions'/j['id']).exists() for j in plan):
        raise FileExistsError('Prior submission claim: reconcile scheduler; never retry automatically.')
    (root/'logs').mkdir(exist_ok=True); receipts=[]
    for j in plan:
        claim=root/'submissions'/j['id'];claim.mkdir(parents=True,exist_ok=False)
        script=launcher(j,root,source,taskroot,activation);(claim/'launcher.sh').write_text(script)
        intent=dict(status='INTENT',job=j,source_commit=p['source_commit'],protocol_sha256=sha(root/'protocol.json'),
                    launcher_sha256=sha(claim/'launcher.sh'),created_unix=time.time())
        write(claim/'intent.json',intent)
        try:
            response=runner(command(j,root),input=script,text=True,capture_output=True,timeout=30)
            stdout,stderr,code=response.stdout,response.stderr,response.returncode
        except (OSError,subprocess.TimeoutExpired) as exc:
            stdout,stderr,code='',str(exc),None
        write(claim/'response.json',dict(stdout=stdout,stderr=stderr,returncode=code))
        match=re.fullmatch(r'\s*Job <(\d+)> is submitted to queue <normal>\.\s*',stdout)
        accepted=code==0 and match is not None and int(match[1]) not in [r['job_id'] for r in receipts]
        receipt=dict(intent,status='ACCEPTED' if accepted else 'UNRESOLVED',
                     accepted=accepted,job_id=int(match[1]) if accepted else None)
        write(claim/'receipt.json',receipt);receipts.append(receipt)
        print(json.dumps(dict(id=j['id'],status=receipt['status'],job_id=receipt['job_id'])),flush=True)
        if not accepted:break
    return receipts


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('root','source','taskroot'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--activation',required=True);p.add_argument('--production',action='store_true');a=p.parse_args()
    protocol=json.loads((a.root/'protocol.json').read_text())
    jobs=production_jobs(protocol,json.loads((a.root/'capacity.json').read_text())) if a.production else technical_jobs()
    receipts=submit(jobs,a.root,a.source,a.taskroot,a.activation)
    write(a.root/('acceptance-research.json' if a.production else 'acceptance-preflight.json'),receipts)
    if len(receipts)!=len(jobs) or not all(r['accepted'] for r in receipts):raise SystemExit(2)

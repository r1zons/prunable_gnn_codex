"""Pinned independent Flickr pruning jobs, single-attempt acceptance protocol."""
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
import time

from .flickr_dense import digest, sha, write
from .flickr_lsf import command
from .flickr_pruning import ARCHITECTURES, matrix


def jobs(protocol, technical=False):
    """Four technical jobs or exactly56 independent research jobs; never dispatch."""
    result=[]
    rows=[dict(id='smoke-'+a,architecture=a) for a in ARCHITECTURES] if technical else matrix(protocol)
    for row in rows:
        expensive='l4-h256' in row['architecture'] or row.get('method') in ('lookahead_depth2','tabular_q')
        result.append(dict(id=row['id'],stage='smoke' if technical else 'research',
                  architecture=row['architecture'],condition=row['id'],dependencies=[],
                  resources=dict(slots=4,memory_mb=32768,wall_minutes=90 if technical else 180,
                                 host_preferences=protocol['c4_host'] if expensive else None)))
    return result


def launcher(job,root,source,taskroot,protocol,activation):
    """Only the frozen pruning module can execute; no Dense training entry point."""
    q=shlex.quote
    args=['python','-u','-m','gnn_pruning.experiments.flickr_pruning',
          '--root',str(root),'--protocol',str(protocol)]
    args+=['--smoke-architecture',job['architecture']] if job['stage']=='smoke' else ['--condition',job['condition']]
    return '\n'.join(['#!/bin/bash','set -euo pipefail','export LC_ALL=C LANG=C',
        'cd '+q(str(taskroot)),
        'echo '+q(activation+'  activate-polus.sh')+' | sha256sum -c -',
        'source ./activate-polus.sh',
        'export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1',
        'export PYTHONPATH='+q(str(source/'src')),'cd '+q(str(source)),
        'sha256sum --quiet -c '+q(str(root/'source-files.sha256')),
        'echo '+q(sha(protocol)+'  '+str(protocol))+' | sha256sum -c -',
        ' '.join(map(q,args))])+'\n'


def submit(plan,root,source,taskroot,protocol_path,activation,runner=subprocess.run):
    """INTENT precedes one bsub; unresolved responses halt without automatic retry."""
    protocol=json.loads(protocol_path.read_text())
    if not re.fullmatch('[0-9a-f]{40}',protocol['source_commit']):
        raise ValueError('Pinned source required.')
    if any(j['stage']=='research' for j in plan):
        gate=json.loads((root/'capacity.json').read_text())
        if (gate['status']!='PASS' or gate['source_commit']!=protocol['source_commit']
            or gate.get('protocol_digest')!=digest(protocol) or time.time()-gate['checked_unix']>3600):
            raise ValueError('Fresh native correctness/resource gate required.')
        if {j['id'] for j in plan}!={r['id'] for r in matrix(protocol)}:
            raise ValueError('No silent reduction of56-condition matrix.')
    if sha(taskroot/'activate-polus.sh')!=activation:
        raise ValueError('Activation changed.')
    for name,expected in json.loads((root/'source-files.json').read_text()).items():
        if sha(source/name)!=expected:raise ValueError('Source changed: '+name)
    if any((root/'submissions'/j['id']).exists() for j in plan):
        raise FileExistsError('Prior submission intent; manual reconciliation required, no retry.')
    (root/'logs').mkdir(exist_ok=True);receipts=[]
    for job in plan:
        claim=root/'submissions'/job['id'];claim.mkdir(parents=True,exist_ok=False)
        script=launcher(job,root,source,taskroot,protocol_path,activation)
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
        print(json.dumps(dict(id=job['id'],status=receipt['status'],job_id=receipt['job_id'])),flush=True)
        if not accepted:break
    return receipts


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','source','taskroot','protocol','jobs'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--activation-sha',required=True);a=p.parse_args()
    plan=json.loads(a.jobs.read_text())
    receipts=submit(plan,a.root,a.source,a.taskroot,a.protocol,a.activation_sha)
    write(a.root/('acceptance-'+a.jobs.stem+'.json'),receipts)
    if len(receipts)!=len(plan) or not all(r['accepted'] for r in receipts):raise SystemExit(2)

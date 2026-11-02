"""Metadata-only Flickr LSF submitter. Standard library only; no research imports.

Historical PASS/FAIL records are referenced verbatim. The present explicit user
instruction authorizes reuse and the narrowly specified execution-resource change;
this code never manufactures a fresh native gate or schedules auxiliary work.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import time

ARCHITECTURES = ('flickr-graphsage-l3-h128-s101','flickr-graphsage-l4-h128-s101',
                 'flickr-graphsage-l4-h256-s101','flickr-gcn-l2-h256-s101')
METHODS = ('static_random','static_magnitude','static_snip','sequential_random',
           'adaptive_greedy','lookahead_depth2','tabular_q')
CPUS = [0,80,8,88,16,96,24,104,32,112,40,120,48,128,56,136,64,144,72,152]


def sha(path):
    """Hash preparation/receipt files; never deserialize scientific tensors."""
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    """Exclusive creation prevents replacing an earlier intent or response."""
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def identities(protocol):
    """The previously prepared56 identities; no historical controller substitution."""
    if (protocol['architectures']!=list(ARCHITECTURES) or protocol['methods']!=list(METHODS)
        or protocol['targets']!=[.3,.5] or any(protocol[k]!=101 for k in ('dense_seed','split_seed','policy_seed'))
        or protocol['budget']!=dict(tolerance=.01,minimum=4)
        or protocol['q_learning']!=dict(episodes=30,alpha=.1,gamma=.9,epsilon=.2)):
        raise ValueError('Authorized frozen Flickr scientific protocol differs.')
    return [f'{arch}-{method}-t{round(target*100)}-budget-v5'
            for arch in ARCHITECTURES for target in (.3,.5) for method in METHODS]


def validate(root):
    """Check manifest metadata, resources and existing evidence references only."""
    p=read(root/'protocol.json');jobs=read(root/'jobs.json');profiles=read(root/'execution-resources.json')
    if len(jobs)!=56 or [j['id'] for j in jobs]!=identities(p) or set(profiles)!=set(identities(p)):
        raise ValueError('Complete unique independent56-job matrix required.')
    if not re.fullmatch('[0-9a-f]{40}',p['source_commit']):raise ValueError('Exact source commit required.')
    basis=read(root/'submission-basis.json')
    if (basis['authority']!='EXPLICIT_USER_AUTHORIZATION_TO_REUSE_EXISTING_VERIFIED_EVIDENCE'
        or basis['new_compute_verification_performed'] or basis['historical_records_refreshed']):
        raise ValueError('Explicit unchanged-evidence submission basis required.')
    for name,pin in basis['evidence_files'].items():
        if sha(root/name)!=pin:raise ValueError('Referenced evidence bytes changed: '+name)
    old_rows={r['condition_id']:r for r in read(root/'prior-manifest.json')}
    for j in jobs:
        old=old_rows[j['id']];r=j['resources'];profile=profiles[j['id']]
        special=j['architecture']=='flickr-graphsage-l4-h256-s101' and j['method']=='tabular_q'
        expected_slots=20 if special else int(old['requested_threads'])
        if (j['dependencies'] or r['queue']!='normal' or r['wall_minutes']!=int(old['wall_minutes'])
            or r['memory_mb']!=int(old['memory_mb']) or not r['exclusive']
            or r['slots']!=expected_slots or r['host']!=(None if old['host_constraint']=='UNBOUND' else old['host_constraint'])
            or j['dense_checkpoint_sha256']!=p['dense'][j['architecture']]['checkpoint_sha256']
            or not 0<j['forecast_minutes']<=170):
            raise ValueError('Changed/unsupported resource or input allocation: '+j['id'])
        if special:
            if (profile['threads']!=12 or profile['interop_threads']!=1 or profile['cpu_affinity']!=CPUS
                or r['host']!=p['c4_host'] or r['slots']!=20 or r['memory_mb']!=32768):
                raise ValueError('Required measured12-thread C4 profile differs.')
        elif profile!=dict(threads=4,interop_threads=None,cpu_affinity=None,
                          thread_environment=dict(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')):
            raise ValueError('Original54 four-thread execution profiles changed.')
    return p,jobs,profiles


def launcher(job,root,source,taskroot,activation,profile):
    """Only one scientific condition; no smoke, audit, controller or dependency."""
    q=shlex.quote
    lines=['#!/bin/bash','set -euo pipefail','export LC_ALL=C LANG=C',
           'cd '+q(str(taskroot)), 'echo '+q(activation+'  activate-polus.sh')+' | sha256sum -c -',
           'source ./activate-polus.sh','export PYTHONPATH='+q(str(source/'src')),
           'export PYTHONDONTWRITEBYTECODE=1']
    if profile['threads']==12:lines.append('unset GOMP_CPU_AFFINITY')
    lines+=['export '+key+'='+q(value) for key,value in profile['thread_environment'].items()]
    lines+=['cd '+q(str(source)), 'sha256sum --quiet -c '+q(str(root/'source-files.sha256'))]
    for name in ('protocol.json','execution-resources.json','jobs.json','submission-basis.json'):
        lines.append('echo '+q(sha(root/name)+'  '+str(root/name))+' | sha256sum -c -')
    args=['python','-u',str(source/'src/gnn_pruning/experiments/flickr_scientific_worker.py'),
          '--root',str(root),'--protocol',str(root/'protocol.json'),
          '--resources',str(root/'execution-resources.json'),'--condition',job['id']]
    return '\n'.join(lines+[' '.join(map(q,args))])+'\n'


def command(job,root):
    r=job['resources'];hosts=['-m',r['host']] if r['host'] else []
    return ['bsub','-q','normal','-n',str(r['slots']),'-W',str(r['wall_minutes']),'-x',*hosts,
            '-R','span[hosts=1]','-R',f'rusage[mem={r["memory_mb"]}]','-M',str(r['memory_mb']),
            '-J','flickr56-'+job['id'],'-o',str(root/'logs'/(job['id']+'.%J.out')),
            '-e',str(root/'logs'/(job['id']+'.%J.err'))]


def submit(root,source,taskroot,activation):
    p,jobs,profiles=validate(root)
    if sha(taskroot/'activate-polus.sh')!=activation:raise ValueError('CPU activation changed.')
    for name,pin in read(root/'source-files.json').items():
        if sha(source/name)!=pin:raise ValueError('Pinned executable source changed: '+name)
    if any((root/'submissions'/j['id']).exists() for j in jobs):
        raise FileExistsError('Existing intent: reconcile response, never silently retry.')
    duplicate=read(root/'duplicate-check.json')
    if duplicate['matching_claims'] or not duplicate['prior_namespaces_inspected']:
        raise ValueError('Duplicate prevention metadata gate not satisfied.')
    # An exclusive batch claim prevents restart even if no first bsub was reached.
    write(root/'batch-intent.json',dict(source_commit=p['source_commit'],conditions=identities(p),created_unix=time.time()))
    (root/'logs').mkdir(exist_ok=True);receipts=[]
    for job in jobs:
        claim=root/'submissions'/job['id'];claim.mkdir(parents=True,exist_ok=False)
        script=launcher(job,root,source,taskroot,activation,profiles[job['id']]);(claim/'launcher.sh').write_text(script)
        intent=dict(status='INTENT',condition_id=job['id'],job=job,source_commit=p['source_commit'],
            protocol_sha256=sha(root/'protocol.json'),execution_resources_sha256=sha(root/'execution-resources.json'),
            launcher_sha256=sha(claim/'launcher.sh'),submission_tool_sha256=sha(Path(__file__)),created_unix=time.time())
        write(claim/'intent.json',intent)
        try:
            response=subprocess.run(command(job,root),input=script,text=True,capture_output=True,timeout=30)
            stdout,stderr,code=response.stdout,response.stderr,response.returncode
        except (OSError,subprocess.TimeoutExpired) as exc:stdout,stderr,code='',str(exc),None
        write(claim/'response.json',dict(stdout=stdout,stderr=stderr,returncode=code))
        match=re.fullmatch(r'\s*Job <(\d+)> is submitted to queue <normal>\.\s*',stdout)
        accepted=code==0 and match is not None and int(match[1]) not in [r['job_id'] for r in receipts]
        # A response hinting at acceptance but not proving it is always unresolved.
        rejected=(not accepted and code not in (None,0) and 'Job not submitted' in stdout+'\n'+stderr
                  and not re.search(r'Job\s*<\d+>',stdout+'\n'+stderr))
        status='ACCEPTED' if accepted else 'REJECTED' if rejected else 'UNRESOLVED'
        receipt=dict(intent,status=status,job_id=int(match[1]) if accepted else None,
                     original_response_sha256=sha(claim/'response.json'))
        write(claim/'receipt.json',receipt);receipts.append(receipt)
        print(json.dumps(dict(condition_id=job['id'],status=status,job_id=receipt['job_id'])),flush=True)
        if not accepted:break
    by_id={r['condition_id']:r for r in receipts}
    result=[dict(condition_id=j['id'],status=by_id[j['id']]['status'] if j['id'] in by_id else 'NOT SUBMITTED',
                 job_id=by_id[j['id']]['job_id'] if j['id'] in by_id else None) for j in jobs]
    write(root/'submission-results.json',dict(source_commit=p['source_commit'],conditions=result,
        counts={s:sum(r['status']==s for r in result) for s in ('ACCEPTED','REJECTED','UNRESOLVED','NOT SUBMITTED')}))
    return all(r['status']=='ACCEPTED' for r in result)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','source','taskroot'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--activation',required=True);a=parser.parse_args()
    raise SystemExit(0 if submit(a.root,a.source,a.taskroot,a.activation) else 2)

"""Bounded dependent V5 release/verification jobs, with immutable single-attempt receipts.

This is orchestration only. The release job audits completed native evidence
before submitting any production condition. Its failure never releases research.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
import time

from gnn_pruning.experiments.flickr_dense import digest, sha, write
from gnn_pruning.experiments.v5_campaign import matrix
from gnn_pruning.experiments.v5_campaign_lsf import C4, command, launcher, production_jobs, submit


def assert_scheduler_done(ids, output, runner=subprocess.run):
    """Require retained LSF success, never infer completion from acceptance."""
    for identity in ids:
        response=runner(['bhist','-l',str(identity)],text=True,capture_output=True,timeout=30)
        write(output/f'lsf-{identity}.json',dict(stdout=response.stdout,stderr=response.stderr,returncode=response.returncode))
        if response.returncode!=0 or 'Done successfully' not in response.stdout or 'Exited with exit code' in response.stdout:
            raise RuntimeError('Required predecessor not confirmed DONE: '+str(identity))


def control_script(root, source, taskroot, activation, control, analysis, hashes, stage):
    """Pinned scripts use the immutable scientific source snapshot and separate reporting files."""
    q=shlex.quote
    prelude=['#!/bin/bash','set -euo pipefail','export LC_ALL=C LANG=C',
        'cd '+q(str(taskroot)), 'echo '+q(activation+'  activate-polus.sh')+' | sha256sum -c -',
        'source ./activate-polus.sh','export PYTHONPATH='+q(str(source/'src')),
        'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg',
        'export MPLCONFIGDIR='+q(str(root/'analysis-mpl-cache')),'cd '+q(str(source)),
        'sha256sum --quiet -c '+q(str(root/'source-files.sha256')),
        'echo '+q(sha(root/'protocol.json')+'  '+str(root/'protocol.json'))+' | sha256sum -c -',
        'echo '+q(hashes['control']+'  '+str(control))+' | sha256sum -c -',
        'echo '+q(hashes['analysis']+'  '+str(analysis))+' | sha256sum -c -']
    args=['python','-u',str(control),'--root',str(root),'--source',str(source),'--taskroot',str(taskroot),
          '--activation',activation,'--analysis',str(analysis),'--'+stage]
    return '\n'.join(prelude+[' '.join(map(q,args))])+'\n'


def submit_control(root,source,taskroot,activation,control,analysis,stage,dependencies,runner=subprocess.run):
    """One persisted intent then one acceptance response; no automatic retries."""
    if stage not in ('release','verify') or not dependencies or len(set(dependencies))!=len(dependencies):
        raise ValueError('Explicit predecessor IDs required.')
    claim=root/'submissions'/('control-'+stage);claim.mkdir(parents=True,exist_ok=False)
    hashes=dict(control=sha(control),analysis=sha(analysis))
    script=control_script(root,source,taskroot,activation,control,analysis,hashes,stage)
    (claim/'launcher.sh').write_text(script)
    job=dict(id='control-'+stage,threads=1,stage='control',
             resources=dict(slots=4,memory_mb=8192,wall_minutes=90,host=C4))
    dependency=' && '.join(f'ended({int(j)})' for j in dependencies)
    args=command(job,root)+['-w',dependency]
    intent=dict(status='INTENT',job=job,dependencies=dependencies,source_commit=json.loads((root/'protocol.json').read_text())['source_commit'],
                controller_sha256=hashes['control'],analysis_sha256=hashes['analysis'],
                launcher_sha256=sha(claim/'launcher.sh'),created_unix=time.time())
    write(claim/'intent.json',intent)
    try:
        response=runner(args,input=script,text=True,capture_output=True,timeout=30)
        stdout,stderr,code=response.stdout,response.stderr,response.returncode
    except (OSError,subprocess.TimeoutExpired) as exc:stdout,stderr,code='',str(exc),None
    write(claim/'response.json',dict(stdout=stdout,stderr=stderr,returncode=code))
    match=re.fullmatch(r'\s*Job <(\d+)> is submitted to queue <normal>\.\s*',stdout)
    accepted=code==0 and match is not None and int(match[1]) not in dependencies
    receipt=dict(intent,status='ACCEPTED' if accepted else 'UNRESOLVED',accepted=accepted,job_id=int(match[1]) if accepted else None)
    write(claim/'receipt.json',receipt)
    if not accepted:raise RuntimeError('Ambiguous/rejected control submission; reconcile before retry.')
    return receipt


def release(root,source,taskroot,activation,analysis):
    """All mandatory gates precede actual bsub of the complete252-condition matrix."""
    from gnn_pruning.experiments.v5_campaign_audit import capacity
    claim=root/'release';claim.mkdir(exist_ok=False)
    try:
        predecessors=json.loads((root/'acceptance-preflight.json').read_text())
        assert len(predecessors)==2 and all(r['accepted'] for r in predecessors)
        assert_scheduler_done([r['job_id'] for r in predecessors],claim)
        tests=json.loads((root/'tests.json').read_text())
        p=json.loads((root/'protocol.json').read_text())
        assert tests['status']=='PASS' and tests['failed']==0 and tests['passed']==846 and tests['source_commit']==p['source_commit']
        assert tests['controller_sha256']==sha(Path(__file__)) and tests['analysis_sha256']==sha(analysis)
        for args in (['bqueues','-l','normal'],['bhosts','-l',C4],['busers'],['bjobs','-a']):
            r=subprocess.run(args,text=True,capture_output=True,timeout=30)
            write(claim/('scheduler-'+args[0]+'.json'),dict(args=args,stdout=r.stdout,stderr=r.stderr,returncode=r.returncode))
            if r.returncode and args[0]!='bjobs':raise RuntimeError('Current scheduler query failed.')
            if args[0]=='bqueues' and not ('Open:Active' in r.stdout and '180.0 min' in r.stdout):
                raise RuntimeError('Normal queue policy changed; no production submission.')
        gate=capacity(root,p)  # strict168-profile independent audit, including frozen native18
        jobs=production_jobs(p,gate)
        write(root/'capacity.json',gate);write(root/'research-jobs.json',jobs)
        resources={identity:j for j in jobs for identity in j['conditions']}
        write(root/'resource-manifest.json',[dict(**c,resources=resources[c['id']]['resources'],
            threads=gate['threads'],forecast=gate['conditions'][c['id']],lsf_group=resources[c['id']]['id']) for c in matrix(p)])
        receipts=submit(jobs,root,source,taskroot,activation)
        write(root/'acceptance-research.json',receipts)
        assert len(receipts)==18 and all(r['accepted'] for r in receipts)
        receipt=submit_control(root,source,taskroot,activation,Path(__file__),analysis,'verify',[r['job_id'] for r in receipts])
        write(root/'acceptance-verify.json',receipt)
        write(claim/'completed.json',dict(status='SUBMITTED',conditions=252,job_ids=[r['job_id'] for r in receipts],verification_job=receipt['job_id']))
    except BaseException as exc:
        write(claim/'blocked.json',dict(status='BLOCKED',error=repr(exc),production_acceptance_exists=(root/'acceptance-research.json').exists()))
        raise


def verify(root,output,analysis):
    """After scheduler success: independent full audit, then validation-only figures."""
    from gnn_pruning.experiments.v5_campaign_audit import audit_one
    claim=root/'verification';claim.mkdir(exist_ok=False)
    try:
        receipts=json.loads((root/'acceptance-research.json').read_text())
        assert len(receipts)==18 and all(r['accepted'] for r in receipts)
        assert_scheduler_done([r['job_id'] for r in receipts],claim)
        p=json.loads((root/'protocol.json').read_text())
        rows=[audit_one(root,p,c,'runs') for c in matrix(p)]
        write(claim/'AUDIT.json',dict(status='PASS',expected=252,verified=len(rows),runs=rows,
                                     auditor_sha256=sha(Path(audit_one.__code__.co_filename))))
        # Load the separately pinned analysis module as a member of the immutable package.
        import importlib.util
        spec=importlib.util.spec_from_file_location('gnn_pruning.experiments.v5_campaign_analysis',analysis)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.analysis(root,claim/'AUDIT.json',output)
        write(claim/'completed.json',dict(status='VERIFIED',scientific_conditions=252,analysis_sha256=sha(analysis)))
    except BaseException as exc:
        write(claim/'blocked.json',dict(status='BLOCKED',error=repr(exc)));raise


if __name__=='__main__':
    # Executed directly from a separately hashed reporting file; import package
    # helpers through the original immutable scientific snapshot.
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','source','taskroot','analysis'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--activation',required=True)
    modes=p.add_mutually_exclusive_group(required=True)
    for key in ('submit-release','release','verify'):modes.add_argument('--'+key,action='store_true')
    a=p.parse_args()
    if a.submit_release:
        predecessors=json.loads((a.root/'acceptance-preflight.json').read_text())
        result=submit_control(a.root,a.source,a.taskroot,a.activation,Path(__file__),a.analysis,'release',[r['job_id'] for r in predecessors])
        write(a.root/'acceptance-release.json',result);print(json.dumps(result))
    elif a.release:release(a.root,a.source,a.taskroot,a.activation,a.analysis)
    else:verify(a.root,a.root/'analysis',a.analysis)

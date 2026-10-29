"""V3 INTENT/ACCEPTED protocol, with a strict aggregate requested-wall ledger."""
import argparse, hashlib, json, os, re, shutil, subprocess, time
from pathlib import Path
from gnn_pruning.experiments.v4_campaign import validate_manifest


def submit(root,source,config_root,taskroot,activation_sha,stage,decision=None,dry_run=False):
    """Validate every condition first; ambiguous accepts are retained and never retried."""
    root,source,config_root=map(lambda p:Path(p).resolve(),(root,source,config_root))
    launcher=source/'research/experiments/polus_v4_diagnostic_20261029/scripts/v4-cpu.lsf'
    text=launcher.read_text()
    for directive in ('#BSUB -q normal','#BSUB -n 4','#BSUB -W 01:10','#BSUB -R "span[hosts=1]"','#BSUB -m polus-c4-ib.bmc.hpc.cs.msu.ru','#BSUB -x'):
        if directive not in text:raise ValueError('Validated LSF resource directive missing.')
    if hashlib.sha256((Path(taskroot)/'activate-polus.sh').read_bytes()).hexdigest()!=activation_sha:raise ValueError('Activation changed.')
    names=('all',) if stage=='capacity' else ('gcn-t30','gcn-t50','graphsage-t30','graphsage-t50')
    wall=15 if stage=='capacity' else 70
    if stage=='campaign':
        if decision is None or decision['status']!='PASS' or not decision['primary_fits']:raise ValueError('Capacity gate required.')
    plans=[]
    for name in names:
        manifest=config_root/f'manifest-{name}.json';m=validate_manifest(manifest)
        claim=root/'submissions'/(stage+'-'+name)
        if claim.exists():raise RuntimeError('Existing claim; do not duplicate accepted or ambiguous jobs.')
        plans.append((name,manifest,m,claim))
    previous=list((root/'submissions').glob('*/intent.json'))
    reserved=sum(json.loads(p.read_text())['wall_minutes'] for p in previous)
    if reserved+wall*len(plans)>300:raise ValueError('Strict aggregate active LSF wall budget exceeded.')
    if stage=='campaign' and sum(len(m['tasks']) for _,_,m,_ in plans)!=decision['campaign_tasks']:raise ValueError('Decision/matrix mismatch.')
    result=dict(stage=stage,jobs=[],reserved_wall_minutes_before=reserved,requested_wall_minutes=wall*len(plans),
                hard_aggregate_wall_minutes=reserved+wall*len(plans),queue_waiting='Excluded from active-job budget; inspect scheduler timestamps later.')
    if dry_run:return dict(result,dry_run=True,conditions=list(names))
    (root/'logs/lsf').mkdir(parents=True,exist_ok=True)
    for name,manifest,m,claim in plans:
        claim.mkdir(parents=True)
        intent=dict(stage=stage,condition=name,wall_minutes=wall,source_commit=m['source_commit'],
                    manifest=str(manifest),manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),created_unix=time.time())
        (claim/'intent.json').write_text(json.dumps(intent,indent=2));(claim/'status').write_text('INTENT\n')
        shutil.copy2(launcher,claim/'v4-cpu.lsf');(claim/'launcher.sha256').write_text(hashlib.sha256((claim/'v4-cpu.lsf').read_bytes()).hexdigest()+'  v4-cpu.lsf\n')
        env=dict(os.environ,V4_TASK_ROOT=str(taskroot),V4_ACTIVATION_SHA=activation_sha,V4_SOURCE=str(source),
                 V4_MANIFEST=str(manifest),V4_SUBMISSION=str(claim),V4_PREFLIGHT='1' if stage=='capacity' else '0',V4_INPUTS=m['input_manifest'])
        command=['bsub','-W','00:15' if stage=='capacity' else '01:10','-J','v4-'+stage+'-'+name]
        response=subprocess.run(command,input=(claim/'v4-cpu.lsf').read_text(),text=True,capture_output=True,cwd=root,env=env)
        (claim/'response.txt').write_text(response.stdout+response.stderr)
        match=re.search(r'Job <(\d+)> is submitted',response.stdout)
        record=dict(intent,accepted=response.returncode==0 and bool(match),bsub_response=response.stdout+response.stderr,
                    task_count=len(m['tasks']),run_ids=[json.loads(Path(t['config']).read_text())['run_id'] for t in m['tasks']],
                    receipt_path=str(claim/'receipt.json'))
        if record['accepted']:
            record['job_id']=int(match[1]);record['stdout']=str(root/'logs/lsf'/f'v4-cpu.{match[1]}.out');record['stderr']=str(root/'logs/lsf'/f'v4-cpu.{match[1]}.err')
        (claim/'status').write_text('ACCEPTED\n' if record['accepted'] else 'UNRESOLVED\n')
        with (claim/'receipt.json').open('x') as stream:json.dump(record,stream,indent=2)
        result['jobs'].append(record)
        # Unique per-accept ledger snapshots preserve every acknowledgement.
        with (root/'submissions'/f'{stage}-receipt-{len(result["jobs"]):02}.json').open('x') as stream:json.dump(result,stream,indent=2)
        if not record['accepted']:break
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--source',required=True);p.add_argument('--config-root',required=True)
    p.add_argument('--taskroot',required=True);p.add_argument('--activation-sha',required=True);p.add_argument('--stage',choices=('capacity','campaign'),required=True)
    p.add_argument('--decision');p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    print(json.dumps(submit(a.root,a.source,a.config_root,a.taskroot,a.activation_sha,a.stage,
          json.loads(Path(a.decision).read_text()) if a.decision else None,a.dry_run),indent=2))

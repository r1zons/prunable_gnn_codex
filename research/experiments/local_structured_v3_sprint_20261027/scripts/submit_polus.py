"""Adapted INTENT/ACCEPTED LSF submission protocol; never polls or retries jobs."""
from __future__ import annotations
import argparse,hashlib,json,os,re,shutil,subprocess
from pathlib import Path


def submit(root, source, config_root, dry_run=False):
    root,source,config_root=Path(root).resolve(),Path(source).resolve(),Path(config_root).resolve()
    launcher=source/'research/experiments/local_structured_v3_sprint_20261027/scripts/v3-cpu.lsf'
    text=launcher.read_text()
    for line in ('#BSUB -q normal','#BSUB -n 4','#BSUB -W 03:00','#BSUB -R "span[hosts=1]"','#BSUB -m polus-c4-ib.bmc.hpc.cs.msu.ru','#BSUB -x'):
        if line not in text:raise ValueError('Required validated LSF resource directive missing.')
    plans=[]
    for condition in ('gcn-t30','gcn-t50','graphsage-t30','graphsage-t50'):
        manifest=config_root/f'manifest-{condition}.json'
        subprocess.run(['python','-m','gnn_pruning.experiments.v3_sprint','validate','--manifest',str(manifest)],check=True,cwd=source)
        claim=root/'submissions'/condition
        if claim.exists():raise RuntimeError('Existing submission claim; inspect receipt instead of duplicating.')
        plans.append((condition,manifest,claim))
    if dry_run:return dict(dry_run=True,conditions=[x[0] for x in plans],queue='normal',slots=4,node='polus-c4',wall_minutes=180)
    (root/'logs/lsf').mkdir(parents=True,exist_ok=True);accepted=[]
    for condition,manifest,claim in plans:
        claim.mkdir(parents=True);(claim/'status').write_text('INTENT\n');shutil.copy2(launcher,claim/'v3-cpu.lsf')
        (claim/'launcher.sha256').write_text(hashlib.sha256((claim/'v3-cpu.lsf').read_bytes()).hexdigest()+'  v3-cpu.lsf\n')
        env=dict(os.environ,V3_SOURCE=str(source),V3_MANIFEST=str(manifest),V3_SUBMISSION=str(claim))
        result=subprocess.run(['bsub','-J','v3-cora-'+condition],input=(claim/'v3-cpu.lsf').read_text(),text=True,capture_output=True,cwd=root,env=env)
        (claim/'response.txt').write_text(result.stdout+result.stderr)
        match=re.search(r'Job <(\d+)> is submitted',result.stdout)
        record=dict(condition=condition,accepted=result.returncode==0 and bool(match),manifest=str(manifest),response=str(claim/'response.txt'))
        if record['accepted']:
            record['job_id']=int(match[1]);(claim/'job_id').write_text(str(match[1])+'\n');(claim/'status').write_text('ACCEPTED\n')
        else:(claim/'status').write_text('UNRESOLVED\n')
        accepted.append(record);(root/'remote_launch_receipt.json').write_text(json.dumps(dict(jobs=accepted),indent=2))
        if not record['accepted']:break
    return dict(jobs=accepted)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--source',required=True)
    p.add_argument('--config-root',required=True);p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    print(json.dumps(submit(a.root,a.source,a.config_root,a.dry_run),indent=2))

"""Later collection:56 acceptances and archived scheduler DONE, no experiments."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
from datetime import datetime,timezone


def collect(login,remote,output):
    output.mkdir(parents=True,exist_ok=False);q=shlex.quote
    def ssh(command):return subprocess.run(['ssh','-o','BatchMode=yes',login,command],capture_output=True,text=True,check=True).stdout
    accepted=json.loads(ssh('cat '+q(remote+'/acceptance-research-jobs.json')))
    if len(accepted)!=56 or not all(r['accepted'] for r in accepted):raise ValueError('All56 accepted identities required.')
    scheduler={}
    for receipt in accepted:
        job=int(receipt['job_id']);history=ssh(f'export LC_ALL=C LANG=C; bhist -l {job}')
        (output/f'bhist-{job}.txt').write_text(history)
        if 'Done successfully.' not in history or 'Exited with exit code' in history:raise ValueError('Job not verified DONE: '+str(job))
        scheduler[receipt['job']['condition']]=dict(job_id=job,status='DONE',history_sha256=hashlib.sha256(history.encode()).hexdigest())
    (output/'scheduler.json').write_text(json.dumps(scheduler,indent=2))
    name='collection-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.tar.gz'
    checksum=ssh('set -e; cd '+q(remote)+'; test ! -e '+q(name)+'; tar -czf '+q(name)+
        ' inputs dense runs smoke protocol.json source-files.json source-files.sha256 submissions logs acceptance-*.json capacity.json *-jobs.json feasibility.json source-*.tar.gz; sha256sum '+q(name)).split()[0]
    subprocess.run(['scp',login+':'+remote+'/'+name,str(output/'results.tar.gz')],check=True)
    h=hashlib.sha256()
    with (output/'results.tar.gz').open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    if h.hexdigest()!=checksum:raise ValueError('Archive transfer digest mismatch.')
    raw=output/'raw';raw.mkdir()
    with tarfile.open(output/'results.tar.gz') as tar:
        for m in tar.getmembers():
            if m.issym() or m.islnk() or m.name.startswith('/') or '..' in Path(m.name).parts:raise ValueError('Unsafe archive member.')
        tar.extractall(raw,filter='data')
    (output/'collection.json').write_text(json.dumps(dict(archive_sha256=checksum,completed_jobs=56),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--login',required=True);p.add_argument('--remote-root',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();collect(a.login,a.remote_root,a.output)

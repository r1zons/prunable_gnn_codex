"""Later collection only: require32 accepted jobs and scheduler success before archive."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import tarfile


def collect(login: str,remote: str,output: Path):
    output.mkdir(parents=True,exist_ok=False)
    q=shlex.quote
    def ssh(command):
        return subprocess.run(['ssh','-o','BatchMode=yes',login,command],text=True,capture_output=True,check=True).stdout
    accepted=json.loads(ssh('cat '+q(remote+'/acceptance-research-jobs.json')))
    if len(accepted)!=32 or not all(r['accepted'] for r in accepted):raise ValueError('32 scheduler acceptances required.')
    statuses={}
    for row in accepted:
        job=int(row['job_id']);history=ssh(f'export LC_ALL=C LANG=C; bhist -l {job}')
        (output/f'bhist-{job}.txt').write_text(history)
        if 'Done successfully.' not in history or 'Exited with exit code' in history:
            raise ValueError(f'Job{job}: DONE not independently established; preserve collected history.')
        statuses[row['job']['condition']]=dict(job_id=job,status='DONE',history_sha256=hashlib.sha256(history.encode()).hexdigest())
    (output/'scheduler.json').write_text(json.dumps(statuses,indent=2))
    archive=remote+'/collection-results.tar.gz'
    # A new archive is created once, never overwriting source/raw run files.
    checksum=ssh('set -e; test ! -e '+q(archive)+'; tar -czf '+q(archive)+' -C '+q(remote)+
                 ' inputs runs protocol.json source-files.json source-files.sha256 submissions logs acceptance-research-jobs.json capacity.json research-jobs.json; sha256sum '+q(archive)).split()[0]
    subprocess.run(['scp',login+':'+archive,str(output/'results.tar.gz')],check=True)
    actual=hashlib.sha256((output/'results.tar.gz').read_bytes()).hexdigest()
    if actual!=checksum:raise ValueError('Transferred archive checksum differs.')
    raw=output/'raw';raw.mkdir()
    with tarfile.open(output/'results.tar.gz') as tar:
        for member in tar.getmembers():
            if member.issym() or member.islnk() or member.name.startswith('/') or '..' in Path(member.name).parts:
                raise ValueError('Unsafe archive member.')
        tar.extractall(raw,filter='data')
    (output/'collection.json').write_text(json.dumps(dict(jobs=32,archive_sha256=actual,scientific_analysis_performed=False),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--login',required=True);p.add_argument('--remote-root',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();collect(a.login,a.remote_root,a.output)

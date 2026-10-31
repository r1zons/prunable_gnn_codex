"""Read-only campaign inventory verification and resumable rsync collection.

No scheduler submission, experiment runner or algorithm is imported here.
Private SSH values are supplied in a local external JSON file, never tracked.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess


def digest(path: Path | str) -> str:
    """Stream a file SHA256 without loading checkpoints into memory."""
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def safe_relative(name: str) -> PurePosixPath:
    """Reject inventory entries that escape the dedicated collection root."""
    p=PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or not p.parts:
        raise ValueError('Unsafe inventory path: '+name)
    return p


def inventory_audit(root: Path | str, inventory: list[dict], required: tuple | list = ()) -> list[dict]:
    """Report missing/corrupt/unexpected paths and byte-identical file groups."""
    root=Path(root)
    expected={}
    for item in inventory:
        safe_relative(item['path'])
        if item['path'] in expected:raise ValueError('Duplicate inventory path: '+item['path'])
        expected[item['path']]=item
    rows=[];groups={}
    for name in sorted(set(expected)|set(required)):
        safe_relative(name);p=root/name;item=expected.get(name)
        row=dict(path=name,expected_bytes=item['bytes'] if item else None,
                 expected_sha256=item['sha256'] if item else None,actual_bytes=None,actual_sha256=None)
        if not p.is_file() or p.is_symlink():
            row['status']='MISSING' if not p.exists() else 'UNSUPPORTED_SYMLINK'
        else:
            row.update(actual_bytes=p.stat().st_size,actual_sha256=digest(p))
            row['status']=('NOT_IN_REMOTE_INVENTORY' if item is None else
                'VERIFIED' if row['actual_bytes']==item['bytes'] and row['actual_sha256']==item['sha256'] else 'CORRUPT')
            groups.setdefault(row['actual_sha256'],[]).append(name)
        rows.append(row)
    for p in sorted(root.rglob('*')):
        if p.is_file() and str(p.relative_to(root)) not in expected and str(p.relative_to(root)) not in required:
            rows.append(dict(path=str(p.relative_to(root)),status='UNEXPECTED',actual_bytes=p.stat().st_size,actual_sha256=digest(p)))
    for row in rows:
        row['same_content_file_count']=len(groups.get(row.get('actual_sha256'),[]))
    return rows


def required_files(spec: dict, submission: dict) -> list[str]:
    """Enumerate required outputs from the pinned explicit matrix and receipts."""
    paths={'protocol.json','submission-result.json','source-files.json','inputs/manifest.json','pilot-gate.json'}
    for c in spec['conditions']:
        paths.update('dense/'+c['id']+'/'+name for name in ('checkpoint.pt','training.json','validated.json'))
        for run in c['runs']:
            paths.update('runs/'+run['id']+'/'+name for name in ('report.json','pruned.pt','completed.json'))
    for job in submission['jobs']:
        node=job.get('node',job.get('job',{}).get('id'));jobid=job['job_id']
        paths.update(f'logs/{node}.{jobid}.{suffix}' for suffix in ('out','err'))
        paths.update(f'submissions/{node}/{name}' for name in ('intent.json','receipt.json','response.json','launcher.sh'))
    return sorted(paths)


def collect(connection: dict, remote_relative: str, destination: Path | str, log_path: Path | str) -> int:
    """Rsync reads remote files; --partial/--checksum make interruption resumable."""
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    ssh=['ssh','-4','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',
         '-o','ConnectTimeout=12','-o','ServerAliveInterval=15','-l',connection['ssh_user']]
    remote=connection['taskroot'].rstrip('/')+'/'+remote_relative.strip('/')+'/'
    command=['/usr/bin/rsync','-a','--partial','--checksum','--stats','-e',shlex.join(ssh),
             connection['ssh_host']+':'+remote,str(destination)+'/']
    with Path(log_path).open('x') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    return result.returncode


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--connection',type=Path,required=True);p.add_argument('--destination',type=Path,required=True)
    p.add_argument('--remote-relative',default='results/polus-complete-pruning-20261031');p.add_argument('--log',type=Path,required=True)
    a=p.parse_args();raise SystemExit(collect(json.loads(a.connection.read_text()),a.remote_relative,a.destination,a.log))

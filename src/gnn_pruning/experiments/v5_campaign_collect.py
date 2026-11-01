"""Read-only, hash-verified V5 collection into a new external provenance group."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shlex
import subprocess

from .complete_collection import inventory_audit, safe_relative


def collect(access, destination, stage='verified', runner=subprocess.run):
    """Require actual successful LSF completion; never collect live mutable outputs."""
    if stage not in ('preflight','verified'):raise ValueError('Unknown collection stage.')
    destination.mkdir(parents=True,exist_ok=False)
    ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',access['login']]
    # System Python2-compatible metadata query; no research imports on login node.
    code="""import os,sys,json,hashlib,subprocess
root=ROOT
stage=STAGE
accept=json.load(open(os.path.join(root,'acceptance-preflight.json' if stage=='preflight' else 'acceptance-verify.json')))
ids=[r['job_id'] for r in accept] if stage=='preflight' else [accept['job_id']]
for identity in ids:
    text=subprocess.check_output(['bhist','-l',str(identity)]).decode('utf-8')
    if 'Done successfully' not in text or 'Exited with exit code' in text:raise RuntimeError('Required LSF job not DONE')
if stage=='verified':
    marker=json.load(open(os.path.join(root,'verification/completed.json')))
    if marker['status']!='VERIFIED' or marker['scientific_conditions']!=252:raise RuntimeError('Independent audit incomplete')
items=[]
for directory,dirs,files in os.walk(root):
    dirs[:]=sorted(d for d in dirs if d not in ('source','analysis-mpl-cache','__pycache__'))
    for name in sorted(files):
        if name=='payload.tar.gz':continue
        path=os.path.join(directory,name)
        if os.path.islink(path):raise RuntimeError('Unexpected symlink')
        h=hashlib.sha256()
        with open(path,'rb') as stream:
            while True:
                block=stream.read(1048576)
                if not block:break
                h.update(block)
        items.append(dict(path=os.path.relpath(path,root),bytes=os.path.getsize(path),sha256=h.hexdigest()))
print(json.dumps(items,sort_keys=True))
""".replace('root=ROOT','root='+repr(access['root'])).replace('stage=STAGE','stage='+repr(stage))
    result=runner(ssh+['export LC_ALL=C LANG=C; python -c '+shlex.quote(code)],text=True,capture_output=True,timeout=120)
    (destination/'inventory-query-private.log').write_text(result.stderr)
    if result.returncode:raise RuntimeError('Remote scheduler/inventory gate failed; inspect external log.')
    inventory=json.loads(result.stdout)
    names=[]
    for row in inventory:
        safe_relative(row['path'])
        if '\n' in row['path'] or '\r' in row['path']:raise ValueError('Unsafe inventory filename.')
        names.append(row['path'])
    if len(set(names))!=len(names):raise ValueError('Duplicate remote paths.')
    (destination/'remote-inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
    listing=destination/'transfer-files.txt';listing.write_text('\n'.join(names)+'\n')
    raw=destination/'raw';raw.mkdir()
    args=['/usr/bin/rsync','-az','--checksum','--files-from='+str(listing),'-e',
          'ssh -o BatchMode=yes -o ConnectTimeout=15',access['login']+':'+access['root'].rstrip('/')+'/',str(raw)+'/']
    with (destination/'rsync-private.log').open('x') as log:
        result=runner(args,stdout=log,stderr=subprocess.STDOUT,timeout=600)
    if result.returncode:raise RuntimeError('Transfer incomplete; preserve partial group and inspect log.')
    rows=inventory_audit(raw,inventory)
    report=dict(status='PASS' if all(r['status']=='VERIFIED' for r in rows) else 'FAIL',
                files=len(rows),bytes=sum(r['actual_bytes'] or 0 for r in rows),rows=rows)
    (destination/'VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n')
    if report['status']!='PASS':raise RuntimeError('Local inventory verification failed.')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--access',type=Path,required=True);p.add_argument('--destination',type=Path,required=True)
    p.add_argument('--stage',choices=('preflight','verified'),default='verified');a=p.parse_args()
    r=collect(json.loads(a.access.read_text()),a.destination,a.stage)
    print(json.dumps(dict(status=r['status'],files=r['files'],bytes=r['bytes'])))

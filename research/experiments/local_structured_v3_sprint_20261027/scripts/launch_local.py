"""Atomic detached launch using established bounded local subprocess conventions."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from datetime import datetime,timezone


def launch(command, receipt, log, cwd, env):
    """Spawn a new session under nohup; never duplicate an existing submission claim."""
    receipt=Path(receipt);receipt.parent.mkdir(parents=True,exist_ok=True)
    with receipt.open('x') as f:json.dump(dict(status='INTENT',command=command),f)
    log=Path(log);log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('xb') as stream:
        child=subprocess.Popen(['nohup',*command],cwd=cwd,env=env,stdin=subprocess.DEVNULL,
                               stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    os.kill(child.pid,0)
    if os.getsid(child.pid)!=child.pid:raise RuntimeError('Child did not detach into its own session.')
    record=dict(status='ACCEPTED',pid=child.pid,session_id=child.pid,command=command,
                log=str(log),cwd=str(cwd),host=os.uname().nodename,
                created_at_utc=datetime.now(timezone.utc).isoformat())
    receipt.write_text(json.dumps(record,indent=2))
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--receipt',required=True)
    p.add_argument('--log',required=True);a=p.parse_args()
    # Validate completely before the OS submission; this is not a foreground experiment.
    subprocess.run([sys.executable,'-m','gnn_pruning.experiments.v3_sprint','validate','--manifest',a.manifest],check=True)
    command=[sys.executable,'-u','-m','gnn_pruning.experiments.v3_sprint','batch','--manifest',str(Path(a.manifest).resolve())]
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    print(json.dumps(launch(command,a.receipt,a.log,Path.cwd(),env),indent=2))

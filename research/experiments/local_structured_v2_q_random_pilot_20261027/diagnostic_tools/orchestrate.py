"""Bounded pilot command recorder; phases are launched only after inspecting prior costs/results."""
import json,os,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parent
assert json.loads((root/'preflight_result.json').read_text())['status']=='PASS'
assert (root/'observer-check.log').read_text().count('observer equivalence PASS')==2
mode=sys.argv[1];env=dict(os.environ,PYTHONPATH='src',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
commands=[]
if mode in ('a30','a100'):
    n=30 if mode=='a30' else 100
    if n==100:assert json.loads((root/'phase_a_gate.json').read_text())['proceed_100']
    for model in ('gcn','graphsage'):
        for target in (.3,.5):commands.append((f'{model}_target{int(target*100)}_episodes{n}_lambda2',['q','--model',model,'--target',str(target),'--episodes',str(n),'--penalty-lambda','2']))
elif mode=='random':commands=[('random_80',['random'])]
elif mode=='b':
    gate=json.loads((root/'phase_b_gate.json').read_text())
    for cell in gate['affected_conditions']:
        for penalty in (.5,5.):commands.append((f"{cell['model']}_target{int(cell['target']*100)}_episodes100_lambda{penalty:g}",['q','--model',cell['model'],'--target',str(cell['target']),'--episodes','100','--penalty-lambda',str(penalty)]))
else:raise ValueError(mode)
recordfile=root/'commands.json';records=json.loads(recordfile.read_text()) if recordfile.exists() else []
for run_id,args in commands:
    assert not any(r['run_id']==run_id for r in records)
    command=[sys.executable,str(root/'pilot_runner.py'),*args];log=root/'logs'/f'{run_id}.log';log.parent.mkdir(exist_ok=True);start=time.perf_counter()
    with log.open('x') as f:
        try:code=subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=300).returncode
        except subprocess.TimeoutExpired:code='timeout_300s'
    record=dict(run_id=run_id,phase=mode,command=command,cwd=str(Path.cwd()),environment={k:env[k] for k in ('PYTHONPATH','PYTHONDONTWRITEBYTECODE','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')},exit_code=code,elapsed_seconds=time.perf_counter()-start,log=str(log))
    records.append(record);recordfile.write_text(json.dumps(records,indent=2));print(json.dumps(record),flush=True)
    if code!=0:raise SystemExit(1)

"""Run exactly the authorized 18 missing conditions through the normal V2 CLI."""
import copy,json,os,subprocess,sys,time
from pathlib import Path
import yaml
repo=Path.cwd();out=repo.parent/'prunable_gnn_artifacts/local-structured-v2-cora-matrix-20261027'
assert json.loads((out/'preflight_result.json').read_text())['status']=='PASS'
prov=json.loads((out/'provenance.json').read_text());template=yaml.safe_load((repo/'configs/experiments/local_structured_v2/example.yaml').read_text())
methods=['random_local_v2','magnitude_local_v2','snip_pt_local_v2','grasp_pt_local_v2','adaptive_greedy_local_v2','q_learning_local_v2']
conditions=[('gcn',.5),('graphsage',.3),('graphsage',.5)]
status=out/'cli_runs.json';runs=json.loads(status.read_text()) if status.exists() else []
env=dict(os.environ,PYTHONPATH='src',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
for model,target in conditions:
    frozen=prov['selected'][model]
    for method in methods:
        run_id=f'{model}_seed101_target{int(target*100)}_{method}'
        assert not any(r['run_id']==run_id for r in runs),'Never repeat or overwrite an existing smoke run'
        cfg=copy.deepcopy(template);cfg['model']['name']=model;cfg['model']['hidden_channels']=32
        cfg['data']={'root':prov['data_root']}
        cfg['run']={'seed':101,'output_dir':str(out/'outputs'/model),'experiment_name':'diagnostic_cora_smoke_matrix_not_paper'}
        cfg['pruning'].update(method=method,target_sparsity=target)
        cfg['v2'].update(split_path=frozen['split'],checkpoint_sha256=frozen['checkpoint_sha256'],split_sha256=frozen['split_sha256'],graph_sha256=prov['graph_tensor_sha256'],penalty_lambda=2.0,max_steps=3,episodes=3,gamma=.9,alpha=.1,epsilon=.2)
        cfg['smoke_test']={'purpose':'functional diagnostic only; NOT approved research settings or paper results','source_checkpoint_group':'POLUS-CPU-101-C4-V2','execution_platform':'local CPU','graph_provenance_file':str(out/'provenance.json'),'thread_limit':2,'run_id':run_id}
        config=out/'configs'/f'{run_id}.yaml';config.parent.mkdir(exist_ok=True);assert not config.exists();config.write_text(yaml.safe_dump(cfg,sort_keys=False))
        command=[sys.executable,'-m','gnn_pruning','prune','--checkpoint',frozen['checkpoint'],'--config',str(config)]
        logpath=out/'logs'/f'{run_id}.log';logpath.parent.mkdir(exist_ok=True)
        start=time.perf_counter()
        with logpath.open('x') as log:
            try:exit_code=subprocess.run(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180).returncode
            except subprocess.TimeoutExpired:exit_code='timeout_180s'
        elapsed=time.perf_counter()-start
        record=dict(run_id=run_id,model=model,method=method,seed=101,target_sparsity=target,command=command,cwd=str(repo),environment={k:env[k] for k in ('PYTHONPATH','PYTHONDONTWRITEBYTECODE','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')},exit_code=exit_code,cli_seconds=elapsed,config=str(config),log=str(logpath))
        if exit_code==0:
            paths=[line.split(': ',1)[1] for line in logpath.read_text().splitlines() if line.startswith('[gnn_pruning] pruned checkpoint saved to: ')]
            assert len(paths)==1;record['artifact_dir']=str(Path(paths[0]).parent)
        runs.append(record);status.write_text(json.dumps(runs,indent=2));print(f'{len(runs)}/18 {run_id}: exit={exit_code}, CLI={elapsed:.3f}s',flush=True)
assert len(runs)==18

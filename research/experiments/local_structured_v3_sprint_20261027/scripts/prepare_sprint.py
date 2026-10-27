"""Prepare approved configs/manifests; performs no scientific run or submission."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from gnn_pruning.pruning.experimental_v3 import VARIANTS


def prepare(root, inputs, source_commit, platform, smoke=False, forward_only=False):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False)
    workers=2 if platform=='local' else 1
    seeds=(101,5001,5002) if platform=='local' and not smoke else (101,)
    targets=(.3,) if smoke else (.3,.5)
    variants=('fixed_probe_v3',) if forward_only else VARIANTS
    tasks=[];groups={}
    for model in ('gcn','graphsage'):
        for target in targets:
            selected=list(variants)
            if not forward_only and target==.3:selected+=['fixed_probe_v3']
            for variant in selected:
                for seed in ((101,) if variant=='fixed_probe_v3' else seeds):
                    run_id=f'{platform}-cora-{model}-t{int(target*100)}-{variant}-p{seed}'
                    cfg=dict(run_id=run_id,model=model,variant=variant,target=target,
                        training_seed=101,policy_seed=seed,episodes=2 if smoke else 100,
                        purpose='smoke' if smoke else 'ablation',timeout_seconds=900,
                        max_steps=20,penalty_lambda=2.,gamma=.9,alpha=.1,epsilon=.2,
                        minimum=4,tolerance=.01,source_commit=source_commit,platform=platform,
                        threads=1 if forward_only else (2 if platform=='local' else 4),
                        input_manifest=str(Path(inputs).resolve()),output_dir=str(root/'outputs'/run_id),
                        measure_inference=not smoke and seed==101)
                    p=root/'configs'/(run_id+'.json');p.parent.mkdir(exist_ok=True);p.write_text(json.dumps(cfg,indent=2))
                    row=dict(config=str(p),config_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                    tasks.append(row);groups.setdefault(f'{model}-t{int(target*100)}',[]).append(row)
    manifests={}
    for name,subset in [('all',tasks),*groups.items()]:
        artifact_root=root/'execution'/name
        payload=dict(source_commit=source_commit,platform=platform,workers=workers,
                     input_manifest=str(Path(inputs).resolve()),artifact_root=str(artifact_root),wall_budget_seconds=10800,tasks=subset)
        path=root/(f'manifest-{name}.json');path.write_text(json.dumps(payload,indent=2));manifests[name]=str(path)
    return dict(config_root=str(root),manifests=manifests,tasks=len(tasks),workers=workers)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--inputs',required=True)
    p.add_argument('--source-commit',required=True);p.add_argument('--platform',choices=('local','polus'),required=True)
    p.add_argument('--smoke',action='store_true');p.add_argument('--forward-only',action='store_true');a=p.parse_args()
    print(json.dumps(prepare(a.root,a.inputs,a.source_commit,a.platform,a.smoke,a.forward_only),indent=2))

"""Copy verified original masked inputs/Dense runs into a fresh artifact root."""
import argparse
import json
from pathlib import Path
import re
import shutil

from gnn_pruning.experiments.flickr_dense import load_inputs, sha, verify_run, write
from gnn_pruning.experiments.flickr_pruning import ARCHITECTURES, METHODS, frozen
from gnn_pruning.pruning.budget_v5 import Budget


def prepare(collection,output,source_commit,c4_host):
    if not re.fullmatch('[0-9a-f]{40}',source_commit):raise ValueError('Pinned commit required.')
    output.mkdir(parents=True,exist_ok=False)
    raw=collection/'raw'
    receipt=json.loads((collection/'collection.json').read_text())
    pins=json.loads(Path('research/experiments/flickr_dense_sweep_20261031/PROVENANCE.json').read_text())
    if sha(collection/'results.tar.gz')!=receipt['archive_sha256']:raise ValueError('Collection archive changed.')
    if sha(raw/'inputs/manifest.json')!=pins['input_manifest_sha256']:raise ValueError('Input pin changed.')
    load_inputs(raw/'inputs')
    shutil.copytree(raw/'inputs',output/'inputs');dense={}
    for arch in ARCHITECTURES:
        directory=raw/'runs'/arch
        cfg=json.loads((directory/'resolved_config.json').read_text())
        report=verify_run(directory,cfg['condition'],raw/'inputs')
        if report['source_commit']!=pins['source_commit']:raise ValueError('Dense source changed.')
        shutil.copytree(directory,output/'dense'/arch)
        dense[arch]=dict(condition=report['condition'],model_config=cfg['architecture'],
                         checkpoint_sha256=sha(directory/'checkpoint.pt'),tensor_sha256=report['tensor_sha256'],
                         validation=report['metrics']['val'],dense_source_commit=report['source_commit'])
    manifest=json.loads((output/'inputs/manifest.json').read_text())
    protocol=dict(campaign='flickr-structural-pruning-20261101',source_commit=source_commit,
         architectures=list(ARCHITECTURES),methods=list(METHODS),targets=[.3,.5],dense_seed=101,
         split_seed=101,policy_seed=101,budget=dict(tolerance=.01,minimum=4),
         q_learning=dict(episodes=30,epsilon=.2,alpha=.1,gamma=.9),
         evaluation_reuse='exact_model_frozen_input_per_run',technical_q_episodes=3,
         execution_revision='exact-inference-cache-r2',
         reward=dict(compression_gain=.6,repeated_dense_accuracy_drop=.4,terminal_lambda=2.),
         input_manifest_sha256=sha(output/'inputs/manifest.json'),split_sha256=manifest['split_sha256'],
         dense=dense,c4_host=c4_host,collection_archive_sha256=receipt['archive_sha256'])
    write(output/'protocol.json',protocol)
    feasibility=[]
    for arch in ARCHITECTURES:
        model,*_=frozen(output,protocol,arch)
        for target in protocol['targets']:
            budget=Budget(model,target,**protocol['budget'])
            feasibility.append(dict(architecture=arch,target=target,horizon=budget.horizon,
                        candidate_width_tuples=budget.candidate_count,goal_width_tuples=budget.goal_count,
                        band_feasible=not budget.fallback,parameters_dense=budget.p0))
    write(output/'feasibility.json',feasibility)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collection',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-commit',required=True);p.add_argument('--c4-host',required=True)
    a=p.parse_args();prepare(a.collection,a.output,a.source_commit,a.c4_host)

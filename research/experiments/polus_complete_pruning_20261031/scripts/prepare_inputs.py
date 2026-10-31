"""Read verified historical POLUS raw inputs; prepare new snapshots without fitting.

Run only inside the existing native environment, with explicit historical root
and fresh output. Original processed/raw/checkpoint files are never written.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import torch
from gnn_pruning.config import load_yaml
from gnn_pruning.data import generate_exact_ratio_split, load_dataset
from gnn_pruning.evaluation import measure_inference_time
from gnn_pruning.experiments.complete_campaign import sha, write, matrix, _metrics
from gnn_pruning.models import build_model
from gnn_pruning.pruning.workflow_v2 import tensor_hash


def prepare(taskroot, root, protocol, reference):
    root.mkdir(parents=True, exist_ok=False)
    (root/'inputs').mkdir()
    raw_manifest = taskroot/'data/raw-manifest.json'
    raw = json.loads(raw_manifest.read_text())
    for row in raw:
        if sha(taskroot/'data'/row['file']) != row['sha256']:
            raise ValueError('Historical raw provenance mismatch.')
    shutil.copyfile(raw_manifest, root/'inputs/historical-raw-manifest.json')
    inputs = {}
    for condition in matrix(protocol):
        if condition['seed'] != 101:
            continue
        dataset, kind = condition['dataset'], condition['model']
        old = f'{dataset}_{kind}_l3_h32'
        historical = taskroot/'results/pilot-cpu-101-c4-v2/jobs'/old/'seed_101'/old
        config = load_yaml(historical/'resolved_config.yaml')
        data_root = Path(config['data']['root'])
        canonical = dict(cora='Cora',citeseer='CiteSeer',pubmed='PubMed')[dataset]
        # Refuse a download or mutation of an absent historical processed cache.
        cache = data_root/'planetoid'/canonical/'processed/data.pt'
        if not cache.is_file():
            raise ValueError('Historical processed graph unavailable; no implicit reconstruction/download.')
        raw_files = [row for row in raw if row['file'].startswith('planetoid/'+canonical+'/')]
        if len(raw_files) != 8:
            raise ValueError('Incomplete historical raw manifest.')
        for row in raw_files:
            if sha(data_root/row['file']) != row['sha256']:
                raise ValueError('Per-condition raw lineage mismatch.')
        graph = load_dataset(dataset, data_root)[0]
        tensors = {key:getattr(graph,key).detach().cpu().clone() for key in ('x','edge_index','y')}
        if not torch.isfinite(tensors['x']).all() or tensors['edge_index'].min() < 0 or tensors['edge_index'].max() >= graph.num_nodes:
            raise ValueError('Invalid graph tensors.')
        if tensors['y'].dtype != torch.long or sorted(tensors['y'].unique().tolist()) != list(range(int(tensors['y'].max())+1)):
            raise ValueError('Invalid node labels.')
        digest = tensor_hash(tensors)
        if dataset in inputs:
            if inputs[dataset]['tensor_sha256'] != digest:
                raise ValueError('GCN/GraphSAGE historical graphs differ.')
        else:
            torch.save(tensors, root/'inputs'/f'{dataset}.pt')
            inputs[dataset] = dict(file=f'{dataset}.pt',sha256=sha(root/'inputs'/f'{dataset}.pt'),
                tensor_sha256=digest, processed_source_sha256=sha(cache),
                raw_manifest_sha256=sha(raw_manifest), nodes=int(graph.num_nodes),
                features=int(graph.num_features), classes=int(graph.y.max())+1, edges=int(graph.num_edges))
        split = generate_exact_ratio_split(int(graph.num_nodes),seed=101).to_dict()
        checkpoint_relative = str((historical/'dense_checkpoint.pt').relative_to(taskroot))
        if reference['checkpoints'].get(checkpoint_relative) != sha(historical/'dense_checkpoint.pt'):
            raise ValueError('Historical checkpoint differs from independently verified archived bytes.')
        cp = torch.load(historical/'dense_checkpoint.pt',map_location='cpu',weights_only=True)
        arch = dict(in_channels=int(graph.num_features),out_channels=int(graph.y.max())+1,**protocol['architecture'])
        cfgtraining = config['training']
        expected_training = dict(epochs=protocol['training']['max_epochs'],lr=protocol['training']['lr'],
            weight_decay=protocol['training']['weight_decay'],early_stopping_patience=protocol['training']['early_stopping_patience'])
        if cp['model_name'] != kind or cp['model_config'] != arch or cfgtraining != expected_training:
            raise ValueError('Historical architecture/training identity mismatch; new training required.')
        if any(config['data'][k] != value for k,value in [('train_ratio',.6),('val_ratio',.2),('test_ratio',.2)]):
            raise ValueError('Historical split protocol mismatch.')
        saved_split = load_yaml(historical/'splits.yaml')
        split_hash = hashlib.sha256(json.dumps(split,sort_keys=True).encode()).hexdigest()
        if saved_split != split or cp['run_signature']['seed'] != 101 or cp['run_signature']['split_hash'] != split_hash:
            raise ValueError('Historical split/seed mismatch.')
        if cp['run_signature']['dataset'] != dataset:
            raise ValueError('Historical dataset mismatch.')
        model = build_model(kind,**arch).eval();model.load_state_dict(cp['model_state_dict'],strict=True)
        validation = _metrics(model,graph,torch.tensor(split['val']))
        saved_metrics = json.loads((historical/'metrics_train.json').read_text())
        if saved_metrics['val'] != validation:
            raise ValueError('Historical validation metric reproduction failed.')
        # Frozen-checkpoint timing is technical preflight, not a pruning search.
        timing = measure_inference_time(model,graph,torch.device('cpu'),warmup_passes=2,timed_passes=5)
        proof = dict(status='PASS', archived_results_sha256=reference['archive_sha256'],origin_campaign='pilot-cpu-101-c4-v2',origin_source_commit='72293149d8fff186dd268789e7b8fea5848c9360',
            checkpoint_sha256=sha(historical/'dense_checkpoint.pt'),tensor_sha256=tensor_hash(model.state_dict()),
            resolved_config_sha256=sha(historical/'resolved_config.yaml'),training_metrics_sha256=sha(historical/'metrics_train.json'),
            split=split,split_sha256=split_hash,graph_tensor_sha256=digest,
            raw_manifest_sha256=sha(raw_manifest),raw_files=raw_files,processed_source_sha256=sha(cache),
            training_protocol=protocol['training'],validation=validation,technical_forward_timing=timing,
            original_training=saved_metrics['training'],provenance_limit='graph reconstructed by preserved processed tensors and verified raw lineage; original campaign had no contemporaneous graph tensor hash')
        target = root/'inputs/reuse'/condition['id'];target.mkdir(parents=True)
        shutil.copyfile(historical/'dense_checkpoint.pt',target/'checkpoint.pt')
        write(target/'verified.json',proof)
        print(json.dumps(dict(condition=condition['id'],status='PASS',forward_seconds=timing['inference_time_mean_sec'])),flush=True)
    write(root/'inputs/manifest.json',inputs)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--taskroot',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(4);prepare(a.taskroot,a.root,load_yaml(a.protocol),json.loads(a.reference.read_text()))

"""Export verified historical graph tensors and byte-identical frozen model inputs."""
from __future__ import annotations
import argparse,hashlib,json,shutil
from pathlib import Path
import torch
from gnn_pruning.data import load_dataset
from gnn_pruning.pruning.workflow_v2 import tensor_hash


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def prepare(repo,root):
    repo,root=Path(repo).resolve(),Path(root).resolve();root.mkdir(parents=True,exist_ok=False)
    reference='research/experiments/local_structured_v2_cora_matrix_20261027/provenance.json'
    provenance=json.loads((repo/reference).read_text())
    if sha(provenance['processed_cache'])!=provenance['processed_cache_sha256']:
        raise ValueError('Previously verified historical graph cache changed.')
    graph=load_dataset('cora',root=provenance['data_root'])[0]
    tensors={k:getattr(graph,k).detach().cpu().clone() for k in ('x','edge_index','y')}
    if tensor_hash(tensors)!=provenance['graph_tensor_sha256']:raise ValueError('Original graph order differs.')
    torch.save(tensors,root/'graph_tensors.pt')
    manifest=dict(training_seed=101,dataset='cora',source_provenance_ref=reference,
        source_provenance_sha256=sha(repo/reference),graph_tensor_sha256=provenance['graph_tensor_sha256'],
        graph=dict(file='graph_tensors.pt',sha256=sha(root/'graph_tensors.pt')),models={})
    for model,entry in provenance['selected'].items():
        if sha(entry['checkpoint'])!=entry['checkpoint_sha256'] or sha(entry['split'])!=entry['split_sha256']:
            raise ValueError('Frozen source inputs changed.')
        shutil.copyfile(entry['checkpoint'],root/f'{model}.pt')
        manifest['models'][model]=dict(file=f'{model}.pt',sha256=entry['checkpoint_sha256'],tensor_sha256=entry['dense_tensor_sha256'])
    gcn=provenance['selected']['gcn'];sage=provenance['selected']['graphsage']
    if gcn['split_sha256']!=sage['split_sha256']:raise ValueError('Model splits are not paired.')
    shutil.copyfile(gcn['split'],root/'splits.yaml');manifest['split']=dict(file='splits.yaml',sha256=gcn['split_sha256'])
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--root',required=True);a=p.parse_args()
    print(json.dumps(prepare(a.repo,a.root),indent=2))

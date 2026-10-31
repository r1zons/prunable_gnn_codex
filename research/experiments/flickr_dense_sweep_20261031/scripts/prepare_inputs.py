"""Freeze the existing Flickr cache, mask held-out labels and pin one shared split."""
import argparse
import json
from pathlib import Path

import torch
from scipy import sparse

from gnn_pruning.data import generate_exact_ratio_split
from gnn_pruning.experiments.flickr_dense import digest, sha, tensors_hash, write


def prepare(cache: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    path = cache/'processed/data.pt'
    # This is the researcher's existing trusted PyG cache, not a downloaded pickle.
    stored = torch.load(path,map_location='cpu',weights_only=False)[0]
    stored = stored if isinstance(stored,dict) else stored.to_dict()
    tensors = {k:stored[k].detach().cpu().clone() for k in ('x','edge_index','y')}
    if tensors['x'].shape != (89250,500) or tensors['edge_index'].shape != (2,899756):
        raise ValueError('Unexpected Flickr release dimensions.')
    if tensors['y'].unique().tolist() != list(range(7)) or not torch.isfinite(tensors['x']).all():
        raise ValueError('Invalid Flickr features/labels.')
    adj = sparse.load_npz(cache/'raw/adj_full.npz')
    if (adj != adj.T).nnz or not adj.has_canonical_format or adj.diagonal().nonzero()[0].size:
        raise ValueError('Unexpected Flickr directionality/duplicates/self-loops.')
    split = generate_exact_ratio_split(89250,seed=101).to_dict()
    original_hash = tensors_hash(tensors)
    # Held-out labels are quarantined locally, never uploaded to the training root.
    heldout = output.parent/'heldout-labels-local-only.pt'
    if heldout.exists(): raise FileExistsError(heldout)
    torch.save(dict(indices=torch.tensor(split['test']),labels=tensors['y'][split['test']].clone()),heldout)
    counts = {k:torch.bincount(tensors['y'][split[k]],minlength=7).tolist() for k in ('train','val')}
    tensors['y'][split['test']] = -1
    torch.save(tensors,output/'graph.pt');write(output/'split.json',split)
    manifest = dict(dataset='flickr',nodes=89250,features=500,classes=7,directed_entries=899756,
        undirected_pairs=449878,self_loops=0,preprocessing='raw float32 conversion; no feature normalization; no reverse-edge append',
        source_processed_sha256=sha(path),raw_files={p.name:sha(p) for p in sorted((cache/'raw').iterdir()) if p.is_file()},
        original_tensor_sha256=original_hash,training_tensor_sha256=tensors_hash(tensors),
        graph_file_sha256=sha(output/'graph.pt'),split_sha256=digest(split),split_file_sha256=sha(output/'split.json'),
        split_seed=101,split_protocol='generated_exact_ratio_60_20_20',official_splits_used=False,
        split_counts={k:len(v) for k,v in split.items()},class_counts=counts,test_labels_masked=True,
        heldout_labels_quarantined_sha256=sha(heldout),historical_graph_pin_limit='New freeze of recovered cache; historical execution graph hashes were absent.')
    write(output/'manifest.json',manifest)
    print(json.dumps(dict(status='PREPARED',manifest_sha256=sha(output/'manifest.json'),split_sha256=digest(split))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();prepare(args.cache,args.output)

"""Questions tensor/split validation, without transformations or split selection."""
from __future__ import annotations

from typing import Any
import torch


def validate_questions_graph(data: Any) -> dict:
    """Reject malformed/nonfinite data, duplicated arcs or invalid official masks.

    Shape checks are generic so small offline fixtures can exercise the real PyG
    processor. Published full-graph dimensions are checked by the readiness gate.
    Official masks remain available; the factory never chooses an official fold.
    """
    x, y, edge = data.x, data.y, data.edge_index
    if not isinstance(x, torch.Tensor) or x.ndim != 2 or not x.is_floating_point():
        raise ValueError('Questions features must be a floating [nodes,features] tensor.')
    n, features = x.shape
    if n < 1 or features < 1 or data.num_nodes != n or not torch.isfinite(x).all():
        raise ValueError('Questions feature dimensions/values are invalid.')
    if not isinstance(y, torch.Tensor) or y.dtype != torch.long or y.shape != (n,):
        raise ValueError('Questions labels must be an int64 vector with one entry per node.')
    if torch.unique(y).tolist() != [0,1]:
        raise ValueError('Questions requires exactly the binary labels 0 and 1.')
    if not isinstance(edge, torch.Tensor) or edge.dtype != torch.long or edge.ndim != 2 or edge.shape[0] != 2:
        raise ValueError('Questions edge_index must be int64 [2,edges].')
    if edge.shape[1] == 0 or int(edge.min()) < 0 or int(edge.max()) >= n:
        raise ValueError('Questions edge endpoints are outside the node range.')
    encoded = torch.sort(edge[0]*n+edge[1]).values
    if len(torch.unique_consecutive(encoded)) != len(encoded):
        raise ValueError('Questions contains duplicate directed edge entries.')
    reverse = torch.sort(edge[1]*n+edge[0]).values
    if not torch.equal(encoded, reverse):
        raise ValueError('Questions must already contain symmetric PyG edge entries.')
    masks = [getattr(data,name,None) for name in ('train_mask','val_mask','test_mask')]
    if any(not isinstance(m,torch.Tensor) or m.dtype != torch.bool or m.shape != (n,10) for m in masks):
        raise ValueError('Questions requires ten official boolean masks of shape [nodes,10].')
    if not torch.all(torch.stack(masks).sum(dim=0) == 1):
        raise ValueError('Questions official splits must be disjoint and cover every node.')
    if any(not torch.all(m.sum(dim=0) > 0) for m in masks):
        raise ValueError('Questions official splits must be nonempty.')
    return dict(dataset='questions',backend='HeterophilousGraphDataset',
        nodes=n,features=features,classes=2,class_counts=torch.bincount(y,minlength=2).tolist(),
        directed_entries=edge.shape[1],unique_directed_entries=len(encoded),
        undirected=True,self_loops=int((edge[0] == edge[1]).sum()),
        official_splits=10,official_split_counts={name:m.sum(dim=0).tolist()
            for name,m in zip(('train','val','test'),masks)},
        factory_selects_official_split=False,
        split_selection='explicit caller indices; primary study uses generated 60/20/20')

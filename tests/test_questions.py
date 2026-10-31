"""Offline Questions processor, split, forward and frozen-workflow integration."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from gnn_pruning.config import resolve_config
from gnn_pruning.data import DATASET_REGISTRY, generate_exact_ratio_split, load_dataset
from gnn_pruning.data.questions import validate_questions_graph
from gnn_pruning.experiments.questions_readiness import binary_metrics
from gnn_pruning.models import build_model
from gnn_pruning.pruning import workflow_v2
from gnn_pruning.pruning.workflow import prune_from_checkpoint
from gnn_pruning.training import workflow as training


@pytest.fixture
def tiny_questions(tmp_path):
    raw = tmp_path/'heterophilous/questions/raw'
    raw.mkdir(parents=True)
    masks = np.zeros((3,10,20),dtype=bool)
    for i in range(10):
        order = np.random.default_rng(i).permutation(20)
        masks[0,i,order[:10]] = True
        masks[1,i,order[10:15]] = True
        masks[2,i,order[15:]] = True
    edges = np.array([(i,(i+1)%20) for i in range(20)],dtype=np.int64)
    np.savez(raw/'questions.npz',node_features=np.random.default_rng(11).normal(size=(20,5)).astype('float32'),
        node_labels=np.array([0,1]*10,dtype=np.int64),edges=edges,
        train_masks=masks[0],val_masks=masks[1],test_masks=masks[2])
    return load_dataset(' Questions ',tmp_path)


def test_factory_registry_native_tensor_shapes_and_no_duplicate_edges(tiny_questions,tmp_path):
    dataset = tiny_questions
    data = dataset[0]
    assert DATASET_REGISTRY['questions'] == 'heterophilous'
    assert type(dataset).__name__ == 'HeterophilousGraphDataset'
    assert data.x.shape == (20,5) and data.y.shape == (20,)
    assert data.x.dtype == torch.float32 and data.y.dtype == torch.long
    assert data.edge_index.shape == (2,40) and data.is_undirected()
    assert len(torch.unique(data.edge_index[0]*20+data.edge_index[1])) == 40
    assert data.train_mask.shape == (20,10) and data.train_mask.dtype == torch.bool
    assert dataset.num_classes == 2 and dataset.metadata['official_splits'] == 10
    assert not dataset.metadata['factory_selects_official_split']
    assert dataset.transform is None and dataset.pre_transform is None
    # Cached second load must not re-double any edges or alter their ordering.
    again = load_dataset('questions',tmp_path)[0]
    assert torch.equal(data.edge_index,again.edge_index)


@pytest.mark.parametrize('problem',['nonfinite','features_shape','labels_shape','label_type','class_count',
                                    'edge_range','edge_type','duplicate_arc','directed','mask_shape','mask_overlap','mask_empty'])
def test_malformed_questions_rejected(tiny_questions,problem):
    data = tiny_questions[0].clone()
    if problem == 'nonfinite':data.x[0,0] = float('nan')
    elif problem == 'features_shape':data.x = data.x[:,0]
    elif problem == 'labels_shape':data.y = data.y[:-1]
    elif problem == 'label_type':data.y = data.y.float()
    elif problem == 'class_count':data.y[0] = 2
    elif problem == 'edge_range':data.edge_index[0,0] = 20
    elif problem == 'edge_type':data.edge_index = data.edge_index.float()
    elif problem == 'duplicate_arc':data.edge_index = torch.cat([data.edge_index,data.edge_index[:,:1]],1)
    elif problem == 'directed':data.edge_index = data.edge_index[:,1:]
    elif problem == 'mask_shape':data.val_mask = data.val_mask[:,:1]
    elif problem == 'mask_overlap':data.train_mask[:,0] = True
    elif problem == 'mask_empty':
        data.val_mask[:,0] |= data.train_mask[:,0]
        data.train_mask[:,0] = False
    with pytest.raises(ValueError):validate_questions_graph(data)


def test_primary_config_and_generated_splits_are_distinct_from_official(tiny_questions,tmp_path):
    path = tmp_path/'cfg.yaml'
    path.write_text('base: base/default\ndataset: questions\nmodel: gcn\n')
    cfg = resolve_config(path)
    assert cfg.data.name == 'questions'
    assert (cfg.data.train_ratio,cfg.data.val_ratio,cfg.data.test_ratio) == (.6,.2,.2)
    s = generate_exact_ratio_split(20,seed=101)
    assert s == generate_exact_ratio_split(20,seed=101) != generate_exact_ratio_split(20,seed=102)
    assert [len(s.train),len(s.val),len(s.test)] == [12,4,4]
    sets = [set(s.train),set(s.val),set(s.test)]
    assert len(set.union(*sets)) == 20 and not any(a & b for i,a in enumerate(sets) for b in sets[i+1:])
    assert not any(set(s.train) == set(tiny_questions[0].train_mask[:,i].nonzero().flatten().tolist()) for i in range(10))


@pytest.mark.parametrize('kind',['gcn','graphsage'])
def test_binary_forward_and_training_workflow_ignores_official_masks(tiny_questions,tmp_path,monkeypatch,kind):
    data = tiny_questions[0]
    model = build_model(kind,in_channels=5,hidden_channels=32,out_channels=2,num_layers=3,dropout=.5).eval()
    with torch.no_grad():logits = model(data)
    assert logits.shape == (20,2) and torch.isfinite(logits).all()
    cfg = dict(base='base/default',dataset='questions',model=dict(name=kind,num_layers=3,hidden_channels=8,dropout=.5),
        run=dict(seed=101,output_dir=str(tmp_path/'train')),training=dict(epochs=2),
        benchmark=dict(inference_warmup_passes=0,inference_timed_passes=1))
    path = tmp_path/'training.yaml';path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(training,'load_dataset',lambda *a,**k:tiny_questions)
    expected = generate_exact_ratio_split(20,101)
    fit = training.DenseTrainer.fit
    def checked(self,*a,**kw):
        assert kw['train_idx'].tolist() == expected.train
        assert kw['val_idx'].tolist() == expected.val
        return fit(self,*a,**kw)
    monkeypatch.setattr(training.DenseTrainer,'fit',checked)
    artifacts = training.train_dense(str(path),resume=False)
    payload = torch.load(artifacts.checkpoint_path,weights_only=True)
    assert payload['run_signature']['dataset'] == 'questions'
    assert payload['model_config']['out_channels'] == 2
    assert yaml.safe_load(artifacts.split_path.read_text()) == expected.to_dict()


@pytest.mark.parametrize('kind',['gcn','graphsage'])
@pytest.mark.parametrize('method',['magnitude_local_v2','adaptive_greedy_local_v2','q_learning_local_v2','lookahead_local_v4'])
def test_questions_frozen_checkpoint_structural_pruning(tiny_questions,tmp_path,monkeypatch,kind,method):
    data = tiny_questions[0]
    dense = build_model(kind,in_channels=5,hidden_channels=8,out_channels=2,num_layers=3,dropout=.5).eval()
    split = generate_exact_ratio_split(20,101).to_dict()
    split_path = tmp_path/'splits.yaml';split_path.write_text(yaml.safe_dump(split))
    architecture = dense.export_architecture_config();architecture.pop('name')
    checkpoint = tmp_path/'dense.pt'
    torch.save(dict(model_name=kind,model_config=architecture,model_state_dict=dense.state_dict(),
        run_signature=dict(dataset='questions',model_name=kind,seed=101,
                           split_hash=hashlib.sha256(json.dumps(split,sort_keys=True).encode()).hexdigest())),checkpoint)
    cfg = dict(base='base/default',dataset='questions',model=dict(name=kind,num_layers=3,hidden_channels=8,dropout=.5),
        run=dict(seed=101,output_dir=str(tmp_path/'out')),
        pruning=dict(method=method,structured=True,target_sparsity=.1),
        v2=dict(split_path=str(split_path),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            split_sha256=hashlib.sha256(split_path.read_bytes()).hexdigest(),
            graph_sha256=workflow_v2.tensor_hash({'x':data.x,'edge_index':data.edge_index,'y':data.y}),
            penalty_lambda=2.,episodes=2,max_steps=2),lookahead=dict(depth=2,gamma=.9))
    path = tmp_path/'frozen.yaml';path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(workflow_v2,'load_dataset',lambda *a,**k:tiny_questions)
    artifacts = prune_from_checkpoint(str(checkpoint),str(path))
    report = json.loads(artifacts.pruning_metrics_path.read_text())
    payload = torch.load(artifacts.pruned_checkpoint_path,weights_only=True)
    model = build_model(kind,**payload['model_config']).eval()
    model.load_state_dict(payload['model_state_dict'],strict=True)
    assert report['provenance']['dataset'] == 'questions'
    assert report['parameters_after'] == sum(p.numel() for p in model.parameters()) < report['parameters_before']
    assert report['original_kept_ids'] == payload['original_kept_ids']
    with torch.no_grad():assert torch.isfinite(model(data)).all()


def test_supplementary_auc_uses_probabilities_without_changing_legacy_metrics():
    # All argmax predictions are class0, but class1 scores rank positives perfectly.
    labels = torch.tensor([0,1,0,1])
    logits = torch.tensor([[3.,0.],[1.,0.],[4.,0.],[.5,0.]])
    report = binary_metrics(labels,logits)
    assert report['accuracy'] == .5 and report['roc_auc'] == 1.
    assert report['predicted_counts'] == [4,0]
    assert binary_metrics(torch.zeros(4,dtype=torch.long),logits)['roc_auc'] is None

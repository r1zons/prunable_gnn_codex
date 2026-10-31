"""Opt-in two-model local Questions readiness check, never a campaign launcher.

Uses the unchanged DenseTrainer/seed/split protocol. Test labels are masked before
training and pruning. No test metric, threshold selection, retuning or fine-tuning.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
import resource
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch_geometric
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from gnn_pruning.config import dump_yaml, resolve_config
from gnn_pruning.data import generate_exact_ratio_split, load_dataset, load_split_indices, save_split_indices
from gnn_pruning.evaluation.metrics import classification_metrics
from gnn_pruning.models import build_model
from gnn_pruning.pruning.adaptive_v2 import LocalV2Env
from gnn_pruning.pruning.workflow_v2 import tensor_hash
from gnn_pruning.training.trainer import DenseTrainer
from gnn_pruning.training.workflow import _build_run_signature
from gnn_pruning.utils import set_seed

REPO = Path(__file__).resolve().parents[3]


def sha(path: Path) -> str:
    """File identity, including raw compressed input bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: dict) -> None:
    """Create fresh evidence; never replace a previous result."""
    with path.open('x') as f:
        json.dump(value,f,indent=2,allow_nan=False)
        f.write('\n')


def binary_metrics(labels: torch.Tensor, logits: torch.Tensor) -> dict:
    """Fixed argmax quality plus supplementary continuous-score ROC-AUC."""
    if logits.ndim != 2 or logits.shape != (len(labels),2) or not torch.isfinite(logits).all():
        raise ValueError('Expected finite binary two-logit predictions.')
    y = labels.detach().cpu().numpy()
    if not set(np.unique(y)) <= {0,1} or not len(y):
        raise ValueError('Expected nonempty binary validation labels.')
    pred = logits.argmax(-1).detach().cpu().numpy()
    probability = logits.softmax(-1)[:,1].detach().cpu().numpy()
    result = classification_metrics(y,pred)
    result.update(roc_auc=float(roc_auc_score(y,probability)) if len(np.unique(y)) == 2 else None,
                  class_counts=np.bincount(y,minlength=2).tolist(),
                  predicted_counts=np.bincount(pred,minlength=2).tolist(),
                  confusion_matrix=confusion_matrix(y,pred,labels=[0,1]).tolist(),
                  per_class=classification_report(y,pred,labels=[0,1],output_dict=True,zero_division=0))
    return result


def prepare(root: Path) -> None:
    """Pin the native processed graph and a generated split, preserving all folds."""
    root.mkdir(parents=True,exist_ok=True)
    if (root/'readiness_inputs.json').exists() or (root/'splits.yaml').exists():
        raise FileExistsError('Readiness input group already exists; choose a fresh artifact root.')
    dataset = load_dataset('questions',root=root/'data')
    data = dataset[0]
    metadata = dataset.metadata
    if (metadata['nodes'],metadata['features'],metadata['classes'],metadata['directed_entries']) != (48921,301,2,307080):
        raise ValueError('Questions release dimensions differ from the reviewed input; inspect before running.')
    split = generate_exact_ratio_split(data.num_nodes,seed=101)
    split_path = save_split_indices(split,root)
    raw = Path(dataset.raw_paths[0])
    raw_arrays = np.load(raw)
    distribution = {k:torch.bincount(data.y[v],minlength=2).tolist() for k,v in split.to_dict().items()}
    feature_rms = data.x.square().mean(0).sqrt()
    summary = dict(**metadata,raw_file=str(raw),raw_sha256=sha(raw),raw_edges=len(raw_arrays['edges']),
        processed_file=str(dataset.processed_paths[0]),processed_sha256=sha(Path(dataset.processed_paths[0])),
        graph_tensor_sha256=tensor_hash({'x':data.x,'edge_index':data.edge_index,'y':data.y}),
        split_file=str(split_path),split_file_sha256=sha(split_path),
        split_protocol='generated_exact_ratio_60_20_20_v1',split_seed=101,
        split_counts={k:len(v) for k,v in split.to_dict().items()},class_counts_by_split=distribution,
        official_masks_used=False,
        generated_equals_any_official=any(all(set(split.to_dict()[k]) == set(getattr(data,k+'_mask')[:,i].nonzero().flatten().tolist())
                                              for k in ('train','val','test')) for i in range(10)),
        feature_summary=dict(minimum=float(data.x.min()),maximum=float(data.x.max()),
                             rms_min=float(feature_rms.min()),rms_max=float(feature_rms.max()),
                             constant_columns=int((data.x.max(0).values == data.x.min(0).values).sum())),
        tensor_bytes=sum(t.numel()*t.element_size() for t in (data.x,data.edge_index,data.y)),
        pyg_version=torch_geometric.__version__,
        pyg_processor_sha256=hashlib.sha256(inspect.getsource(type(dataset).process).encode()).hexdigest())
    if summary['generated_equals_any_official']:
        raise ValueError('Generated and official split identities unexpectedly coincide.')
    write(root/'readiness_inputs.json',summary)
    print(json.dumps(summary),flush=True)


def run_model(root: Path, kind: str) -> None:
    """Train one approved seed/architecture and make one fixed physical deletion."""
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    inputs = json.loads((root/'readiness_inputs.json').read_text())
    if sha(Path(inputs['raw_file'])) != inputs['raw_sha256'] or sha(root/'splits.yaml') != inputs['split_file_sha256']:
        raise ValueError('Pinned inputs changed.')
    dataset = load_dataset('questions',root=root/'data')
    data = dataset[0]
    original_hash = tensor_hash({'x':data.x,'edge_index':data.edge_index,'y':data.y})
    if original_hash != inputs['graph_tensor_sha256']:
        raise ValueError('Pinned graph tensors changed.')
    split = load_split_indices(root/'splits.yaml')
    expected = generate_exact_ratio_split(data.num_nodes,seed=101)
    if split != expected:
        raise ValueError('Primary generated split changed.')
    output = root/kind
    output.mkdir(exist_ok=False)
    cfg = dict(base='base/default',dataset='questions',
        model=dict(name=kind,hidden_channels=32,num_layers=3,dropout=.5),
        run=dict(seed=101,output_dir=str(output),experiment_name='questions_readiness_20261031'),
        data=dict(root=str(root/'data')),device=dict(device='cpu'))
    config_path = output/'config.yaml'
    dump_yaml(cfg,config_path)
    resolved = resolve_config(config_path)
    dump_yaml(resolved.to_dict(),output/'resolved.yaml')
    set_seed(101)
    # Data.y outside train/validation is unavailable to the trainer and smoke.
    work = data.clone()
    train, val, test = [torch.tensor(split.to_dict()[k],dtype=torch.long) for k in ('train','val','test')]
    work.y[test] = -1
    assert (work.y[test] == -1).all()
    model = build_model(kind,in_channels=data.num_features,hidden_channels=32,
                        out_channels=2,num_layers=3,dropout=.5).eval()
    trainer = DenseTrainer(model,'cpu',resolved.training.lr,resolved.training.weight_decay,
                            resolved.training.epochs,resolved.training.early_stopping_patience)
    source_names = ['src/gnn_pruning/data/factory.py','src/gnn_pruning/data/questions.py',
                    'src/gnn_pruning/experiments/questions_readiness.py','src/gnn_pruning/training/trainer.py',
                    'src/gnn_pruning/data/splits.py','src/gnn_pruning/surgery/local_v2.py']
    source = dict(base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
                  source_files={n:sha(REPO/n) for n in source_names})
    history = []
    def progress(row):
        history.append(row)
        with (output/'epochs.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
        if int(row['epoch']) % 10 == 0:
            print(json.dumps(dict(model=kind,**row)),flush=True)
    started = time.perf_counter()
    result = trainer.fit(work,train,val,progress_callback=progress)
    training_seconds = time.perf_counter()-started
    model = trainer.model.eval()
    with torch.no_grad():
        dense_validation = binary_metrics(work.y[val],model(work)[val])
    majority = int(torch.bincount(work.y[train],minlength=2).argmax())
    baseline_pred = torch.full_like(work.y[val],majority)
    majority_metrics = classification_metrics(work.y[val].numpy(),baseline_pred.numpy())
    signature = _build_run_signature(resolved,hashlib.sha256(json.dumps(split.to_dict(),sort_keys=True).encode()).hexdigest())
    arch = model.export_architecture_config();arch.pop('name')
    torch.save(dict(model_name=kind,model_config=arch,model_state_dict=model.state_dict(),
                    run_signature=signature,best_epoch=result.best_epoch,best_val_loss=result.best_val_loss),
               output/'dense_checkpoint.pt')
    dense_hash = tensor_hash(model.state_dict())
    validation_calls = []
    def evaluate(candidate):
        assert (work.y[test] == -1).all()
        started = time.perf_counter()
        with torch.no_grad():
            accuracy = (candidate(work).argmax(-1)[val] == work.y[val]).float().mean().item()
        validation_calls.append(time.perf_counter()-started)
        return accuracy
    # One fixed magnitude-based action; no adaptive-policy or target-grid search.
    smoke_started = time.perf_counter()
    env = LocalV2Env(model,evaluate,target=.03,penalty_lambda=2.,greedy=True)
    env.step((0,1))
    small_arch = env.model.export_architecture_config();small_arch.pop('name')
    torch.save(dict(model_name=kind,model_config=small_arch,model_state_dict=env.model.state_dict(),
                    original_kept_ids=env.original_kept_ids,source_checkpoint_sha256=sha(output/'dense_checkpoint.pt')),
               output/'smoke_pruned.pt')
    payload = torch.load(output/'smoke_pruned.pt',weights_only=True)
    rebuilt = build_model(kind,**payload['model_config']).eval()
    rebuilt.load_state_dict(payload['model_state_dict'],strict=True)
    with torch.no_grad():
        before_logits, after_logits = env.model(work)[val], rebuilt(work)[val]
    assert torch.equal(before_logits,after_logits)
    assert tensor_hash(model.state_dict()) == dense_hash
    after = sum(p.numel() for p in rebuilt.parameters())
    p0 = sum(p.numel() for p in model.parameters())
    assert after < p0 and rebuilt.hidden_channel_dims == [31,32]
    smoke = dict(action='prune:0:1',selection='fixed readiness smoke, not policy comparison',
        parameters_before=p0,parameters_after=after,achieved_sparsity=1-after/p0,
        hidden_widths=rebuilt.hidden_channel_dims,retained_ids=env.original_kept_ids,
        validation=binary_metrics(work.y[val],after_logits),trace=env.trace,
        strict_reload_equal=True,dense_unchanged=True,
        wall_seconds=time.perf_counter()-smoke_started,search_validation_calls=len(validation_calls),
        search_validation_forward_seconds=validation_calls)
    write(output/'smoke.json',smoke)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mib = peak/(1024**2 if platform.system() == 'Darwin' else 1024)
    report = dict(model=kind,dataset='questions',seed=101,hidden_width=32,layers=3,
        training_protocol=resolved.training.__dict__,selection='minimum validation CE loss, unchanged DenseTrainer',
        training=result.to_dict(),validation=dense_validation,majority_train_class=majority,
        majority_validation=majority_metrics,split_protocol=inputs['split_protocol'],
        split_sha256=inputs['split_file_sha256'],graph_tensor_sha256=original_hash,
        raw_sha256=inputs['raw_sha256'],checkpoint_sha256=sha(output/'dense_checkpoint.pt'),
        checkpoint_bytes=(output/'dense_checkpoint.pt').stat().st_size,
        parameter_count=p0,training_seconds=training_seconds,
        epoch_seconds=[row['elapsed_sec'] for row in history],process_peak_rss_mib=peak_mib,
        resource_scope='isolated worker process high-water RSS incl. imports/data/optimizer/smoke',
        source=source,runtime=dict(python=platform.python_version(),torch=str(torch.__version__),
                                  pyg=torch_geometric.__version__,threads=torch.get_num_threads(),device='cpu'),
        test_labels_exposed=False,test_metrics_computed=False,retuning=False,
        smoke_checkpoint_sha256=sha(output/'smoke_pruned.pt'))
    write(output/'report.json',report)
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--model',choices=['gcn','graphsage'])
    args = parser.parse_args()
    if args.model:
        run_model(args.root.expanduser().resolve(),args.model)
    else:
        prepare(args.root.expanduser().resolve())

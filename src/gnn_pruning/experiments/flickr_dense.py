"""Isolated, frozen-input Flickr Dense sweep. No pruning or test evaluation."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import resource
import signal
import statistics
import sys
import time

import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data

from gnn_pruning.config import load_yaml
from gnn_pruning.data import generate_exact_ratio_split
from gnn_pruning.evaluation.metrics import classification_metrics
from gnn_pruning.models import build_model
from gnn_pruning.training.trainer import DenseTrainer
from gnn_pruning.utils import set_seed


def sha(path: Path) -> str:
    """Hash file bytes without changing them."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def digest(value: object) -> str:
    """Canonical JSON identity, independent of mapping insertion order."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def tensors_hash(tensors: dict) -> str:
    """Pin names, shapes, dtypes and exact contiguous tensor bytes."""
    value = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        tensor = tensor.detach().cpu().contiguous()
        value.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        value.update(tensor.numpy().tobytes())
    return value.hexdigest()


def write(path: Path, value: object) -> None:
    """Create an immutable JSON receipt; an existing attempt is never replaced."""
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def matrix(protocol: dict) -> list[dict]:
    """Expand exactly the authorized 32 conditions, independently of performance."""
    expected = dict(dataset='flickr', models=['gcn', 'graphsage'], layers=[2, 3, 4, 5],
                    widths=[32, 64, 128, 256], seed=101, split_seed=101)
    if any(protocol.get(k) != v for k, v in expected.items()):
        raise ValueError('Authorized Flickr matrix differs from protocol.')
    t = protocol['training']
    if t['max_epochs'] != 500 or t['early_stopping_patience'] != 50:
        raise ValueError('Research epoch budget/patience must remain 500/50.')
    if not 0 < t['lr'] <= .01 or t['weight_decay'] != .0005 or protocol['dropout'] != .5:
        raise ValueError('Unexpected common training protocol.')
    return [dict(id=f'flickr-{model}-l{layers}-h{width}-s101', model=model, layers=layers, width=width, seed=101)
            for model in protocol['models'] for layers in protocol['layers'] for width in protocol['widths']]


def validate_split(split: dict, nodes: int) -> None:
    """Require the frozen generated 60/20/20 split, with no overlapping nodes."""
    if set(split) != {'train', 'val', 'test'}:
        raise ValueError('Split keys differ.')
    flat = [i for key in ('train', 'val', 'test') for i in split[key]]
    if any(type(i) is not int for i in flat) or sorted(flat) != list(range(nodes)):
        raise ValueError('Split must be disjoint and cover every node.')
    expected = generate_exact_ratio_split(nodes, seed=101).to_dict()
    if split != expected:
        raise ValueError('Frozen split is not generated seed-101 60/20/20.')


def load_inputs(inputs: Path) -> tuple[Data, dict, dict]:
    """Read only the masked training input; held-out labels are never loaded."""
    manifest = json.loads((inputs/'manifest.json').read_text())
    split = json.loads((inputs/'split.json').read_text())
    if sha(inputs/'graph.pt') != manifest['graph_file_sha256'] or digest(split) != manifest['split_sha256']:
        raise ValueError('Frozen graph/split bytes changed.')
    tensors = torch.load(inputs/'graph.pt', map_location='cpu', weights_only=True)
    if tensors_hash(tensors) != manifest['training_tensor_sha256']:
        raise ValueError('Graph tensor identity differs.')
    data = Data(**tensors)
    validate_split(split, data.num_nodes)
    if data.x.ndim != 2 or data.x.dtype != torch.float32 or not torch.isfinite(data.x).all():
        raise ValueError('Expected finite float32 node features.')
    if data.edge_index.dtype != torch.long or data.edge_index.shape[0] != 2:
        raise ValueError('Invalid edge index.')
    if data.edge_index.numel() and (data.edge_index.min() < 0 or data.edge_index.max() >= data.num_nodes):
        raise ValueError('Edge index outside graph.')
    if data.y.dtype != torch.long or data.y.shape != (data.num_nodes,):
        raise ValueError('Invalid node labels.')
    if not bool((data.y[split['test']] == -1).all()):
        raise ValueError('Held-out test labels must be absent from training input.')
    observed = data.y[split['train']+split['val']]
    if observed.min() < 0 or observed.max() >= manifest['classes']:
        raise ValueError('Training/validation label range differs.')
    return data, split, manifest


def architecture(condition: dict, inputs: dict, protocol: dict) -> dict:
    """Make depth/width explicit; output classes come from pinned input metadata."""
    return dict(in_channels=inputs['features'], out_channels=inputs['classes'],
                num_layers=condition['layers'], hidden_channels=condition['width'], dropout=protocol['dropout'])


def environment() -> dict:
    """Record actual execution hardware, CPU clocks and bounded software metadata."""
    cpuinfo = Path('/proc/cpuinfo')
    text = cpuinfo.read_text() if cpuinfo.exists() else ''
    clocks = [line.strip() for line in text.splitlines() if any(k in line.lower() for k in ('clock', 'cpu mhz', 'model name', 'cpu\t'))]
    frequencies = {}
    for kind in ('scaling_cur_freq', 'cpuinfo_max_freq', 'scaling_max_freq', 'scaling_min_freq'):
        values = []
        for path in Path('/sys/devices/system/cpu').glob('cpu[0-9]*/cpufreq/'+kind):
            try:
                values.append(int(path.read_text()))
            except (OSError, ValueError):
                pass
        if values:
            frequencies[kind] = dict(min=min(values), max=max(values), median=statistics.median(values), n=len(values))
    clock_values = [float(re.search(r'([0-9.]+)\s*MHz', line).group(1)) for line in clocks
                    if re.search(r'([0-9.]+)\s*MHz', line)]
    return dict(host=platform.node(), machine=platform.machine(), platform=platform.platform(), python=sys.version,
                torch=torch.__version__, pyg=torch_geometric.__version__, numpy=np.__version__,
                torch_threads=torch.get_num_threads(), cpu_count=os.cpu_count(), clocks=clocks[:24], frequencies_khz=frequencies,
                proc_cpu_clock_mhz_median=statistics.median(clock_values) if clock_values else None,
                meminfo=Path('/proc/meminfo').read_text() if Path('/proc/meminfo').exists() else None,
                lsf={k: os.environ.get(k) for k in ('LSB_JOBID', 'LSB_QUEUE', 'LSB_HOSTS', 'LSB_MCPU_HOSTS', 'LSB_DJOB_NUMPROC')},
                thread_environment={k: os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')},
                torch_build_config=torch.__config__.show())


def metrics(model, data: Data, split: dict) -> dict:
    """Evaluate training and validation only, with dropout disabled."""
    model.eval()
    with torch.no_grad():
        logits = model(data)
        if not torch.isfinite(logits).all():
            raise ValueError('Nonfinite logits.')
        result = {}
        for name in ('train', 'val'):
            indices = torch.tensor(split[name], dtype=torch.long)
            y, selected = data.y[indices], logits[indices]
            prediction = selected.argmax(-1)
            result[name] = dict(**classification_metrics(y.numpy(), prediction.numpy()),
                                loss=float(torch.nn.functional.cross_entropy(selected, y)),
                                class_counts=torch.bincount(y, minlength=logits.shape[1]).tolist(),
                                prediction_counts=torch.bincount(prediction, minlength=logits.shape[1]).tolist())
    return result


def convergence(history: list[dict], result: dict, maximum: int) -> dict:
    """Transparent flags; neither short-run quality nor flags remove conditions."""
    finite = all(math.isfinite(row[k]) for row in history for k in ('train_loss','val_loss','train_acc','val_acc'))
    losses = [r['val_loss'] for r in history]
    best = int(result['best_epoch'])+1
    return dict(finite=finite, stopped_by='early_stopping_patience' if len(history) < maximum else 'max_epochs',
                best_epoch_1based=best, total_epochs=len(history),
                best_in_last_50_epochs=best > max(0, len(history)-50),
                validation_loss_ended_above_initial=bool(losses and losses[-1] > losses[0]),
                large_early_loss_excursion=bool(losses and max(losses[:20]) > 2*max(losses[0],1e-12)),
                convergence_established=False,
                note='Patience is an operational stop; max-epoch/latest-best and instability require curve review.')


def _save_checkpoint(path: Path, payload: dict) -> None:
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def run(inputs: Path, output: Path, condition: dict, protocol: dict, technical: bool = False,
        epochs: int | None = None, lr: float | None = None) -> dict:
    """Fit one fresh Dense condition with unchanged DenseTrainer, never resume."""
    if not technical and (epochs is not None or lr is not None):
        raise ValueError('Research jobs cannot override common training settings.')
    if not technical and condition not in matrix(protocol):
        raise ValueError('Unknown research identity.')
    output.mkdir(parents=True, exist_ok=False)
    history = []
    started = time.perf_counter()
    try:
        torch.set_num_threads(protocol['threads'])
        data, split, input_manifest = load_inputs(inputs)
        if protocol['input_manifest_sha256'] != sha(inputs/'manifest.json'):
            raise ValueError('Protocol input pin differs.')
        arch = architecture(condition, input_manifest, protocol)
        training = dict(protocol['training'])
        if technical:
            if epochs is None or not 1 <= epochs <= 20 or lr not in (.01, .001):
                raise ValueError('Technical sanity runs require <=20 epochs and a declared LR.')
            training.update(max_epochs=epochs, early_stopping_patience=epochs, lr=lr)
        config = dict(condition=condition, architecture=arch, training=training, technical=technical,
                      seed=101, split_sha256=input_manifest['split_sha256'], input_manifest_sha256=sha(inputs/'manifest.json'),
                      source_commit=protocol['source_commit'], protocol_sha256=digest(protocol),
                      optimizer='Adam', loss='CrossEntropyLoss', device='cpu', test_labels_exposed=False)
        write(output/'resolved_config.json', config)
        write(output/'environment.json', environment())
        set_seed(101)
        model = build_model(condition['model'], **arch)
        if len(model.convs) != condition['layers'] or model.hidden_channel_dims != [condition['width']]*(condition['layers']-1):
            raise ValueError('Architecture not applied.')
        worker = DenseTrainer(model, device='cpu', **training)
        metadata = dict(model_name=condition['model'], model_config=arch, condition=condition,
                        training_protocol=training, split_sha256=input_manifest['split_sha256'],
                        graph_tensor_sha256=input_manifest['training_tensor_sha256'], config_sha256=digest(config),
                        source_commit=protocol['source_commit'], seed=101, test_labels_exposed=False)
        with (output/'history.jsonl').open('x') as stream:
            def record(row):
                if not all(math.isfinite(row[k]) for k in ('train_loss','val_loss','train_acc','val_acc')):
                    raise ValueError('Nonfinite training observation.')
                history.append(dict(row))
                stream.write(json.dumps(row, allow_nan=False)+'\n'); stream.flush()
                print(json.dumps(dict(condition=condition['id'], **row)), flush=True)
                if row['epoch'] == row['best_epoch']:
                    _save_checkpoint(output/'best.partial.pt', dict(metadata, model_state_dict=model.state_dict(), best_epoch_1based=int(row['epoch'])))
            fit_start = time.perf_counter()
            result = worker.fit(data, torch.tensor(split['train']), torch.tensor(split['val']), progress_callback=record)
            fit_wall = time.perf_counter()-fit_start
        selected_metrics = metrics(model, data, split)
        payload = dict(metadata, model_state_dict=model.state_dict(), tensor_sha256=tensors_hash(model.state_dict()),
                       result=result.to_dict(), best_epoch_1based=result.best_epoch+1)
        _save_checkpoint(output/'checkpoint.pt', payload)
        reloaded = torch.load(output/'checkpoint.pt', map_location='cpu', weights_only=True)
        verify_checkpoint(reloaded, condition, config, input_manifest)
        restored = build_model(condition['model'], **arch)
        restored.load_state_dict(reloaded['model_state_dict'], strict=True)
        repeated = metrics(restored, data, split)
        for subset in ('train', 'val'):
            if any(abs(repeated[subset][k]-selected_metrics[subset][k]) > 1e-7 for k in ('accuracy','macro_f1','loss')):
                raise ValueError('Checkpoint metrics do not reload.')
        epoch_times = [r['elapsed_sec'] for r in history]
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_bytes = int(peak if sys.platform == 'darwin' else peak*1024)
        report = dict(status='COMPLETED', technical=technical, condition=condition, metrics=selected_metrics,
                      result=result.to_dict(), convergence=convergence(history,result.to_dict(),training['max_epochs']),
                      parameters=sum(p.numel() for p in model.parameters()), checkpoint_bytes=(output/'checkpoint.pt').stat().st_size,
                      checkpoint_sha256=sha(output/'checkpoint.pt'), tensor_sha256=payload['tensor_sha256'],
                      config_sha256=digest(config), protocol_sha256=digest(protocol), source_commit=protocol['source_commit'],
                      split_sha256=input_manifest['split_sha256'], graph_tensor_sha256=input_manifest['training_tensor_sha256'],
                      training_wall_seconds=fit_wall, total_worker_wall_seconds=time.perf_counter()-started,
                      epoch_seconds_median=statistics.median(epoch_times), epoch_seconds_max=max(epoch_times),
                      peak_process_rss_bytes=peak_bytes, actual_host=platform.node(),
                      cpu_user_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime,
                      cpu_system_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_stime,
                      strict_reload_verified=True, test_metrics_evaluated=False)
        write(output/'environment_after.json', environment())
        write(output/'report.json', report)
        write(output/'completed.json', {p.name:sha(p) for p in output.iterdir() if p.is_file()})
        return report
    except BaseException as exc:
        write(output/'failed.json', dict(status='FAILED',error=repr(exc),completed_epochs=len(history),
              actual_host=platform.node(),elapsed_seconds=time.perf_counter()-started,
              partial_checkpoint_present=(output/'best.partial.pt').exists()))
        raise


def verify_checkpoint(checkpoint: dict, condition: dict, config: dict, inputs: dict) -> None:
    """Reject another architecture, source, split, config, input or tensor state."""
    expected = dict(condition=condition, model_name=condition['model'], model_config=config['architecture'],
                    training_protocol=config['training'], split_sha256=inputs['split_sha256'],
                    graph_tensor_sha256=inputs['training_tensor_sha256'], config_sha256=digest(config),
                    source_commit=config['source_commit'], seed=101, test_labels_exposed=False)
    if any(checkpoint.get(k) != v for k,v in expected.items()):
        raise ValueError('Checkpoint provenance/architecture mismatch; reuse forbidden.')
    if tensors_hash(checkpoint['model_state_dict']) != checkpoint['tensor_sha256']:
        raise ValueError('Checkpoint tensor hash differs.')
    if any(not torch.isfinite(v).all() for v in checkpoint['model_state_dict'].values()):
        raise ValueError('Nonfinite checkpoint.')


def verify_run(directory: Path, condition: dict, inputs: Path) -> dict:
    """Offline integrity/reload verification; never evaluates test labels."""
    for name, expected in json.loads((directory/'completed.json').read_text()).items():
        if sha(directory/name) != expected:
            raise ValueError('Output hash mismatch: '+name)
    config = json.loads((directory/'resolved_config.json').read_text())
    manifest = json.loads((inputs/'manifest.json').read_text())
    checkpoint = torch.load(directory/'checkpoint.pt', map_location='cpu', weights_only=True)
    verify_checkpoint(checkpoint,condition,config,manifest)
    model = build_model(condition['model'], **config['architecture'])
    model.load_state_dict(checkpoint['model_state_dict'],strict=True)
    report = json.loads((directory/'report.json').read_text())
    if report['technical'] or report['checkpoint_sha256'] != sha(directory/'checkpoint.pt'):
        raise ValueError('Research report identity differs.')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('inputs','output','protocol'): parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--condition',required=True)
    parser.add_argument('--technical',action='store_true')
    parser.add_argument('--epochs',type=int); parser.add_argument('--lr',type=float)
    args=parser.parse_args(); protocol=load_yaml(args.protocol)
    condition=next(c for c in matrix(protocol) if c['id']==args.condition)
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(InterruptedError('SIGTERM: preserve partial artifacts')))
    run(args.inputs,args.output,condition,protocol,args.technical,args.epochs,args.lr)

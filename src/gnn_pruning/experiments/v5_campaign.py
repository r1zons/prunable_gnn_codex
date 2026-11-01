"""Citation-dataset V5 orchestration. Frozen inputs only; no fitting entry point.

Scientific decisions are delegated unchanged to budget_v5. Each worker receives
an already masked per-split graph, never the historical unmasked input directory.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.metrics import f1_score

from gnn_pruning.models import build_model
from gnn_pruning.pruning.budget_v5 import Budget, METHODS, VERSION, run, static_plan
from gnn_pruning.pruning.static_v2 import apply_static_plan
from .complete_audit import validate_split
from .flickr_dense import digest, environment, sha, tensors_hash, write
from .flickr_pruning import verify_projection
from .frozen_validation import FrozenAccuracy

DATASETS = ('cora', 'citeseer', 'pubmed')
MODELS = ('gcn', 'graphsage')
SEEDS = (101, 202, 303)


def protocol():
    """Return the fixed authorized scientific protocol, before source/input pinning."""
    return dict(campaign='polus-v5-main-20261102', datasets=list(DATASETS), models=list(MODELS),
                dense_seeds=list(SEEDS), targets=[.3, .5], methods=list(METHODS),
                version=VERSION, architecture=dict(num_layers=3, hidden_channels=32, dropout=.5),
                training=dict(lr=.01, weight_decay=.0005, max_epochs=500, early_stopping_patience=50),
                budget=dict(tolerance=.01, minimum=4),
                q_learning=dict(episodes=30, alpha=.1, gamma=.9, epsilon=.2),
                inference=dict(warmup=5, repeats=20), technical_q_episodes=3,
                policy_seed_rule='same integer as dense training/split seed; deterministic methods ignore it')


def matrix(p):
    """Require exactly eighteen frozen inputs and 252 unique V5 conditions."""
    fixed = protocol()
    if any(p.get(k) != v for k, v in fixed.items()):
        raise ValueError('Authorized V5 protocol differs.')
    return [dict(id=f'{d}-{m}-s{s}-{method}-t{round(t*100)}-v5',
                 dense_id=f'{d}-{m}-s{s}', dataset=d, model=m, dense_seed=s,
                 split_seed=s, policy_seed=s, method=method,
                 implementation=method+'_'+VERSION, target=t)
            for d in DATASETS for m in MODELS for s in SEEDS for t in p['targets'] for method in METHODS]


def validation(model, data, split):
    """One finite forward; report validation only, with double-precision aggregation."""
    with torch.no_grad():
        logits = model.eval()(data)[split['val']]
    if not torch.isfinite(logits).all():
        raise ValueError('Nonfinite validation output.')
    y = data.y[split['val']]
    pred = logits.argmax(-1)
    return dict(accuracy=float((pred == y).double().mean()),
                macro_f1=float(f1_score(y.numpy(), pred.numpy(), average='macro', zero_division=0)),
                loss=float(torch.nn.functional.cross_entropy(logits, y)))


def prepare(raw, audit_csv, output):
    """Independently verify archived Dense identities, then isolate masked inputs.

    The original archived split is copied verbatim; it is never regenerated. Test
    labels are masked before any current validation forward. Original files are
    only read. This function neither trains nor executes a pruning controller.
    """
    output.mkdir(parents=True, exist_ok=False)
    (output/'inputs').mkdir(); (output/'dense').mkdir()
    rows = list(csv.DictReader(audit_csv.open()))
    expected_ids = {f'{d}-{m}-s{s}' for d in DATASETS for m in MODELS for s in SEEDS}
    if len(rows) != 18 or {r['condition_id'] for r in rows} != expected_ids or any(r['status'] != 'PASS' for r in rows):
        raise ValueError('Archived eighteen-Dense audit differs.')
    original_manifest = json.loads((raw/'inputs/manifest.json').read_text())
    pins = {}; graphs = {}
    for row in rows:
        key = row['condition_id']; d = row['dataset']; m = row['architecture']; seed = int(row['seed'])
        original = original_manifest[d]; path = raw/'inputs'/original['file']
        if sha(path) != original['sha256']:
            raise ValueError('Original graph bytes differ.')
        tensors = torch.load(path, map_location='cpu', weights_only=True)
        if tensors_hash(tensors) != original['tensor_sha256']:
            raise ValueError('Original graph tensors differ.')
        directory = raw/'dense'/key; cp_path = directory/'checkpoint.pt'
        if sha(cp_path) != row['checkpoint_sha256']:
            raise ValueError('Archived checkpoint SHA mismatch.')
        cp = torch.load(cp_path, map_location='cpu', weights_only=True)
        split_hash = validate_split(cp['split'], len(tensors['y']))
        cert = json.loads((directory/'validated.json').read_text())
        cfg = dict(in_channels=original['features'], out_channels=original['classes'], **protocol()['architecture'])
        if (cp['model_name'] != m or cp['model_config'] != cfg or cp['seed'] != seed
            or cp['training_protocol'] != protocol()['training'] or cp['graph_tensor_sha256'] != original['tensor_sha256']
            or split_hash != row['split_sha256'] or split_hash != cert['split_sha256']):
            raise ValueError('Dense graph/split/architecture/training provenance differs.')
        model = build_model(m, **cfg).eval(); model.load_state_dict(cp['model_state_dict'], strict=True)
        identity = tensors_hash(model.state_dict())
        if identity != row['tensor_sha256'] or identity != cp['tensor_sha256'] or identity != cert['tensor_sha256']:
            raise ValueError('Dense tensor identities differ.')
        if any(not bool(torch.isfinite(v).all()) for v in model.state_dict().values()):
            raise ValueError('Nonfinite Dense tensors.')
        tensors['y'] = tensors['y'].clone(); tensors['y'][cp['split']['test']] = -1
        data = Data(**tensors)
        actual = validation(model, data, cp['split'])
        for metric in ('accuracy', 'macro_f1'):
            if abs(actual[metric]-cert['validation'][metric]) > 1e-12:
                raise ValueError('Frozen Dense validation does not reproduce: '+key)
        graph_key = f'{d}-s{seed}'
        if graph_key not in graphs:
            graph_path = output/'inputs'/(graph_key+'.pt'); torch.save(tensors, graph_path)
            split_path = output/'inputs'/(graph_key+'-split.json'); write(split_path, cp['split'])
            graphs[graph_key] = dict(file=graph_path.name, sha256=sha(graph_path), tensor_sha256=tensors_hash(tensors),
                split_file=split_path.name, split_file_sha256=sha(split_path), split_sha256=split_hash,
                original_graph=original, masked_test_labels=True)
        elif graphs[graph_key]['tensor_sha256'] != tensors_hash(tensors):
            raise ValueError('Paired architectures do not share graph and split.')
        (output/'dense'/key).mkdir()
        shutil.copyfile(cp_path, output/'dense'/key/'checkpoint.pt')
        pins[key] = dict(dataset=d, model=m, seed=seed, graph=graph_key, checkpoint_sha256=sha(cp_path),
            tensor_sha256=identity, split_sha256=split_hash, model_config=cfg, validation=actual,
            parameters=sum(p.numel() for p in model.parameters()), origin_source_commit=cp['source_commit'],
            status='PASS', archived_validation=cert['validation'])
        print(key, 'PASS', flush=True)
    p = protocol(); p.update(dense=pins, graphs=graphs, archived_dense_audit_sha256=sha(audit_csv),
                            original_input_manifest_sha256=sha(raw/'inputs/manifest.json'))
    write(output/'prepared-protocol.json', p)
    return p


def frozen(root, p, key):
    """Strict reload of pinned original checkpoint and masked original graph/split."""
    pin = p['dense'][key]; g = p['graphs'][pin['graph']]
    cp_path = root/'dense'/key/'checkpoint.pt'; graph_path = root/'inputs'/g['file']
    split_path = root/'inputs'/g['split_file']
    if sha(cp_path) != pin['checkpoint_sha256'] or sha(graph_path) != g['sha256'] or sha(split_path) != g['split_file_sha256']:
        raise ValueError('Frozen input file changed.')
    cp = torch.load(cp_path, map_location='cpu', weights_only=True)
    tensors = torch.load(graph_path, map_location='cpu', weights_only=True)
    split = json.loads(split_path.read_text())
    if (validate_split(split, len(tensors['y'])) != pin['split_sha256'] or cp['split'] != split
        or cp['model_config'] != pin['model_config'] or cp['model_name'] != pin['model']
        or cp['seed'] != pin['seed'] or cp['training_protocol'] != p['training']
        or tensors_hash(tensors) != g['tensor_sha256']
        or cp['graph_tensor_sha256'] != g['original_graph']['tensor_sha256']):
        raise ValueError('Frozen graph/checkpoint metadata changed.')
    data = Data(**tensors)
    if not bool((data.y[split['test']] == -1).all()):
        raise ValueError('Test labels exposed.')
    if data.x.ndim != 2 or not torch.isfinite(data.x).all() or data.edge_index.shape[0] != 2:
        raise ValueError('Malformed graph.')
    model = build_model(pin['model'], **pin['model_config']).eval()
    model.load_state_dict(cp['model_state_dict'], strict=True)
    if tensors_hash(model.state_dict()) != pin['tensor_sha256']:
        raise ValueError('Frozen tensor mismatch.')
    actual = validation(model, data, split)
    if any(abs(actual[k]-pin['validation'][k]) > (5e-5 if k == 'loss' else 1e-12) for k in actual):
        raise ValueError('Dense validation reproduction failed.')
    return model, data, split


def search(dense, data, split, c, p, technical=False):
    """Unchanged V5 engine, exact cache and explicit real-forward accounting."""
    budget = Budget(dense, c['target'], **p['budget']); durations = []
    def accuracy(model):
        if not bool((data.y[split['test']] == -1).all()):
            raise ValueError('Test labels exposed.')
        start = time.perf_counter()
        with torch.no_grad():
            logits = model.eval()(data)[split['val']]
            if not torch.isfinite(logits).all():
                raise ValueError('Nonfinite search output.')
            value = float((logits.argmax(-1) == data.y[split['val']]).float().mean())
        durations.append(time.perf_counter()-start)
        return value
    callback = FrozenAccuracy(accuracy, digest([p['graphs'][p['dense'][c['dense_id']]['graph']], c['dense_id']]))
    method = c['method']; started = time.perf_counter(); cpu = time.process_time()
    if method.startswith('static_'):
        criterion = method.removeprefix('static_')
        selected = static_plan(dense, criterion, budget, c['policy_seed'], data, torch.tensor(split['train']))
        final = apply_static_plan(dense, selected); kept = selected.kept_indices
        trace = []; parent = dense
        for layer, ids in enumerate(kept):
            if len(ids) == parent.hidden_channel_dims[layer]:
                continue
            from gnn_pruning.surgery.local_v2 import prune_hidden_channels_local_v2
            before = list(parent.hidden_channel_dims)
            parent = prune_hidden_channels_local_v2(parent, layer, ids, 4)
            trace.append(dict(action=f'static_prune:{layer}:{before[layer]-len(ids)}',
                widths_before=before, widths_after=list(parent.hidden_channel_dims),
                sparsity=1-sum(v.numel() for v in parent.parameters())/budget.p0,
                intermediate_validation_recorded=False))
        details = dict(training=[], decisions=[], q_table=[], deployment=trace,
            statistics=dict(version=VERSION, complete=True, terminal_reason='closest_feasible' if budget.fallback else 'target_band',
                validation_evaluations=0, candidate_evaluations=0, static_plan=asdict(selected),
                scoring_gradient_forwards=int(criterion == 'snip'), horizon=budget.horizon))
    else:
        settings = dict(p['q_learning'])
        if technical:
            settings['episodes'] = p['technical_q_episodes']
        result = run(dense, callback, method, c['target'], seed=c['policy_seed'], budget=budget,
                     max_deployment_steps=2 if technical and method == 'tabular_q' else None, **settings)
        final, kept = result.model, result.kept_indices
        details = dict(training=result.training_trace, decisions=result.decisions,
            q_table=[dict(state=s, actions=a) for s, a in result.q_table.items()],
            deployment=result.trace, statistics=result.statistics)
    seconds = time.perf_counter()-started; cpu_seconds = time.process_time()-cpu
    stats = details['statistics']
    if callback.requests != stats['validation_evaluations'] or callback.forward_calls != len(durations):
        raise ValueError('Search evaluation accounting failed.')
    details.update(evaluation_cache=callback.events)
    stats.update(validation_callback_requests=callback.requests, actual_validation_forwards=callback.forward_calls,
                 exact_cache_hits=callback.requests-callback.forward_calls,
                 evaluation_seconds=sum(durations), validation_forward_seconds=durations)
    if not technical and not stats['complete']:
        raise ValueError('Incomplete scientific deployment.')
    return final, kept, details, seconds, cpu_seconds, budget


def execute(root, p, c, threads, group='runs', technical=False):
    """One isolated scientific/technical worker, strict final projection and reload."""
    if c not in matrix(p) or type(threads) is not int or not 1 <= threads <= 4:
        raise ValueError('Invalid condition/resource identity.')
    output = root/group/c['id']; output.mkdir(parents=True, exist_ok=False)
    try:
        torch.set_num_threads(threads); torch.set_num_interop_threads(1)
        dense, data, split = frozen(root, p, c['dense_id'])
        original_hash = tensors_hash(dense.state_dict())
        final, kept, details, seconds, cpu_seconds, budget = search(dense, data, split, c, p, technical)
        verify_projection(dense, final, kept)
        if tensors_hash(dense.state_dict()) != original_hash:
            raise ValueError('Dense weights mutated.')
        before = budget.p0; after = sum(v.numel() for v in final.parameters())
        if after != budget.count(list(map(len, kept))) or not 0 < after < before:
            raise ValueError('Physical parameter reduction failed.')
        cfg = final.export_architecture_config(); cfg.pop('name')
        cp = dict(model_name=c['model'], model_config=cfg, model_state_dict=final.state_dict(),
                  original_kept_ids=kept, condition=c, source_commit=p['source_commit'],
                  dense_checkpoint_sha256=p['dense'][c['dense_id']]['checkpoint_sha256'])
        torch.save(cp, output/'checkpoint.pt')
        loaded = torch.load(output/'checkpoint.pt', map_location='cpu', weights_only=True)
        restored = build_model(c['model'], **loaded['model_config']).eval()
        restored.load_state_dict(loaded['model_state_dict'], strict=True)
        if tensors_hash(restored.state_dict()) != tensors_hash(final.state_dict()):
            raise ValueError('Strict final reload failed.')
        warmup = p['inference']['warmup']; repeats = p['inference']['repeats']; times = []
        with torch.no_grad():
            for i in range(warmup+repeats):
                started = time.perf_counter(); restored(data); elapsed = time.perf_counter()-started
                if i >= warmup:
                    times.append(elapsed)
        achieved = 1-after/before
        env = environment(); env.update(interop_threads=torch.get_num_interop_threads(),
                                       affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None)
        write(output/'environment.json', env)
        for name, value in details.items():
            write(output/(name+'.json'), value)
        report = dict(condition=c, technical=technical, source_commit=p['source_commit'], protocol_sha256=digest(p),
            dense_checkpoint_sha256=p['dense'][c['dense_id']]['checkpoint_sha256'],
            graph_tensor_sha256=p['graphs'][p['dense'][c['dense_id']]['graph']]['tensor_sha256'],
            split_sha256=p['dense'][c['dense_id']]['split_sha256'],
            parameters_before=before, parameters_after=after, achieved_compression=achieved,
            signed_target_deviation=achieved-c['target'], band_feasible=not budget.fallback,
            target_miss=abs(achieved-c['target']) > p['budget']['tolerance']+1e-14,
            fallback=budget.fallback, final_widths=list(map(len, kept)), retained_original_ids=kept,
            dense_validation=p['dense'][c['dense_id']]['validation'], validation=validation(restored, data, split),
            statistics=details['statistics'], search_seconds=seconds, search_cpu_seconds=cpu_seconds,
            inference=dict(warmup=warmup, repeats=repeats, full_graph_forward_seconds=times,
                           median_seconds=float(np.median(times)), mean_seconds=float(np.mean(times)),
                           std_seconds=float(np.std(times, ddof=1))),
            peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system() == 'Darwin' else 1024)),
            actual_host=platform.node(), threads=threads, strict_reload=True, retained_projection=True,
            checkpoint_sha256=sha(output/'checkpoint.pt'), tensor_sha256=tensors_hash(final.state_dict()),
            test_labels_accessed=False, scientific_complete=not technical and details['statistics']['complete'])
        write(output/'report.json', report)
        write(output/'completed.json', {f.name:sha(f) for f in output.iterdir() if f.is_file()})
        return report
    except BaseException as exc:
        write(output/'failed.json', dict(error=repr(exc), condition=c)); raise


def batch(root, p, dense_id, threads, technical=False):
    """Fresh subprocess per condition; failures isolated, no automatic retries."""
    rows = [c for c in matrix(p) if c['dense_id'] == dense_id]
    if len(rows) != 14:
        raise ValueError('Unknown frozen Dense identity.')
    group = f'preflight-t{threads}' if technical else 'runs'
    write(root/(f'batch-intent-{group}-{dense_id}.json'),
          dict(dense_id=dense_id, threads=threads, technical=technical, conditions=[c['id'] for c in rows]))
    results = []
    for c in rows:
        args = [sys.executable, '-u', '-m', 'gnn_pruning.experiments.v5_campaign', '--root', str(root), '--protocol', str(root/'protocol.json'),
                '--condition', c['id'], '--threads', str(threads), '--group', group]
        if technical:
            args += ['--technical']
        started = time.perf_counter()
        response = subprocess.run(args, check=False)
        results.append(dict(id=c['id'], exit_code=response.returncode, worker_seconds=time.perf_counter()-started))
    write(root/(f'batch-{group}-{dense_id}.json'), results)
    if any(r['exit_code'] != 0 for r in results):
        raise RuntimeError('One or more isolated conditions failed; inspect immutable failures.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'protocol'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--condition'); parser.add_argument('--dense-id')
    parser.add_argument('--threads', type=int, required=True); parser.add_argument('--group', default='runs')
    parser.add_argument('--technical', action='store_true'); a = parser.parse_args()
    p = json.loads(a.protocol.read_text())
    if a.dense_id:
        batch(a.root, p, a.dense_id, a.threads, a.technical)
    else:
        execute(a.root, p, next(c for c in matrix(p) if c['id'] == a.condition), a.threads, a.group, a.technical)

"""Frozen-input orchestration for the separately authorized 18/252 POLUS campaign.

This module delegates all pruning decisions to existing implementations. It never
changes their controllers, rewards or tie rules. Manifests are preparation only;
the separate submitter requires native preflight evidence before calling bsub.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import resource
import time

import torch
from torch_geometric.data import Data

from gnn_pruning.config import load_yaml
from gnn_pruning.data import generate_exact_ratio_split
from gnn_pruning.evaluation import measure_inference_time
from gnn_pruning.models import build_model
from gnn_pruning.pruning.adaptive_v2 import run_greedy, run_q_learning
from gnn_pruning.pruning.experimental_v3 import run_variant
from gnn_pruning.pruning.lookahead import run_lookahead
from gnn_pruning.pruning.static_v2 import compute_scores, plan_static, apply_static_plan
from gnn_pruning.pruning.workflow_v2 import tensor_hash, _metrics, _validate_split
from gnn_pruning.surgery.local_v2 import prune_hidden_channels_local_v2
from gnn_pruning.training.trainer import DenseTrainer
from gnn_pruning.utils import set_seed

METHODS = {
    'static_random': 'random_local_v2',
    'static_magnitude': 'magnitude_local_v2',
    'static_snip': 'snip_pt_local_v2',
    'sequential_random': 'random_reachable_v3',
    'adaptive_greedy': 'adaptive_greedy_local_v2',
    'lookahead_depth2': 'lookahead_local_v4',
    'tabular_q': 'q_learning_local_v2',
}


def sha(path):
    """Return the SHA256 of immutable file bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    """Create evidence exactly once; never overwrite an earlier attempt."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def matrix(protocol):
    """Expand the fixed authorized matrix without dispatching any computation."""
    if protocol['datasets'] != ['cora', 'citeseer', 'pubmed'] or protocol['models'] != ['gcn', 'graphsage']:
        raise ValueError('Campaign dataset/model scope differs from authorization.')
    if protocol['dense_seeds'] != [101, 202, 303] or protocol['targets'] != [.3, .5]:
        raise ValueError('Campaign seed/target scope differs from authorization.')
    if protocol['methods'] != METHODS:
        raise ValueError('Explicit method identity map is required.')
    conditions = []
    for dataset in protocol['datasets']:
        for model in protocol['models']:
            for seed in protocol['dense_seeds']:
                identity = f'{dataset}-{model}-s{seed}'
                conditions.append(dict(id=identity, dataset=dataset, model=model, seed=seed,
                    pilot=dataset == 'cora' and seed == 101,
                    runs=[dict(id=f'{identity}-{method}-t{round(target*100)}', method=method,
                               implementation=implementation, target=target, policy_seed=seed)
                          for method, implementation in METHODS.items() for target in protocol['targets']]))
    return conditions


def dependency_plan(protocol):
    """Build a topological DAG; failed pilot/validation never releases bulk work."""
    conditions = matrix(protocol)
    jobs = []
    for c in [c for c in conditions if c['pilot']]:
        jobs += _condition_jobs(c, [])
    jobs.append(dict(id='pilot-gate', stage='gate', dependencies=[f"prune-{c['id']}" for c in conditions if c['pilot']]))
    for c in [c for c in conditions if not c['pilot']]:
        jobs += _condition_jobs(c, ['pilot-gate'])
    return jobs


def _condition_jobs(condition, roots):
    key = condition['id']
    return [dict(id='train-'+key, stage='train', condition=key, dependencies=roots),
            dict(id='validate-'+key, stage='validate', condition=key, dependencies=['train-'+key]),
            dict(id='prune-'+key, stage='prune', condition=key, dependencies=['validate-'+key])]


def _inputs(root, condition, protocol):
    if sha(root/'inputs/manifest.json') != protocol['input_manifest_sha256']:
        raise ValueError('Pinned input manifest changed.')
    manifest = json.loads((root/'inputs/manifest.json').read_text())
    item = manifest[condition['dataset']]
    path = root/'inputs'/item['file']
    if sha(path) != item['sha256']:
        raise ValueError('Graph file changed.')
    tensors = torch.load(path, map_location='cpu', weights_only=True)
    if tensor_hash(tensors) != item['tensor_sha256']:
        raise ValueError('Graph tensor hash mismatch.')
    data = Data(**tensors)
    split = generate_exact_ratio_split(int(data.num_nodes), seed=condition['seed']).to_dict()
    _validate_split(split, int(data.num_nodes))
    return data, split, item


def _architecture(data, protocol):
    return dict(in_channels=int(data.num_features), out_channels=int(data.y.max())+1,
                **protocol['architecture'])


def _load(root, condition, protocol):
    data, split, item = _inputs(root, condition, protocol)
    path = root/'dense'/condition['id']/'checkpoint.pt'
    cp = torch.load(path, map_location='cpu', weights_only=True)
    if cp['model_name'] != condition['model'] or cp['model_config'] != _architecture(data, protocol):
        raise ValueError('Checkpoint architecture/model mismatch.')
    if cp['split'] != split or cp['training_protocol'] != protocol['training'] or cp['seed'] != condition['seed']:
        raise ValueError('Checkpoint split/training/seed mismatch.')
    if cp['graph_tensor_sha256'] != item['tensor_sha256']:
        raise ValueError('Checkpoint graph mismatch.')
    model = build_model(cp['model_name'], **cp['model_config']).eval()
    model.load_state_dict(cp['model_state_dict'], strict=True)
    if tensor_hash(model.state_dict()) != cp['tensor_sha256']:
        raise ValueError('Checkpoint model tensor hash mismatch.')
    if any(not torch.isfinite(v).all() for v in model.state_dict().values()):
        raise ValueError('Nonfinite frozen checkpoint tensors.')
    return model, data, split, cp


def train(root, condition, protocol):
    """Use the existing DenseTrainer and unchanged validation-loss selection."""
    if not condition['pilot']:
        gate = json.loads((root/'pilot-gate.json').read_text())
        if gate['status'] != 'PASS' or gate['source_commit'] != protocol['source_commit']:
            raise ValueError('Pilot acceptance required before remaining dense training.')
    output = root/'dense'/condition['id']
    output.mkdir(parents=True, exist_ok=False)
    data, split, item = _inputs(root, condition, protocol)
    set_seed(condition['seed'])
    architecture = _architecture(data, protocol)
    model = build_model(condition['model'], **architecture)
    training = protocol['training']
    reused = root/'inputs/reuse'/condition['id']
    if (reused/'verified.json').exists():
        proof = json.loads((reused/'verified.json').read_text())
        original = torch.load(reused/'checkpoint.pt', map_location='cpu', weights_only=True)
        if proof['checkpoint_sha256'] != sha(reused/'checkpoint.pt') or proof['split'] != split:
            raise ValueError('Reused checkpoint provenance mismatch.')
        if proof['graph_tensor_sha256'] != item['tensor_sha256'] or proof['training_protocol'] != training:
            raise ValueError('Reused graph/training mismatch.')
        if original['model_name'] != condition['model'] or original['model_config'] != architecture:
            raise ValueError('Reused architecture mismatch.')
        model.load_state_dict(original['model_state_dict'], strict=True)
        if tensor_hash(model.state_dict()) != proof['tensor_sha256']:
            raise ValueError('Reused tensor identity mismatch.')
        cp = dict(model_name=condition['model'], model_config=architecture,
            model_state_dict=model.state_dict(), seed=condition['seed'], split=split,
            training_protocol=training, graph_tensor_sha256=item['tensor_sha256'],
            tensor_sha256=proof['tensor_sha256'], source_commit=protocol['source_commit'],
            origin=proof)
        torch.save(cp, output/'checkpoint.pt')
        write(output/'training.json', dict(reused=True, origin=proof, wall_seconds=0.,
            test_labels_exposed=False, validation=proof['validation'],
            checkpoint_sha256=sha(output/'checkpoint.pt'), source_commit=protocol['source_commit']))
        return
    worker = DenseTrainer(model, device='cpu', **training)
    work = data.clone()
    work.y[split['test']] = -1
    history = []
    started = time.perf_counter()
    result = worker.fit(work, torch.tensor(split['train']), torch.tensor(split['val']),
                        progress_callback=lambda row: history.append(row))
    cp = dict(model_name=condition['model'], model_config=architecture,
              model_state_dict=model.state_dict(), seed=condition['seed'], split=split,
              training_protocol=training, graph_tensor_sha256=item['tensor_sha256'],
              tensor_sha256=tensor_hash(model.state_dict()), source_commit=protocol['source_commit'])
    torch.save(cp, output/'checkpoint.pt')
    write(output/'training.json', dict(result=result.to_dict(), history=history,
          wall_seconds=time.perf_counter()-started, test_labels_exposed=False,
          validation=_metrics(model.eval(), data, torch.tensor(split['val'])),
          checkpoint_sha256=sha(output/'checkpoint.pt'), source_commit=protocol['source_commit']))


def validate(root, condition, protocol):
    """Strict reload, frozen graph/split/tensors and repeatable dense metrics gate."""
    model, data, split, cp = _load(root, condition, protocol)
    directory = root/'dense'/condition['id']
    recorded = json.loads((directory/'training.json').read_text())
    if sha(directory/'checkpoint.pt') != recorded['checkpoint_sha256']:
        raise ValueError('Checkpoint file hash changed.')
    val = _metrics(model, data, torch.tensor(split['val']))
    if val != recorded['validation']:
        raise ValueError('Dense validation metrics do not reload exactly.')
    write(directory/'validated.json', dict(status='PASS', checkpoint_sha256=sha(directory/'checkpoint.pt'),
        tensor_sha256=cp['tensor_sha256'], graph_tensor_sha256=cp['graph_tensor_sha256'],
        split_sha256=hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest(),
        validation=val, test=_metrics(model, data, torch.tensor(split['test'])),
        source_commit=protocol['source_commit']))


def search(model, data, split, run, protocol):
    """Delegate one frozen search; expose only train/val labels, count forwards."""
    work = data.clone()
    work.y[split['test']] = -1
    calls = 0
    def evaluate(candidate):
        nonlocal calls
        if not bool((work.y[split['test']] == -1).all()):
            raise RuntimeError('Test labels exposed during search.')
        calls += 1
        with torch.no_grad():
            pred = candidate.eval()(work).argmax(-1)[split['val']]
        return float((pred == work.y[split['val']]).float().mean())
    method = run['method']
    settings = protocol['adaptive']
    started = time.perf_counter()
    if method.startswith('static_'):
        criterion = {'static_random': 'random', 'static_magnitude': 'magnitude', 'static_snip': 'snip'}[method]
        scores = compute_scores(model, criterion, work, torch.tensor(split['train']))
        plan = plan_static(model, scores, criterion, run['target'], seed=run['policy_seed'],
                           tolerance=settings['tolerance'], min_channels=settings['min_channels'])
        final = apply_static_plan(model, plan)
        keeps, trace, training, qtable, decisions = plan.kept_indices, [], [], [], []
        stats = dict(validation_evaluations=0, terminal_reason='static_nearest_fallback' if plan.fallback else 'static_target_band', static_plan=asdict(plan))
    else:
        common = dict(target=run['target'], **settings)
        if method == 'adaptive_greedy':
            result = run_greedy(model, evaluate, **common)
        elif method == 'lookahead_depth2':
            result = run_lookahead(model, evaluate, depth=2, gamma=.9, **common)
        elif method == 'tabular_q':
            result = run_q_learning(model, evaluate, seed=run['policy_seed'], **protocol['q_learning'], **common)
        elif method == 'sequential_random':
            result = run_variant(model, evaluate, variant='random_reachable_v3', seed=run['policy_seed'], **common)
        else:
            raise ValueError('Unknown method; substitution prohibited.')
        final, keeps, trace, training, stats = result.model, result.kept_indices, result.trace, result.training_trace, result.statistics
        qtable = [dict(state=state, actions=row) for state, row in result.q_table.items()]
        decisions = getattr(result, 'decisions', [])
    elapsed = time.perf_counter()-started
    if calls != stats['validation_evaluations']:
        raise RuntimeError('Validation forward accounting mismatch.')
    return final, keeps, dict(trace=trace, training=training, q_table=qtable,
                             decisions=decisions, statistics=stats, search_seconds=elapsed,
                             observed_validation_evaluations=calls, test_labels_exposed=False)


def verify_projection(dense, final, keeps):
    """Check retained original identities against physical checkpoint tensors."""
    projected = dense
    for layer, ids in enumerate(keeps):
        if ids != sorted(set(ids)) or not ids:
            raise ValueError('Invalid retained IDs.')
        if len(ids) != dense.hidden_channel_dims[layer]:
            projected = prune_hidden_channels_local_v2(projected, layer, ids, 4)
    if tensor_hash(projected.state_dict()) != tensor_hash(final.state_dict()):
        raise ValueError('Physical tensors do not match retained original channels.')


def verify_completed(directory):
    """Reject missing, modified or additional output files."""
    marker = json.loads((directory/'completed.json').read_text())
    actual = {p.name: sha(p) for p in directory.iterdir() if p.is_file() and p.name != 'completed.json'}
    if marker['files'] != actual:
        raise ValueError('Run artifact integrity failed.')
    return json.loads((directory/'report.json').read_text())


def prune(root, condition, protocol):
    """Run the 14 predeclared tasks after successful checkpoint validation."""
    certificate = json.loads((root/'dense'/condition['id']/'validated.json').read_text())
    checkpoint = root/'dense'/condition['id']/'checkpoint.pt'
    if certificate['status'] != 'PASS' or certificate['source_commit'] != protocol['source_commit'] or certificate['checkpoint_sha256'] != sha(checkpoint):
        raise ValueError('Validated checkpoint required.')
    dense, data, split, cp = _load(root, condition, protocol)
    dense_hash = tensor_hash(dense.state_dict())
    if not condition['pilot']:
        gate = json.loads((root/'pilot-gate.json').read_text())
        if gate['status'] != 'PASS' or gate['source_commit'] != protocol['source_commit']:
            raise ValueError('Pilot acceptance required.')
    for run in condition['runs']:
        output = root/'runs'/run['id']
        output.mkdir(parents=True, exist_ok=False)
        final, keeps, evidence = search(dense, data, split, run, protocol)
        if tensor_hash(dense.state_dict()) != dense_hash:
            raise RuntimeError('Search mutated frozen dense model.')
        verify_projection(dense, final, keeps)
        arch = final.export_architecture_config(); arch.pop('name')
        torch.save(dict(model_name=condition['model'], model_config=arch,
            model_state_dict=final.state_dict(), original_kept_ids=keeps,
            source_checkpoint_sha256=sha(checkpoint)), output/'pruned.pt')
        loaded = torch.load(output/'pruned.pt', map_location='cpu', weights_only=True)
        rebuilt = build_model(loaded['model_name'], **loaded['model_config']).eval()
        rebuilt.load_state_dict(loaded['model_state_dict'], strict=True)
        if tensor_hash(rebuilt.state_dict()) != tensor_hash(final.state_dict()):
            raise RuntimeError('Pruned checkpoint failed strict reload.')
        before, after = (sum(p.numel() for p in m.parameters()) for m in (dense, final))
        achieved = 1-after/before
        report = dict(run=run, condition={k:v for k,v in condition.items() if k != 'runs'},
            source_commit=protocol['source_commit'], checkpoint_sha256=sha(checkpoint),
            graph_tensor_sha256=cp['graph_tensor_sha256'], split_sha256=certificate['split_sha256'],
            parameters_before=before, parameters_after=after, parameter_bytes=sum(p.numel()*p.element_size() for p in final.parameters()),
            checkpoint_bytes=(output/'pruned.pt').stat().st_size, achieved_compression=achieved,
            signed_target_deviation=achieved-run['target'], overshoot=max(0., achieved-run['target']),
            outside_symmetric_band=abs(achieved-run['target']) > protocol['adaptive']['tolerance'],
            final_widths=final.hidden_channel_dims, retained_original_ids=keeps,
            validation=_metrics(final, data, torch.tensor(split['val'])),
            test=_metrics(final, data, torch.tensor(split['test'])),
            inference=measure_inference_time(final, data, torch.device('cpu'), **protocol['benchmark']),
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            runtime=dict(python=platform.python_version(), torch=str(torch.__version__), threads=torch.get_num_threads()),
            strict_reload=True, retained_tensor_projection=True, **evidence)
        write(output/'report.json', report)
        write(output/'completed.json', dict(files={p.name:sha(p) for p in output.iterdir() if p.is_file()}))
        verify_completed(output)
        print(json.dumps(dict(completed=run['id'], search_seconds=evidence['search_seconds'])), flush=True)


def pilot_gate(root, protocol):
    """Technical acceptance only; no accuracy threshold, winner or protocol tuning."""
    records = []
    for c in matrix(protocol):
        if c['pilot']:
            for run in c['runs']:
                report = verify_completed(root/'runs'/run['id'])
                if not report['strict_reload'] or not report['retained_tensor_projection'] or report['test_labels_exposed']:
                    raise ValueError('Pilot integrity/isolation failed.')
                if report['parameters_after'] >= report['parameters_before']:
                    raise ValueError('Pilot did not physically prune.')
                records.append(dict(run=run['id'], search_seconds=report['search_seconds'],
                    validation_evaluations=report['observed_validation_evaluations']))
    if len(records) != 28:
        raise ValueError('Incomplete 28-run pilot.')
    write(root/'pilot-gate.json', dict(status='PASS', source_commit=protocol['source_commit'],
          records=records, accuracy_used_for_acceptance=False))


def main():
    """Execute one bounded DAG node; invocation itself never submits a job."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--stage', choices=('train','validate','prune','gate'), required=True)
    p.add_argument('--condition')
    a = p.parse_args()
    protocol = load_yaml(a.protocol)
    torch.set_num_threads(4)
    condition = next((c for c in matrix(protocol) if c['id'] == a.condition), None)
    if a.stage == 'gate':
        pilot_gate(a.root, protocol)
    elif condition is None:
        raise ValueError('Known condition required.')
    else:
        {'train':train, 'validate':validate, 'prune':prune}[a.stage](a.root, condition, protocol)


if __name__ == '__main__':
    main()

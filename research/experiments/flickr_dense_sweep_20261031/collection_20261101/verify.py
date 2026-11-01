"""Offline collection audit: no optimizer, training, pruning or held-out labels."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import tarfile

import torch

from gnn_pruning.experiments.flickr_dense import (
    digest, load_inputs, matrix, metrics, sha, tensors_hash, verify_run, write,
)
from gnn_pruning.models import build_model


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(collection: Path, pins: Path, output: Path):
    """Verify original pins, all output inventories, curves, models and metrics."""
    output.mkdir(exist_ok=False, parents=True)
    root = collection / 'raw'
    provenance = json.loads(pins.read_text())
    protocol = json.loads((root / 'protocol.json').read_text())
    receipt = json.loads((collection / 'collection.json').read_text())
    require(sha(collection / 'results.tar.gz') == receipt['archive_sha256'], 'Archive digest')
    require(sha(root / 'protocol.json') == provenance['protocol_file_sha256'], 'Protocol pin')
    require(sha(root / 'inputs/manifest.json') == provenance['input_manifest_sha256'], 'Input pin')
    require(protocol['source_commit'] == provenance['source_commit'], 'Source revision')
    require(digest(protocol) == provenance['protocol_canonical_sha256'], 'Canonical protocol')
    source = root / ('source-' + protocol['source_commit'] + '.tar.gz')
    require(sha(source) == provenance['source_archive_sha256'], 'Source archive pin')
    source_files = json.loads((root / 'source-files.json').read_text())
    with tarfile.open(source) as archive:
        members = {m.name.removeprefix('./'): m for m in archive.getmembers() if m.isfile()}
        require(set(members) == set(source_files), 'Source inventory')
        for name, expected in source_files.items():
            require(hashlib.sha256(archive.extractfile(members[name]).read()).hexdigest() == expected,
                    'Source file: ' + name)
    accepted = json.loads((root / 'acceptance-research-jobs.json').read_text())
    with (pins.parent / 'JOB_MANIFEST.csv').open() as stream:
        planned = {r['condition_id']: r for r in csv.DictReader(stream)}
    jobs = {r['job']['condition']: int(r['job_id']) for r in accepted if r['accepted']}
    conditions = matrix(protocol)
    require(set(jobs) == {c['id'] for c in conditions} and len(set(jobs.values())) == 32, 'Acceptance identities')
    require({p.name for p in (root / 'runs').iterdir()} == set(jobs), 'Run-directory identities')
    scheduler = json.loads((collection / 'scheduler.json').read_text())
    torch.set_num_threads(4)
    data, split, manifest = load_inputs(root / 'inputs')
    rows = []
    metric_deltas = []
    files = []
    for condition in conditions:
        identity = condition['id']
        run = root / 'runs' / identity
        report = verify_run(run, condition, root / 'inputs')
        inventory = json.loads((run / 'completed.json').read_text())
        require({p.name for p in run.iterdir()} == set(inventory) | {'completed.json'}, 'Unexpected run files')
        require(report['condition'] == condition and report['status'] == 'COMPLETED', 'Run identity')
        require(report['source_commit'] == protocol['source_commit'] and report['protocol_sha256'] == digest(protocol), 'Run source')
        require(not report['test_metrics_evaluated'], 'Test access flag')
        config = json.loads((run / 'resolved_config.json').read_text())
        require(config['training'] == protocol['training'] and not config['test_labels_exposed'], 'Training settings')
        require(report['config_sha256'] == digest(config), 'Config hash')
        require(report['config_sha256'] == planned[identity]['resolved_config_sha256'], 'Pre-submission config pin')
        environment = json.loads((run / 'environment.json').read_text())
        require(environment['torch_threads'] == 4 and int(environment['lsf']['LSB_JOBID']) == jobs[identity], 'Execution resources/identity')
        history_text = (collection / f'bhist-{jobs[identity]}.txt').read_text()
        require(sha(collection / f'bhist-{jobs[identity]}.txt') == scheduler[identity]['history_sha256'], 'History digest')
        require('Done successfully.' in history_text and 'Exited with exit code' not in history_text, 'Scheduler completion')
        require(report['actual_host'] in re.sub(r'\s+', '', history_text), 'Execution host')
        allocated = re.search(r'Allocated\s+(\d+)\s+Slot\(s\)', history_text)
        require(allocated is not None and int(allocated.group(1)) == 20, 'Exclusive slot allocation')
        require('4 Task(s)' in history_text and 'Exclusive Execution' in history_text, 'Requested CPU allocation')
        if planned[identity]['host_constraint'] != 'UNBOUND':
            require('polus-c4-' in report['actual_host'], 'Large-job host constraint')
        states = re.search(r'PEND\s+PSUSP\s+RUN\s+USUSP\s+SSUSP\s+UNKWN\s+TOTAL\s+([\d\s]+)', history_text)
        require(states is not None, 'Scheduler durations missing')
        durations = [int(v) for v in states.group(1).split()]
        require(len(durations) == 7 and sum(durations[:6]) == durations[6], 'Scheduler durations invalid')
        memory = re.search(r'MAX MEM:\s*([\d.]+)\s*([MG])bytes', history_text)
        require(memory is not None, 'Scheduler memory missing')
        history = [json.loads(line) for line in (run / 'history.jsonl').read_text().splitlines()]
        require([r['epoch'] for r in history] == list(range(1, len(history) + 1)), 'Epoch sequence')
        require(all(math.isfinite(v) for r in history for v in r.values()), 'Nonfinite curve')
        best = min(range(len(history)), key=lambda i: history[i]['val_loss']) + 1
        require(best == report['convergence']['best_epoch_1based'], 'Best epoch')
        require(len(history) == report['convergence']['total_epochs'] <= 500, 'Epoch count')
        require(len(history) == 500 or history[-1]['early_stopping_counter'] == 50, 'Patience stop')
        checkpoint = torch.load(run / 'checkpoint.pt', weights_only=True, map_location='cpu')
        partial = torch.load(run / 'best.partial.pt', weights_only=True, map_location='cpu')
        require(tensors_hash(partial['model_state_dict']) == checkpoint['tensor_sha256'] == report['tensor_sha256'], 'Best-state identity')
        model = build_model(condition['model'], **config['architecture'])
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        dims = [manifest['features']] + [condition['width']] * (condition['layers'] - 1) + [manifest['classes']]
        factor = 1 if condition['model'] == 'gcn' else 2
        count = sum(factor * a * b + b for a, b in zip(dims, dims[1:]))
        require(sum(p.numel() for p in model.parameters()) == report['parameters'] == count, 'Parameter oracle')
        repeated = metrics(model, data, split)  # existing frozen train/validation nodes only
        for subset in ('train', 'val'):
            for key in ('class_counts', 'prediction_counts'):
                require(repeated[subset][key] == report['metrics'][subset][key], 'Prediction/count disagreement')
            for key in ('accuracy', 'macro_f1', 'loss'):
                delta = abs(repeated[subset][key] - report['metrics'][subset][key])
                metric_deltas.append(dict(id=identity, subset=subset, metric=key, absolute_delta=delta))
                require(delta <= (5e-5 if key == 'loss' else 1e-12), 'Reload metric disagreement')
        row = dict(id=identity, job_id=jobs[identity], status='DONE', audit='PASS', actual_host=report['actual_host'],
                   requested_threads=4, allocated_slots=int(allocated.group(1)), lsf_pending_seconds=durations[0], lsf_run_seconds=durations[2],
                   lsf_cpu_seconds=float(re.search(r'CPU time used is ([\d.]+) seconds', history_text).group(1)),
                   lsf_peak_memory_reported=memory.group(1) + memory.group(2) + 'bytes', epochs=len(history), best_epoch=best,
                   output_inventory_files=len(inventory), parameters=count, test_evaluated=False)
        rows.append(row)
        print(identity + ' PASS', flush=True)
    for path in sorted(root.rglob('*')):
        if path.is_file():
            files.append(dict(path=path.relative_to(root).as_posix(), bytes=path.stat().st_size, sha256=sha(path)))
    write(output / 'RAW_FILES.json', files)
    with (output / 'audit_runs.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    write(output / 'METRIC_RELOAD_DELTAS.json', metric_deltas)
    write(output / 'AUDIT.json', dict(expected=32, PASS=len(rows), FAIL=0, SKIPPED=0,
          source_files_verified=len(source_files), raw_files=len(files), raw_bytes=sum(r['bytes'] for r in files),
          raw_manifest_sha256=sha(output / 'RAW_FILES.json'), archive_sha256=receipt['archive_sha256'],
          maximum_metric_delta={k:max(r['absolute_delta'] for r in metric_deltas if r['metric'] == k)
                                for k in ('accuracy', 'macro_f1', 'loss')},
          metric_tolerances=dict(accuracy=1e-12, macro_f1=1e-12, loss=5e-5),
          lsf_active_job_hours=sum(r['lsf_run_seconds'] for r in rows)/3600,
          lsf_pending_job_hours=sum(r['lsf_pending_seconds'] for r in rows)/3600,
          lsf_cpu_hours=sum(r['lsf_cpu_seconds'] for r in rows)/3600,
          test_labels_loaded=False, training_or_pruning_executed=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('collection', 'pins', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    audit(args.collection, args.pins, args.output)

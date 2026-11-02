"""Verify downloaded Flickr evidence and export compact recorded results.

Standard library only: no model loading, inference, training, pruning or tests.
Usage: python3 collect.py --raw EXTERNAL_COLLECTION --output NEW_DIRECTORY
The archive must already be downloaded together with its server receipt.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import tarfile


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def count(model, widths):
    dims = [500, *widths, 7]
    factor = 2 if model == 'graphsage' else 1
    return sum(factor * a * b + b for a, b in zip(dims, dims[1:]))


def main(raw, output):
    receipt = read(raw / 'receipt.json')
    archive = raw / 'results.tar.gz'
    require(sha(archive) == receipt['archive_sha256'], 'Archive SHA256 differs')
    root = raw / 'extracted'
    if not root.exists():
        root.mkdir()
        with tarfile.open(archive) as tar:
            members = tar.getmembers()
            require(all(m.isfile() and not Path(m.name).is_absolute()
                        and '..' not in Path(m.name).parts for m in members), 'Unsafe archive')
            tar.extractall(root, members=members, filter='data')
    inventory = read(root / '_collection/inventory.json')
    require(sha(root / '_collection/inventory.json') == receipt['inventory_sha256'], 'Inventory SHA differs')
    require(len(inventory) == receipt['files'], 'Inventory count differs')
    for item in inventory:
        p = root / item['path']
        require(p.stat().st_size == item['size'] and sha(p) == item['sha256'], 'File differs: ' + item['path'])
    source_files = read(root / 'source-files.json')
    for name, pin in source_files.items():
        require(sha(root / 'source' / name) == pin, 'Source pin differs: ' + name)
    parent = Path(__file__).resolve().parent.parent
    accepted = read(parent / 'SUBMISSION_RECEIPTS.json')
    protocol = read(root / 'protocol.json')
    tree = subprocess.check_output(['git', 'ls-tree', '--full-tree', '-rz', protocol['source_commit']], cwd=parent)
    git_files = {}
    for entry in tree.split(b'\0'):
        if not entry:
            continue
        metadata, name = entry.split(b'\t', 1)
        mode, kind, pin = metadata.decode().split()
        require(kind == 'blob' and mode in ('100644', '100755'), 'Unexpected Git entry')
        git_files[name.decode()] = pin
    require(set(git_files) == set(source_files), 'Pinned Git tree file set differs')
    for name, pin in git_files.items():
        content = (root / 'source' / name).read_bytes()
        require(hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest() == pin, 'Git source blob differs: ' + name)
    protocol_digest = hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    require(sha(root / 'protocol.json') == sha(parent / 'protocol.json'), 'Protocol differs')
    profiles = read(root / 'execution-resources.json')
    require(len(accepted) == 56 and len({r['job_id'] for r in accepted}) == 56, 'Acceptance matrix differs')
    require({p.name for p in (root / 'runs').iterdir()} == {r['condition_id'] for r in accepted}, 'Run set differs')
    history = (root / '_collection/scheduler-history.txt').read_text()
    blocks = re.split(r'(?=Job <\d+>,)', history)
    by_job = {int(re.match(r'Job <(\d+)>', b)[1]): b for b in blocks if re.match(r'Job <\d+>', b)}
    rows = []
    hashed_outputs = 0
    for receipt_row in accepted:
        identity = receipt_row['condition_id']
        job = receipt_row['job']
        jid = receipt_row['job_id']
        directory = root / 'runs' / identity
        require(not (directory / 'failed.json').exists(), 'Failed marker: ' + identity)
        marker = read(directory / 'completed.json')
        require(set(marker) == {p.name for p in directory.iterdir() if p.is_file()} - {'completed.json'}, 'Output set differs')
        for name, pin in marker.items():
            require(sha(directory / name) == pin, 'Completion hash differs: ' + identity + '/' + name)
            hashed_outputs += 1
        for name, key in [('receipt.json', 'receipt_file_sha256'), ('intent.json', 'intent_file_sha256')]:
            require(sha(root / 'submissions' / identity / name) == receipt_row[key], 'Original claim differs')
        report = read(directory / 'report.json')
        config = read(directory / 'resolved_config.json')
        c = report['condition']
        require(c == config['condition'] and c['id'] == identity, 'Condition identity differs')
        for key in ('architecture', 'method', 'target', 'dense_seed', 'split_seed', 'policy_seed'):
            require(c[key] == job[key], 'Condition field differs: ' + key)
        require(not report['technical'] and not config['technical'], 'Technical output masquerades as science')
        require(report['source_commit'] == protocol['source_commit'] == receipt_row['source_commit'], 'Source identity differs')
        require(report['protocol_sha256'] == config['protocol_sha256'] == protocol_digest, 'Canonical protocol digest differs')
        require(report['dense_checkpoint_sha256'] == job['dense_checkpoint_sha256'], 'Dense checkpoint differs')
        require(report['split_sha256'] == protocol['split_sha256'], 'Split differs')
        manifest = read(root / 'inputs/manifest.json')
        require(report['graph_tensor_sha256'] == manifest['training_tensor_sha256'], 'Graph differs')
        require(sha(root / 'inputs/manifest.json') == protocol['input_manifest_sha256'], 'Input pin differs')
        require(sha(root / 'inputs/graph.pt') == manifest['graph_file_sha256'] and sha(root / 'inputs/split.json') == manifest['split_file_sha256'], 'Frozen graph/split bytes differ')
        require(sha(root / 'dense' / c['architecture'] / 'checkpoint.pt') == job['dense_checkpoint_sha256'], 'Dense bytes differ')
        require(sha(directory / 'checkpoint.pt') == report['checkpoint_sha256'], 'Final checkpoint bytes differ')
        widths = report['final_widths']
        kept = report['retained_original_ids']
        require(widths == list(map(len, kept)) and len(widths) == job['layers'] - 1, 'Final mask dimensions differ')
        require(all(len(set(ids)) == len(ids) and all(0 <= i < job['hidden_width'] for i in ids) for ids in kept), 'Invalid original IDs')
        require(all(w >= 4 for w in widths), 'Minimum width violated')
        before = count(job['model'], [job['hidden_width']] * (job['layers'] - 1))
        after = count(job['model'], widths)
        require((before, after) == (report['parameters_before'], report['parameters_after']) and after < before, 'Physical count inconsistent')
        compression = 1 - after / before
        require(abs(compression - report['achieved_compression']) < 1e-14, 'Compression inconsistent')
        require(report['test_labels_accessed'] is False and report['strict_reload'] and report['retained_projection'], 'Worker integrity flags failed')
        stats = report['statistics']
        require(stats['complete'], 'Incomplete deployment')
        profile = profiles[identity]
        require(report['threads'] == profile['threads'], 'Computational threads differ')
        if profile['threads'] == 12:
            require('polus-c4' in report['actual_host'] and report['interop_threads'] == 1, 'C4 special resources differ')
            require(sorted(report['execution_resources']['observed_affinity']) == sorted(profile['cpu_affinity']), 'C4 affinity differs')
        training = read(directory / 'training.json')
        qtable = read(directory / 'q_table.json')
        cache = read(directory / 'evaluation_cache.json')
        deployment = read(directory / 'deployment.json')
        require(deployment[-1]['widths_after'] == widths and deployment[-1]['original_kept_ids'] == kept, 'Deployment endpoint differs')
        for metric in ('accuracy', 'macro_f1', 'loss'):
            require(math.isfinite(report['validation'][metric]), 'Nonfinite validation metric')
        require(report['dense_validation'] == protocol['dense'][c['architecture']]['validation'], 'Recorded dense metrics differ')
        require(len(cache) == report['validation_evaluation_requests'], 'Cache request count differs')
        require(sum(not e['reused'] for e in cache) == report['observed_validation_evaluations'], 'Forward count differs')
        if c['method'] == 'tabular_q':
            require(stats['episodes'] == 30 and {t['episode'] for t in training} == set(range(30)), 'Not 30 complete Q episodes')
            for episode in range(30):
                require([t for t in training if t['episode'] == episode][-1]['terminal_reason'] is not None, 'Q episode lacks terminal transition')
            require(sum(len(r['actions']) for r in qtable) == stats['q_entries'], 'Q entry count differs')
        log = (root / 'logs' / (identity + '.' + str(jid) + '.out')).read_text()
        require('Successfully completed.' in log, 'LSF stdout lacks success: ' + str(jid))
        require(jid in by_job and 'Done successfully.' in by_job[jid], 'History lacks DONE: ' + str(jid))
        flat = re.sub(r'\n\s+', '', by_job[jid])
        cpu = re.search(r'CPU time used is ([0-9.]+) seconds', flat)
        peak = re.search(r'MAX MEM:\s*([0-9.]+\s*\w+)', flat)
        time_row = re.search(r'PEND\s+PSUSP\s+RUN\s+USUSP\s+SSUSP\s+UNKWN\s+TOTAL\s+([\d\s]+)', by_job[jid])
        times = [int(v) for v in time_row[1].split()] if time_row else []
        rows.append(dict(condition_id=identity,job_id=jid,scheduler_state='DONE',architecture=c['architecture'],model=job['model'],layers=job['layers'],hidden_width=job['hidden_width'],target=c['target'],method=c['method'],dense_seed=101,split_seed=101,policy_seed=101,actual_host=report['actual_host'],threads=report['threads'],interop_threads=report['interop_threads'],validation_accuracy=report['validation']['accuracy'],validation_macro_f1=report['validation']['macro_f1'],dense_validation_accuracy=report['dense_validation']['accuracy'],dense_validation_macro_f1=report['dense_validation']['macro_f1'],parameters_before=before,parameters_after=after,achieved_compression=compression,outside_band=report['outside_band'],final_widths=json.dumps(widths),retained_ids_sha256=hashlib.sha256(json.dumps(kept,separators=(',',':')).encode()).hexdigest(),search_seconds=report['search_seconds'],validation_requests=report['validation_evaluation_requests'],actual_search_forwards=report['observed_validation_evaluations'],cache_hits=stats['evaluation_cache_hits'],training_transitions=len(training),deployment_actions=len(deployment),q_entries=stats.get('q_entries',0),q_table_coverage=stats.get('q_table_coverage',0),deployment_unseen_choices=stats.get('deployment_unseen_choices',0),terminal_reason=stats['terminal_reason'],peak_process_rss_bytes=report['peak_process_rss_bytes'],lsf_peak_memory=peak[1] if peak else '',lsf_cpu_seconds=float(cpu[1]) if cpu else '',lsf_pending_seconds=times[0] if len(times)==7 else '',lsf_run_seconds=times[2] if len(times)==7 else '',source_commit=report['source_commit'],checkpoint_sha256=report['checkpoint_sha256'],dense_checkpoint_sha256=report['dense_checkpoint_sha256'],split_sha256=report['split_sha256'],graph_tensor_sha256=report['graph_tensor_sha256'],collection_verification='PASS',worker_strict_reload_reported=True,worker_projection_reported=True))
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'runs.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n");writer.writeheader();writer.writerows(rows)
    save(output / 'VERIFICATION.json', dict(status='PASS',completed=56,failed=0,pending=0,running=0,inventory_files_verified=len(inventory),completion_output_hashes_verified=hashed_outputs,source_files_verified=len(source_files),source_git_blobs_verified=len(git_files),source_commit=protocol['source_commit'],archive=receipt,raw_directory='${GNN_ARTIFACT_ROOT}/'+raw.name,scope='Archive/file/source/input hashes and recorded metadata consistency; no independent tensor reload or inference.',new_experiments=0,pytest_executed=False,pytorch_executed=False))
    print(json.dumps(dict(completed=len(rows),files=len(inventory),source_files=len(source_files),outside_band=sum(r['outside_band'] for r in rows),hosts=sorted({r['actual_host'] for r in rows})),indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.raw, args.output)

"""Collect completed condition summaries serially; no model or dataset imports."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import os
import re
from datetime import datetime, timezone

root = Path(__file__).resolve().parent
campaign_name = os.environ.get('POLUS_CAMPAIGN', 'pilot-cpu-101-c4-v2')
campaign = root / 'results' / campaign_name
conditions = [f'{dataset}_{model}_l3_h32' for dataset in ('cora', 'citeseer', 'pubmed') for model in ('gcn', 'graphsage')]
expected_commit = '72293149d8fff186dd268789e7b8fea5848c9360'
methods = {'random', 'global_magnitude', 'snip', 'grasp', 'group_lasso', 'adaptive_layerwise', 'q_learning_tabular'}
statuses = []
rows = []
for condition in conditions:
    job_root = campaign / 'jobs' / condition
    output = job_root / 'seed_101' / condition
    claim = root / 'submissions' / campaign_name / condition
    record = {'condition': condition, 'job_id': (claim / 'job_id').read_text().strip() if (claim / 'job_id').exists() else None,
              'required_host': 'polus-c4-ib.bmc.hpc.cs.msu.ru', 'status': 'NOT_COMPLETE'}
    lsf_log = root / 'logs/lsf' / ('pilot-cpu.' + str(record['job_id']) + '.out')
    if lsf_log.exists():
        text = lsf_log.read_text(errors='replace')
        start = re.search(r'^Started at (.+)$', text, re.M)
        stop = re.search(r'^Terminated at (.+)$', text, re.M)
        host = re.search(r'^Job was executed on host\(s\) <([^>]+)>', text, re.M)
        if host:
            record['execution_host'] = host.group(1).split('*')[-1]
            if record['execution_host'] != record['required_host']:
                raise SystemExit('Unexpected execution host: ' + condition)
        if start and stop:
            fmt = '%a %b %d %H:%M:%S %Y'
            record['job_walltime_sec'] = (datetime.strptime(stop.group(1), fmt) - datetime.strptime(start.group(1), fmt)).total_seconds()
            if record['job_walltime_sec'] > 180 * 60:
                raise SystemExit('Job exceeded its 180-minute budget: ' + condition)
        if re.search(r'^Subject: Job [0-9]+: .+ Done$', text, re.M):
            record['lsf_status'] = 'DONE'
    if (output / 'failure.json').exists():
        record['status'] = 'FAILED'
        record['failure'] = json.loads((output / 'failure.json').read_text())
    elif (record.get('lsf_status') == 'DONE' and (output / 'condition_complete.json').exists()
          and (job_root / 'pilot_run_manifest.json').exists()):
        complete = json.loads((output / 'condition_complete.json').read_text())
        manifest = json.loads((job_root / 'pilot_run_manifest.json').read_text())
        if complete['git_commit'] != expected_commit or manifest.get('status') != 'complete':
            raise SystemExit('Completion/provenance mismatch: ' + condition)
        csv_path = output / 'validation_summary.csv'
        with csv_path.open(newline='') as handle:
            source_rows = list(csv.DictReader(handle))
        observed = {(row['method'], float(row['requested_sparsity'])) for row in source_rows}
        expected = {(method, level) for method in methods for level in (0.3, 0.5)}
        if len(source_rows) != 14 or observed != expected or {row['condition'] for row in source_rows} != {condition}:
            raise SystemExit('Incomplete method/target matrix: ' + condition)
        record.update(status='COMPLETE', elapsed_sec=complete['elapsed_sec'], post_prune_rows=len(source_rows),
                      validation_csv_sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest())
        rows.extend(source_rows)
    elif (output / 'progress.log').exists():
        lines = (output / 'progress.log').read_text(errors='replace').splitlines()
        record['last_progress_line'] = lines[-1] if lines else ''
    statuses.append(record)

payload = {'generated_at_utc': datetime.now(timezone.utc).isoformat(), 'git_commit': expected_commit,
           'platform': 'polus_cpu', 'campaign': campaign_name, 'seed': 101, 'required_host': 'polus-c4-ib.bmc.hpc.cs.msu.ru',
           'all_conditions_complete': all(row['status'] == 'COMPLETE' for row in statuses),
           'complete_conditions': sum(row['status'] == 'COMPLETE' for row in statuses),
           'expected_conditions': 6, 'post_prune_rows_present': len(rows), 'expected_post_prune_rows': 84,
           'test_results_used_for_selection': False, 'historical_results_pooled': False, 'conditions': statuses}
campaign.mkdir(parents=True, exist_ok=True)
def save_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)
save_json(campaign / 'campaign-status.json', payload)
# Only this serial collector writes the campaign-level table. It excludes test metrics.
target = campaign / 'pilot_validation_summary.csv'
temporary = target.with_suffix('.csv.tmp')
with temporary.open('w', newline='') as handle:
    if rows:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
temporary.replace(target)
print(json.dumps(payload, indent=2))

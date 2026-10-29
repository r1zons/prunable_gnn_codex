"""Verify compact accounting and unchanged evidence; bind the analysis by hashes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay-log', type=Path, required=True)
    parser.add_argument('--test-log', type=Path, required=True)
    args = parser.parse_args()
    v = read(ROOT/'verification.json')
    assert v['status'] == 'PASS' and v['primary_q_runs'] == 84 and v['deployment_steps'] == 803
    rows = list(csv.DictReader((ROOT/'run_diagnostics.csv').open()))
    steps = list(csv.DictReader((ROOT/'deployment_steps.csv').open()))
    assert len(rows) == 84 and len(steps) == 803
    assert sum(int(r['train_observations']) for r in rows) == v['training_decisions'] == 136118
    assert sum(int(r['train_aliased_states']) for r in rows) == 837
    assert sum(int(r['train_material_conflicts']) for r in rows) == 245
    assert sum(int(r['deployment_training_alias_exposure']) for r in rows) == 24
    assert sum(int(r['deployment_cross_mask_value_transfer']) for r in rows) == 1
    assert sum(r['same_final_mask']=='True' for r in rows) == 53
    assert sum(r['first_action_divergence']!='' for r in rows) == 82
    assert sum(r['selected_known']=='False' for r in steps) == 29
    assert len(list(csv.DictReader((ROOT/'summary.csv').open()))) == 28
    prior = read(ROOT.parent/'analysis_manifest.json')
    for name, metadata in prior['files'].items():
        assert sha(ROOT.parent/name) == metadata['sha256'], name
    for name, expected in v['prior_files_sha256'].items():
        assert sha(ROOT.parent/name) == expected
    collection = Path(read(ROOT.parent/'provenance.json')['archive']).parent
    assert sha(collection/'v4-primary-results.tar.gz') == v['archive_sha256']
    inventory = read(collection/'polus/collection/inventory.json')
    assert len(inventory) == 1285
    for entry in inventory:
        path = collection/'polus'/entry['path']
        assert sha(path) == entry['sha256'] and path.stat().st_size == entry['bytes']
    external = Path(v['external_root'])
    for name, metadata in v['external_indexes'].items():
        assert sha(external/name) == metadata['sha256']
    tests = args.test_log.read_text()
    assert 'Ran 11 tests' in tests and tests.rstrip().endswith('OK')
    assert '84/84 ' in args.replay_log.read_text() and 'Traceback' not in args.replay_log.read_text()
    files = {str(p.relative_to(ROOT)): dict(sha256=sha(p), bytes=p.stat().st_size)
             for p in sorted(ROOT.rglob('*')) if p.is_file() and p != ROOT/'manifest.json'}
    for p in ROOT.rglob('*.svg'):
        assert all(line == line.rstrip() for line in p.read_text().splitlines())
    manifest = dict(status='PASS', base_branch='research/v4-tabular-diagnostic-20261029',
        base_commit='26632ab71beffd62955eca3b939139b9d9df367a',
        scientific_source=v['original_scientific_source'], archive_sha256=v['archive_sha256'],
        primary_q_policies=84, deployment_decisions=803, training_decisions=136118,
        summary_groups=28, offline_diagnostic_tests=11, figures_visually_reviewed=2,
        original_inventory_files_unchanged=1285, earlier_analysis_unchanged=True,
        new_experiments=False, algorithms_changed=False,
        replay_log_sha256=sha(args.replay_log), tests_log_sha256=sha(args.test_log),
        external_root=str(external.parent), files=files)
    (ROOT/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('files','external_root')}))


if __name__ == '__main__':
    main()

"""Pin or verify compact offline analysis; never execute an experimental model."""
from pathlib import Path
import argparse
import csv
import json
import platform
import re

from gnn_pruning.experiments.complete_scientific import load_rows, sha, validate

COUNTS = {
    'PAIRED_COMPARISONS.csv': 756, 'PAIRED_SUMMARY.csv': 252,
    'METHOD_SUMMARY.csv': 84, 'PARETO_DIAGNOSTICS.csv': 252,
    'GREEDY_LOOKAHEAD_TRAJECTORIES.csv': 36, 'GREEDY_PREVIEW_BRIDGE.csv': 72,
    'Q_RUN_DIAGNOSTICS.csv': 36, 'Q_DEPLOYMENT_DECISIONS.csv': 720,
    'Q_EPISODE_SUMMARY.csv': 1080, 'Q_TARGET_PAIR_CHECK.csv': 18,
}
CONTEXT = (
    'research/experiments/local_structured_v2_q_random_pilot_20261027/README.md',
    'research/experiments/local_structured_v3_sprint_20261027/results/REPORT.md',
    'research/experiments/polus_v4_diagnostic_20261029/results/REPORT.md',
)


def pin(path):
    return dict(sha256=sha(path), bytes=path.stat().st_size)


def verify(repo, analysis):
    manifest = json.loads((analysis/'ANALYSIS_MANIFEST.json').read_text())
    for category in ('primary_inputs', 'context_only_inputs', 'analysis_code', 'outputs'):
        for name, expected in manifest[category].items():
            if pin(repo/name) != expected:
                raise ValueError('MANIFEST_HASH_MISMATCH: '+name)
    print(json.dumps(dict(status='PASS', files_verified=sum(len(manifest[k]) for k in
          ('primary_inputs','context_only_inputs','analysis_code','outputs')),
          raw_inventory_rechecked=False, note='Use table generator for complete raw verification.')))


def build(args):
    repo=args.repo.resolve(); analysis=args.analysis.resolve()
    collection=analysis.parent/'collection'; raw=args.raw.resolve(); logs=args.logs.resolve()
    rows=load_rows(collection/'verified_results.csv')
    population=validate(rows,collection,raw)
    receipt=json.loads((collection/'collection_receipt.json').read_text())
    source=raw.parent/('source-'+receipt['source_commit']+'.tar.gz')
    anchors={str(source):receipt['source_archive_sha256'], str(raw/'protocol.json'):receipt['protocol_sha256'],
             str(raw/'inputs/manifest.json'):'414fbd317890e12c3be85fbe3bc21b8e09326662a8b66b6de5305b8a9fab265f',
             str(raw/'submission-evidence-20261031T021106Z.tar.gz'):receipt['original_submission_receipt_archive_sha256']}
    for name, expected in anchors.items():
        if sha(Path(name))!=expected: raise ValueError('ARCHIVE_ANCHOR: '+name)
    for name, expected in COUNTS.items():
        with (analysis/name).open() as stream:
            if len(list(csv.DictReader(stream)))!=expected: raise ValueError('OUTPUT_ROW_COUNT: '+name)
    final_log=logs/'focused-tests-release.log'
    if not re.search(r'^19 passed in [0-9.]+s$',final_log.read_text(),re.M):
        raise ValueError('FOCUSED_TEST_LOG')
    if any(word in final_log.read_text().lower() for word in ('failed','skipped','error')):
        raise ValueError('UNEXPECTED_TEST_OUTCOME')
    repeated={}
    for p in sorted(args.repeat_a.iterdir()):
        if p.suffix not in ('.json','.csv'): continue
        second=args.repeat_b/p.name
        if not second.is_file() or sha(p)!=sha(second) or sha(analysis/p.name)!=sha(p):
            raise ValueError('NONDETERMINISTIC_CORE_OUTPUT: '+p.name)
        repeated[p.name]=sha(p)
    if len(repeated)!=12: raise ValueError('REPEAT_OUTPUT_COUNT')
    rel=lambda p:str(p.relative_to(repo))
    inputs={rel(p):pin(p) for p in sorted(collection.rglob('*')) if p.is_file()}
    code=[repo/'src/gnn_pruning/experiments/complete_scientific.py',repo/'tests/test_complete_scientific.py']+sorted((analysis/'scripts').glob('*.py'))
    outputs={rel(p):pin(p) for p in sorted(analysis.rglob('*')) if p.is_file() and p.name!='ANALYSIS_MANIFEST.json' and p not in code}
    dense={}
    for r in rows:
        dense[r['condition_id']]={k:r[k] for k in ('dataset','architecture','dense_training_seed','dense_checkpoint_sha256',
          'dense_tensor_sha256','graph_tensor_sha256','split_sha256','dense_origin_source_commit')}
    manifest=dict(schema='polus-offline-scientific-analysis-v1',date='2026-10-31',
        branch='analysis/polus-complete-scientific-20261031',
        audited_start_commit='8f24513da09c0291a562fbb31a688da2698671b9',experimental_source_commit=receipt['source_commit'],
        analysis_commit='Git commit containing this manifest; code pinned below to avoid self-reference',
        population=population,csv_rows=COUNTS,independent_unit='3 paired dense training/split seeds per dataset/model/target',
        dense_input_identities=dense,
        primary_inputs=inputs,context_only_inputs={name:pin(repo/name) for name in CONTEXT},historical_rows_pooled=0,
        raw_anchors={name:dict(sha256=value,bytes=Path(name).stat().st_size) for name,value in anchors.items()},
        raw_inventory=dict(path=rel(collection/'artifact_manifest.csv'),sha256=sha(collection/'artifact_manifest.csv'),
                           files=1226,bytes=213152398,reverification='all file sizes and SHA256 rechecked in this analysis'),
        raw_run_report_hashes={r['run_id']:r['output_report_sha256'] for r in rows},
        analysis_code={rel(p):pin(p) for p in code},outputs=outputs,
        tests=dict(PASS=19,FAIL=0,SKIPPED=0,command='PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest tests/test_complete_scientific.py -ra',
                   runtime='Python 3.12.9',external_log=str(final_log),log_hash=pin(final_log),full_suite_run_in_this_task=False),
        determinism=dict(independent_table_generations=2,byte_identical_files=repeated,first=str(args.repeat_a),second=str(args.repeat_b)),
        runtime=dict(tables_and_tests='Python 3.12.9',reports='Python 3.14.3',manifest='Python '+platform.python_version(),
                     figures=json.loads((analysis/'FIGURES/ENVIRONMENT.json').read_text())),
        external_analysis_root=str(logs),raw_root=str(raw),
        constraints=dict(new_training=False,new_pruning=False,new_inference=False,new_model_timing=False,new_jobs=False,
                         server_access=False,algorithm_edits=False,historical_evidence_edits=False,manuscript_edits=False,
                         test_based_selection=False,push=False,merge=False))
    (analysis/'ANALYSIS_MANIFEST.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    verify(repo,analysis)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path('.'))
    parser.add_argument('--analysis',type=Path,required=True)
    parser.add_argument('--verify-only',action='store_true')
    for option in ('raw','logs','repeat-a','repeat-b'):parser.add_argument('--'+option,type=Path)
    args=parser.parse_args()
    if args.verify_only:verify(args.repo.resolve(),args.analysis.resolve())
    else:
        if any(getattr(args,k) is None for k in ('raw','logs','repeat_a','repeat_b')):
            parser.error('Building requires --raw, --logs, --repeat-a and --repeat-b.')
        build(args)

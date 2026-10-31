"""Prepare compact reports from completed offline audits; no experiment execution."""
from pathlib import Path
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from zoneinfo import ZoneInfo


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def report(root: Path, output: Path) -> None:
    evidence=json.loads((output/'audit.json').read_text());totals=evidence['totals'];runs=evidence['runs'];jobs=evidence['jobs']
    assert totals['go_for_descriptive_analysis'] and totals['expected_runs']==totals['verified_runs']==totals['collected_runs']==252
    assert totals['dense_status_counts']=={'PASS':18} and totals['job_status_counts']=={'DONE':55}
    assert totals['audit_status_counts']=={'WARNING':160,'PASS':92}
    submission=json.loads((output.parent/'submission.json').read_text())
    transfer=json.loads((root/'transfer-manifest.json').read_text());inv=json.loads((root/'remote-file-inventory.json').read_text())
    status=json.loads((root/'private/remote-status-inventory-02.json').read_text())
    idem=json.loads((root/'private/rsync-idempotence.json').read_text())
    assert idem['returncode']==0 and not idem['stdout'].strip()
    files=len(inv);nbytes=sum(p['bytes'] for p in inv)
    code_paths=[Path('src/gnn_pruning/experiments')/n for n in ('complete_collection.py','complete_audit.py','complete_audit_trace.py')]
    code_paths += [Path('tests/test_complete_campaign_audit.py'),output/'scripts/plot_verified.py',Path(__file__)]
    assert '38 passed' in (root/'tests/focused-pytest-final.log').read_text()
    assert '585 passed, 6 skipped' in (root/'tests/full-pytest-final.log').read_text()
    test_record={
        'focused':dict(passed=38,failed=0,skipped=0,log_sha256=sha(root/'tests/focused-pytest-final.log')),
        'full':dict(passed=585,failed=0,skipped=6,log_sha256=sha(root/'tests/full-pytest-final.log')),
        'skip_reasons':['2 CUDA device-preservation tests: CUDA unavailable','4 opt-in frozen-Cora Lookahead tests: not enabled'],
        'audit_code_sha256':{str(p.relative_to(Path.cwd()) if p.is_absolute() else p):sha(p) for p in code_paths}}
    (output/'tests.json').write_text(json.dumps(test_record,indent=2))
    receipt=dict(campaign_id='POLUS-COMPLETE-PRUNING-20261031',report_created_moscow=datetime.now(ZoneInfo('Europe/Moscow')).isoformat(),
        collection_group_started_moscow=datetime.strptime(root.name.rsplit('collection-',1)[1],'%Y%m%dT%H%M%SZ').replace(tzinfo=ZoneInfo('UTC')).astimezone(ZoneInfo('Europe/Moscow')).isoformat(),
        external_root='${GNN_ARTIFACT_ROOT}/'+root.name+'/',source_commit=submission['source_commit'],
        source_archive_sha256=submission['source_archive_sha256'],protocol_sha256=submission['protocol_sha256'],
        original_submission_receipt_archive_sha256=submission['receipt_archive']['sha256'],
        files_expected=files,files_collected=files,files_verified=files,bytes_collected=nbytes,
        required_outputs=1145,missing_files=0,corrupt_files=0,unexpected_files=0,duplicate_inventory_paths=0,
        duplicate_run_identities=0,remote_and_local_checksum_dry_run_identical=True,
        remote_file_inventory_sha256=sha(root/'remote-file-inventory.json'),transfer_manifest_sha256=sha(root/'transfer-manifest.json'),
        artifact_manifest_csv_sha256=sha(output/'artifact_manifest.csv'),bhist_evidence_sha256=sha(root/'private/bhist.txt'),
        rsync_log_sha256=sha(root/'private/rsync-01.log'),idempotence_evidence_sha256=sha(root/'private/rsync-idempotence.json'),
        source_preservation='Original remote artifacts were only read; collection and audit evidence are new local directories.',
        results_expected=252,results_collected=252,results_core_verified=252,dense_expected=18,dense_verified=18,
        jobs_done=55,jobs_failed=0,jobs_cancelled=0,jobs_running=0,jobs_missing_history=0)
    (output/'collection_receipt.json').write_text(json.dumps(receipt,indent=2))
    issues=[]
    descriptions={
        'TARGET_OUTSIDE_SYMMETRIC_BAND':('WARNING','96 observed target deviations under preserved controllers; do not treat them as equal-compression comparisons.'),
        'STATIC_NEAREST_FALLBACK':('WARNING','18 CiteSeer/30% static outcomes use the recorded nearest-width fallback.'),
        'DECLARED_INFEASIBLE_UNPRUNED':('WARNING','6 CiteSeer/30% Sequential Random outcomes retain the dense graph model; exact band is infeasible.'),
        'GREEDY_UNSELECTED_CANDIDATES_NOT_RECORDED':('WARNING','36 Greedy selected trajectories are legal and consistent, but unselected candidate values are unavailable; argmax cannot be independently verified without forbidden reruns.'),
        'REUSED_DENSE_ORIGINAL_TRAINING_WALL_NOT_SAVED':('WARNING','Original fitting wall is unavailable for six reused seed101 checkpoints (84 linked pruning rows); current-campaign zero fitting cost is not original training time.')}
    for code,count in totals['issue_counts'].items():
        severity,message=descriptions[code];issues.append(dict(code=code,severity=severity,affected_run_rows=count,message=message))
    for code,severity,count,message in [
        ('DENSE_INFERENCE_LATENCY_NOT_MEASURED','WARNING',18,'No standardized dense inference benchmark for these18 conditions. Preflight login timings are not a C4 dense baseline; do not calculate inference speedup.'),
        ('LATENCY_RAW_SAMPLES_NOT_SAVED','INFO',252,'Only recorded means/std and pass counts are available; no new timing was performed.'),
        ('RSS_IS_WORKER_HIGH_WATER','INFO',252,'Peak RSS is cumulative for the14-run worker and cannot be attributed as incremental per-method memory.'),
        ('STATIC_SALIENCY_AND_OPTIMIZATION_NOT_REEXECUTED','INFO',108,'Static feasibility, masks, projection and final metrics are verified; full saliency/width optimization was not rerun. Per-parameter SNIP scores were not saved.'),
        ('INTERMEDIATE_ACCURACIES_OBSERVED_NOT_REEVALUATED','INFO',144,'Recorded adaptive intermediate observations are audited arithmetically; only final frozen models are independently inferred.'),
        ('REUSED_ORIGIN_GRAPH_DIGEST_RECONSTRUCTED','WARNING',6,'The six20261024 origins lack a contemporaneous graph tensor hash. Preserved raw/processed lineage and original checkpoint bytes were verified; current campaign frozen graph hashes are complete.'),
        ('BACCT_QUERY_TIMEOUT_RESOLVED_WITH_BHIST','INFO',55,'The broad bacct query timed out; complete bhist accounting and successful stdout logs cover all55 IDs. Missing bjobs entries are scheduler cleanup, not missing experiments.')]:
        issues.append(dict(code=code,severity=severity,affected_count=count,message=message))
    unresolved=dict(blocking_failures=[],go_for_descriptive_analysis=True,issues=issues,
                    no_go_claims=['Greedy argmax independently proved','SNIP saliency independently reproduced','dense-to-pruned inference speedup','all methods matched at requested compression','policy episodes as independent trained GNNs'])
    (output/'unresolved_issues.json').write_text(json.dumps(unresolved,indent=2))
    active=sum(j['active_wall_seconds'] for j in jobs);pending=sum(j['queue_pending_seconds'] for j in jobs);cpu=sum(j['cpu_seconds'] for j in jobs)
    search=sum(r['search_seconds'] for r in runs);fitting=sum(d.get('training_seconds') or 0 for d in evidence['dense'])
    resources=dict(aggregate_active_lsf_seconds=active,sum_per_job_pending_seconds=pending,
                   cpu_accounting_seconds=cpu,actual_allocated_slot_hours=sum(j['allocated_slots']*j['active_wall_seconds']/3600 for j in jobs),
                   requested_four_slot_active_hours=active*4/3600,measured_search_seconds=search,
                   measured_new_dense_fitting_seconds=fitting,hard_requested_active_wall_minutes=845,
                   warning='Pending is a sum over55 jobs and includes dependency waits; it is not elapsed campaign duration.')
    (output/'resources.json').write_text(json.dumps(resources,indent=2))
    coverage=[]
    for dataset in ('cora','citeseer','pubmed'):
        for model in ('gcn','graphsage'):
            subset=[r for r in runs if r['dataset']==dataset and r['architecture']==model]
            coverage.append(f'| {dataset} | {model} |3|42|{sum(r["verified"] for r in subset)}|')
    method_rows=[]
    for method in dict.fromkeys(r['method'] for r in runs):
        cell=[r for r in runs if r['method']==method]
        method_rows.append(f'| `{method}` |36|36|{sum(r["physical_pruning"] for r in cell)}|{sum(r["outside_symmetric_band"] for r in cell)}|')
    collection=f'''# Collection report — POLUS complete pruning campaign

Collection/audit date:2026-10-31 (Europe/Moscow).

**Collected18/18 dense checkpoints and252/252 pruning condition artifact sets.**
All55 submitted jobs1611561–1611615 are **DONE** in retained LSF history and their
stdout logs report successful completion. FAILED:0; CANCELLED:0; RUNNING:0;
missing history:0. No completion claim is inferred from submission receipts alone.

## Campaign identity and immutable evidence

The authority is the explicit matrix inside executable snapshot
`{submission['source_commit']}`, tied to the actual deployment protocol and
55 original INTENT/ACCEPTED receipts. It is not reconstructed from directory names
or confused with V1–V4. Cora/CiteSeer/PubMed, GCN/GraphSAGE,3layers/hidden32,
dense seeds101/202/303,generated60/20/20,max500epochs/patience50,targets30/50%,
seven existing method/controller packages; no fine-tuning. Six dense seed101
checkpoints are verified historical native reuses;12 are new fits. The28-run
Cora101 pilot is part of252 and is counted once, not an extra primary sample.
Native synthetic preflight smoke is metadata, not a research observation.

- Source archive SHA256:`{submission['source_archive_sha256']}`.
- Deployment protocol SHA256:`{submission['protocol_sha256']}`.
- Original submission evidence SHA256:`{submission['receipt_archive']['sha256']}`.
- Frozen input manifest SHA256:`414fbd317890e12c3be85fbe3bc21b8e09326662a8b66b6de5305b8a9fab265f`.
- Full per-file hashes/sizes:[artifact_manifest.csv](artifact_manifest.csv).
- Collection receipt/hashes:[collection_receipt.json](collection_receipt.json).
- All55 status/accounting rows:[job_status.csv](job_status.csv).

## Transfer

**{files:,} expected available files →{files:,} collected →{files:,} SHA256/size verified**,
{nbytes:,} bytes ({nbytes/1024**2:.2f}MiB). All1,145 explicitly required output paths
are present. Missing:0; corrupted:0; unexpected:0; duplicate inventory paths:0;
duplicate result identities:0. Byte-identical file contents, including empty
stderr or identical final masks, are not counted as duplicate scientific runs.

Rsync used `--partial --checksum` and only read original remote files. A second
checksum **dry-run reported zero changes**, establishing idempotence. No remote
archive/result was created, overwritten or deleted. The717-file source snapshot
was reused from the independently hash-verified original local source archive.
Raw transfer inventory, rsync logs and private scheduler text remain external.

New local external root:
`${{GNN_ARTIFACT_ROOT}}/{root.name}/`:

- `raw/`: untouched collected campaign artifacts, including18 dense and252 pruned
  checkpoints, original frozen graph tensors/splits in checkpoints,252 reports,
  deployment/Q-learning/Lookahead records, training histories,55 stdout/stderr
  pairs, source/input hashes, configuration and submission metadata.
- `source/` and `source-{submission['source_commit']}.tar.gz`: pinned717-file source.
- `private/`: original status queries, full bhist text, transfer logs and audit logs;
  access-specific values stay here and are not committed.
- `remote-file-inventory.json`, `transfer-manifest.json`: original remote hashes
  and local verification outcomes.
- `audit-01/`, `audit-02/`, `audit-03/`, `tests/`: independently generated audit/test evidence.

Original remote root remains
`${{POLUS_TASK_ROOT}}/results/polus-complete-pruning-20261031/`.
No historical checkpoint/archive/report or SITITO manuscript was modified.

## Scheduler evidence and resources

`bjobs` had already expired these55 completed entries. A broad `bacct` request
exceeded its20-second bound; that failed query is preserved. `bhist -l` returned
complete records for all55 jobs, including success, CPU/memory and state duration;
all55 scheduler stdout logs corroborate success. No accounting data is missing.

Aggregate active LSF wall:**{active:.0f}s ({active/60:.2f}min)**, below the845-minute
hard reservation. CPU accounting:{cpu:.1f}s. Exclusive C4 actually allocated
**20 slots per job**, although Python used/requested4 threads/slots: actual
allocated slot-hours are{resources['actual_allocated_slot_hours']:.3f}, compared
with{resources['requested_four_slot_active_hours']:.3f} four-slot active hours.
Do not report the latter as actual exclusive allocation. Peak sampled LSF memory
is715MB; per-job values/units are preserved in the accounting table.

Sum of per-job PEND durations:{pending:.0f}s ({pending/3600:.3f}h), including dependency
waits. This sum is **not elapsed campaign duration** and is not active computation.
Native recorded pruning search totals:{search:.6f}s; twelve new dense fits:
{fitting:.6f}s. The original fitting wall for six reused checkpoints is unavailable;
current-campaign reuse is zero fitting, not a zero-cost original training run.
These components do not sum to scheduler wall because import, checkpoint/data
I/O, final evaluation/inference timing and launcher work are additional.

No bsub, research-model training, scientific pruning search, environment installation or network/credential
change was performed during collection. Read-only inference was local. See the
[independent audit report](AUDIT_REPORT.md) before scientific use.
'''
    (output/'COLLECTION_REPORT.md').write_text(collection)
    audit=f'''# Independent audit — verified POLUS research dataset

**GO for descriptive scientific analysis**, subject to the explicit limits below.
Expected/collected/core-verified pruning conditions:**252/252/252**; dense models:
**18/18**. Per-run audit statuses:**92 PASS,160 WARNING,0 FAIL,0 NOT VERIFIED**.
Every WARNING retains verified identity, file integrity, physical tensors, final
metrics and applicable recorded-trace checks. It is not discarded from summaries.
This is not a claim of complete counterfactual/policy-optimality verification.

## Coverage and unit of replication

| Dataset | Model | Independent dense seeds | Expected runs | Core verified |
|---|---|---:|---:|---:|
'''+ '\n'.join(coverage)+f'''

| Method | Expected | Verified | Physically pruned | Outside symmetric band |
|---|---:|---:|---:|---:|
'''+ '\n'.join(method_rows)+f'''

The statistical unit is the independently trained dense checkpoint,**n=3 per
dataset/model/method/target**. Seed101 is one reused native training checkpoint,
not an additional platform replicate. Policy seeds101/202/303 are paired with
the dense seed for stochastic methods and are not separately varied. The1,080
Q episodes are not1,080 independent trained GNNs. No V1–V4, local CPU, Questions,
Flickr, OGB or unrelated course results enter the primary tables.

## Independent checks

The audit reads the **explicit submitted matrix**, verifies all55 acceptance IDs,
responses, launcher hashes/dependencies/resources against the pre-collection
committed receipts, and checks the deployment protocol/source archive hashes.
All717 pinned source files match; the five local model-code files used for
inference are byte-identical to their submitted versions.

All18 dense checkpoints: strict reconstruction, finite tensors, architecture and
seed, original saved split, graph/checkpoint/tensor hashes, disjoint60/20/20 node
coverage, recorded metrics. Twelve new training histories also pass epoch order,
validation-loss best-epoch selection, patience50 and time-accounting checks.
All six origins retain their separately identified20261024 source and input proof.
All six dataset/model groups have three distinct saved split hashes and model
tensor identities, consistent with the three independently trained conditions.

All252 final checkpoints: strict reconstruction and finite forward output with
correct node/class dimensions. Physical parameter counts and memory/serialized
sizes recompute. Direct tensor indexing, independent of production Local Surgery,
projects the dense retained rows/columns/biases onto every final state_dict;
all tensors match exactly, including output-layer parameters. Thus246 models are
physically smaller and six explicitly infeasible outcomes equal their dense
models. Retained original channel identities, not merely final widths, are checked.

Final validation/test accuracy and macro-F1 are recomputed using **original frozen
graph tensors and splits read from checkpoints**; no split generator or dataset
loader is called. Existing local CPU Torch2.2.2/PyG2.6.1,Python3.12.9,two threads
verifies native POWER8 Torch2.0.1/PyG2.4.0,four-thread outputs. Final metric absolute
tolerance is1e-12; maximum observed difference is2.220446049250313e-16. Integer
parameter counts and tensor projections are exact; compression tolerance1e-12.
Recorded intermediate accuracies are float32 means, so reward/drop arithmetic
allows1.2e-7 absolute error, independently chosen for float32 quantization. Q
updates and recorded Lookahead returns use stored rewards with1e-12 tolerance.

No training, saliency search, pruning rollout, policy evaluation experiment,
new timing measurement or scheduler job was executed. Only the270 final frozen
models were independently inferred per complete audit pass. Three separate audit
passes are preserved; the second adds submission anchors, resource and training
checks; the third also verifies static criterion identities and stop-reason metadata. No observation was repaired or replaced.

## Method-specific verification and limits

- **Static methods:** retained IDs, minimum widths, physical counts, target/fallback
  semantics, frozen projection and final metrics checked. SNIP means post-training
  train-node abs(weight×gradient). Full saliency/width optimization is not rerun;
  per-parameter SNIP scores were not saved. A static mask with the same width but
  different original IDs fails projection verification.
- **Sequential Random:** each stored action is legal under an independent exact
  width/horizon reachability calculation; seeded uniform legal-action choices,
  magnitude channel deletion, rewards and terminal reasons match. Six
  CiteSeer/30% conditions are genuinely infeasible in the exact band and retain
  dense widths32/32. They are valid recorded failures to prune at that target,
  not successful30% compression and not missing/corrupted experiments.
- **Greedy:** legal actions, magnitude IDs, chosen rewards, counts and native
  threshold/max-step/no-action stopping match. **Unselected candidate observations
  were not saved for36 runs**, so the Greedy argmax cannot be independently
  re-established without a forbidden rerun. Source and preflight evidence support
  the executed implementation; the subcheck remains NOT VERIFIED, not fabricated.
- **Lookahead:** all36 recorded depth-2 trees,4,175 candidate leaf paths, first-action
  execution/replanning, immediate+.9×next reward, terminal handling, stable ties,
  retained IDs, reuse and actual evaluation counts pass. This checks recorded
  counterfactual scores; their intermediate accuracies are not newly inferred.
- **Q:**30 learning episodes per run reset dense; deployment resets again and the
  table is frozen. The17,930 stored training transitions independently reproduce
  seeded exploration, legal actions, rewards, Bellman updates, all13,474 final Q
  entries, coverage and deployment argmax/termination. No episode or rollout is
  newly generated. Aggregate states omit retained masks and do not establish an
  MDP guarantee. These are **native V2 soft-target Q**, not constrained V4 Q.

Total recorded validation-oracle calls:**24,698**, independently reconciled.
Candidate enumeration, oracle calls and full scoring/training forward/backward
work are different quantities: static SNIP has zero validation-oracle calls but
nonzero scoring/search cost; Q candidate enumeration is not applicable/missing,
while its chosen-transition oracle calls are measured.

## Test-label isolation and comparability

Pinned executed source masks test labels before scoring/search; SNIP receives
train indices, and adaptive methods receive only the validation callback. Test
metrics follow completed decisions. Frozen labels also serve graph identity and
known class-count metadata; that is distinct from test-performance feedback.
The audit is based on source/path/launcher hashes and preserved native preflight
isolation assertions, not a retroactive runtime taint trace. No test metric changes
the protocol, method choice, report inclusion or plot selection.

Controllers differ by design. Greedy/audited Lookahead use lower-threshold
stopping with overshoot; static methods use the symmetric band/nearest fallback;
Sequential Random enforces exact reachability; native V2 Q permits STOP and does
not stop merely on crossing a target. **96/252 final models fall outside the
symmetric±.01 band**:18 static fallbacks,6 infeasible Random,18 Greedy,18 Lookahead,
36 Q. These warnings are preserved. Requested30/50% does not imply equal achieved
compression, widths or masks. This dataset supports comparisons of existing
method/controller packages, not an isolated causal effect of planning/learning.

## Missing measurements and unresolved issues

Critical failures/blockers:0. All per-run warning codes and counts are in
[audit_runs.csv](audit_runs.csv); severities and limitations are in
[unresolved_issues.json](unresolved_issues.json).

| Severity | Evidence limitation | Scope / consequence |
|---|---|---|
| WARNING | Greedy unselected candidate values absent |36 trajectories: argmax not independently verified |
| WARNING | Original fit wall absent for reused dense models |6 models/84 linked rows: missing, not0 |
| WARNING | Standardized dense inference latency absent |18 models: no dense-to-pruned measured speedup |
| WARNING |20261024 origin graph digest not contemporaneous |6 origins: preserved raw/processed lineage reconstructed; current frozen hashes complete |
| INFO | Raw inference latency samples absent |252 mean/std/pass-count records: consistency checked, no new timing |
| INFO | Worker RSS is cumulative high-water |252 rows across18 workers: not incremental per-method memory |
| INFO | Static saliency/optimization and intermediate forwards not rerun |Clearly separated from verified masks/recorded arithmetic/final outputs |

All target deviations and infeasible/fallback conditions remain in canonical
results. `summary.csv` includes n_expected,n_collected,n_verified,n_fail,n_warning,
physical/target-band counts and stop reasons for all84 dataset/model/method/target
cells; means/sample SD use the three independent dense seeds. Missing quantities
remain empty, never imputed. Dense training time repeated in per-run provenance
must not be summed fourteen times; unique dense records live in `dense_audit.csv`.

Search/pruning wall is separate from measured inference mean/std (5 warmup,20
timed passes). Timing verification checks recorded units, nonnegative values,
pass counts and scheduler/resource consistency; it does not reproduce latency.
Actual exclusive20-slot allocation and active/waiting/CPU time are separated in
[the collection report](COLLECTION_REPORT.md) and [resources.json](resources.json).

## Tables, figures, reproducibility and tests

- [Canonical per-run dataset](verified_results.csv):252 rows, including all warnings,
  dense/pruned validation/test accuracy/F1 and differences, exact compression,
  dimensions/original IDs, parameter/checkpoint sizes, measured costs, audit and
  provenance. Architecture is GCN/GraphSAGE; all dense models are3layers,width32,
  dropout.5, as pinned in the protocol. Six infeasible rows remain explicit.
- [Summary](summary.csv):84 cells; [dense verification](dense_audit.csv):18;
  [job accounting](job_status.csv):55; [machine audit](audit.json):full compact scopes.
- Figures:[accuracy/compression](figures/accuracy_compression.png),
  [macro-F1/compression](figures/f1_compression.png),
  [quality/search cost](figures/quality_search_cost.png),
  [coverage/physical pruning](figures/coverage.png). All use validation metrics;
  no scientific winner, test-based selection or tuning is proposed. Overlapping
  plotted points do not prove identical masks; the CSV keeps original IDs.

Focused pytest:**38 PASS/0 FAIL/0 SKIPPED**. Full pytest:**585 PASS/0 FAIL/6 SKIPPED**.
Skips:2 CUDA device checks (no CUDA),4 opt-in archived-Cora Lookahead checks not
enabled in this invocation. They are not reported as passes. Adversarial tests
cover missing/corrupt/duplicated/inconsistent outputs, altered metrics/rewards/Q
values, terminal tree extensions, false counts, equal-width different masks,
source/seed/leakage metadata, missing essential evidence and accounting status.
[Test log and audit-code hashes](tests.json) identify the exact evidence.

Rerun from this canonical checkout, using the existing ML environment and fresh
output directory (never overwrite raw evidence or earlier audits):

```bash
export PYTHONPATH="$PWD/src"
export COLLECTION="$GNN_ARTIFACT_ROOT/{root.name}"
python -m gnn_pruning.experiments.complete_audit \\
  --root "$COLLECTION/raw" --source "$COLLECTION/source" \\
  --inventory "$COLLECTION/remote-file-inventory.json" \\
  --accounting "$COLLECTION/private/bhist.txt" \\
  --output "$COLLECTION/audit-rerun-NEW"
python -m pytest tests/test_complete_campaign_audit.py -ra
```

The preserved committed pre-collection submission reference is checked by the
script. It never calls a split generator, dataset downloader, training/pruning
runner or bsub. Further report/plot generation only reads audited JSON/CSV.

**GO:** analyze this verified native dataset descriptively, with achieved
compression, paired dense seeds and the recorded missingness/controllers.
**NO-GO:** infer a universally superior strategy, treat requested targets as
matched physical models, claim independently proved Greedy argmax, or compute
unmeasured dense inference speedups. No additional experiment is authorized here.
'''
    (output/'AUDIT_REPORT.md').write_text(audit)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();report(a.root,a.output)

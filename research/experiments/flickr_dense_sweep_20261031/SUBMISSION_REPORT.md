# Flickr Dense sweep — accepted by LSF, not a completed-results report

**32 research jobs submitted /32 ACCEPTED /0 blocked /0 rejected /0 unresolved.**
Research IDs **1611757–1611788**. Separately, three native technical jobs
**1611753/1611754/1611755** completed DONE; their18 short runs are not research
results. There were35 accepted jobs across both stages and no ambiguous retries.

Research RUNNING/COMPLETED states were **not queried after acceptance**. No research
checkpoint, metric or learning curve was collected or interpreted. Jobs run under
LSF independently of this session. RESULTS.csv, ANALYSIS.md and scientific figures
will be generated only after the documented completion/integrity gates.

## Exact identities and artifacts

| Model/depth | Width32 | Width64 | Width128 | Width256 |
|---|---:|---:|---:|---:|
| GCN2 |1611757|1611758|1611759|1611760|
| GCN3 |1611761|1611762|1611763|1611764|
| GCN4 |1611765|1611766|1611767|1611768|
| GCN5 |1611769|1611770|1611771|1611772|
| GraphSAGE2 |1611773|1611774|1611775|1611776|
| GraphSAGE3 |1611777|1611778|1611779|1611780|
| GraphSAGE4 |1611781|1611782|1611783|1611784|
| GraphSAGE5 |1611785|1611786|1611787|1611788|

[JOB_MANIFEST.csv](JOB_MANIFEST.csv) records each identity, acceptance, expected
resolved-config hash, split, planned resources, log/result paths and no dependencies.
Actual execution-host fields are intentionally blank until execution evidence is
collected. [SUBMISSION_RECEIPTS.json](SUBMISSION_RECEIPTS.json) pins each original
INTENT/response/receipt/launcher; all32 return codes and exact acceptance/job IDs
were independently rechecked from the downloaded metadata archive.

Raw local preparation/technical/acceptance evidence:
`${GNN_ARTIFACT_ROOT}/flickr-dense-sweep-20261031/`.
Remote campaign: `${POLUS_TASK_ROOT}/results/flickr-dense-sweep-20261031/`.
Remote stdout/stderr: `logs/<condition>.<job_id>.out` / `.err`.
Remote final/partial run artifacts: `runs/<condition>/`.
Access/account values remain in external `access-private.json`, not tracked files.

## Common scientific protocol

Full authorized32: Flickr × GCN/GraphSAGE × layers2/3/4/5 × widths32/64/128/256.
Seed101; common frozen generated60/20/20 split, seed101; Adam LR**0.001**, weight
decay0.0005, dropout0.5; maximum500 epochs, validation-loss patience50 and best-loss
checkpoint restoration. No normalization, pruning, fine-tuning, checkpoint reuse,
architecture-specific LR or additional seeds.

LR0.001 was predeclared from the historical optimization concern. Both models'
LR0.01/0.001 technical traces were recorded separately, not used to select research
architectures by short-run accuracy. All32 conditions remain in the submitted grid.

Test labels were masked to−1 **before upload**. The held-out labels stay quarantined
locally. No test evaluation code is invoked by training, preflight or analysis.
This is full-batch transductive classification: all-node features/edges remain;
it is not the original inductive/official-split GraphSAINT benchmark. Final train
and validation accuracy/F1/loss describe the restored checkpoint; dropout-on
training-loss curves describe successive training iterates.

Training executable source: **3995c6057f3460005b44bc57d98c6d5e7df7f9e2**.
Scientific source snapshot remains immutable at
`${POLUS_TASK_ROOT}/source-snapshots/flickr-dense-3995c6057f3460005b44bc57d98c6d5e7df7f9e2/`.
Scheduling clarification is separately pinned at orchestration commit
**fd998538237c2fcc7bc96761d4e12ea4246333d7**. The versioned submission/capacity tools
live beside the campaign, with hashes in every research receipt/capacity proof;
the training runner, models, trainer and seeding files are byte-identical to the
native preflight snapshot. No technical runs were repeated merely to update
submission policy. [PROVENANCE.json](PROVENANCE.json) records both revisions.

## Native resource gate and scheduling

Verified normal queue: Open:Active,180-minute absolute limit, EXCLUSIVE allowed,
compute membership C1–C4. C1 unavailable/MAX1; C2/C3/C4 available/MAX20. C2/C3 have
256GiB nominal RAM; C4 has1TiB. LSF memory-limit units are MB, ABS_RUNLIMIT=Y,
default pending capacity300; no unfinished user jobs immediately before research
submission. Queue/user JL limits were not finite in current displayed policy.
Three technical jobs actually ran simultaneously; future concurrency is an LSF
decision, not guaranteed from node count.

The researcher clarified: **large jobs on C4; others no host binding**. Twelve
conditions (all width256 plus4/5-layer width128) specify C4. Twenty omit `-m`
entirely. Every job requests4 slots/threads, one host, exclusive allocation and
180minutes; memory limits/reservations are12,288MB GraphSAGE or13,312MB GCN.
There are no research-job dependency edges. Exclusive scheduling withholds a whole
node: native history shows20 allocated slots despite4 requested tasks. At most
one of these exclusive jobs fits each eligible compute node; other users/fairshare
can reduce concurrency.

LSF reports CPU factor250 for all three, and /proc/sysfs reported4023MHz for all.
These observations do **not** verify a reduced-frequency cause. No such label was
invented. Actual timing differs, is preserved per host, and is never pooled into
a hardware-independent architecture timing ranking.

The technical stage used12epochs for each2/64 LR trial and6epochs for each5/256
maximum-size trial, one fresh process per trial. **18/18** complete output inventories
hash-verify, strict checkpoint reload/metrics pass and losses are finite. Actual
maximum-size measurements:

| Host | GCN5/256 median s/epoch | GCN peak RSS GiB | SAGE5/256 median s/epoch | SAGE peak RSS GiB |
|---|---:|---:|---:|---:|
| C2 |9.974|3.512|11.768|3.039|
| C3 |11.540|3.526|13.843|3.042|
| C4 |6.876|3.530|8.151|3.038|

All small/LR measurements, observed epoch maxima and frequencies are preserved in
[NATIVE_PREFLIGHT.json](NATIVE_PREFLIGHT.json). Native software: Python3.10.22,
Torch2.0.1, PyG2.4.0; the existing activation script hash is unchanged. No package,
system environment, network route or credential was modified.

Resources use native small/maximum timings, affine work interpolation preserving
fixed overhead and exact measured maximum,1.75 wall margin plus120seconds. Each
unbound condition fits a full500-epoch budget on all three measured hosts; large
conditions fit C4. The largest safety projection is**122.67minutes**, below the
175-minute acceptance threshold/180-minute job cap. No epoch budget was shortened.
Memory requests use3× observed process RSS+2GiB, rounded upward.

All500-epoch aggregate fitting projection is**11.45–13.75 active job-hours**,
depending on where unbound jobs run; early stopping can reduce it. This is not
elapsed campaign time or a completion guarantee. Queue waiting is additional.
Hard aggregate wall reservation is**96 job-hours** (32×180min),384 requested
slot-hours; exclusive20-slot occupancy can withhold up to1,920 slot-hours at that
cap. No current scientific cost/quality outcome is claimed.

## Tests and remaining limits

| Verification | PASS | FAIL | SKIPPED |
|---|---:|---:|---:|
| Final focused orchestration/Dense tests |26|0|0|
| Full suite after scheduling change |630|0|6|
| Native technical Dense/reload outputs |18|0|0|

Six skips:2 CUDA-unavailable checks and4 opt-in archived-Cora Lookahead gates.
They are not counted as passes. Local test runtime: Python3.12.9/Torch2.2.2,
existing plotting dependencies and MPLBACKEND=Agg. Initial preparation logs retain
missing-Matplotlib, graphical-backend abort and an overly broad import-boundary
test assertion; the test was corrected, the existing plotting dependency path and
headless backend were used, and final checks pass. No historical algorithm was
changed to obtain a pass. One transfer attempt had a nonexistent package path;
it created no scheduler job and was corrected before submission.

One seed establishes screening, not independent-seed confirmation. Patience does
not prove convergence/generalization. Depth2/4/5 remain incompatible with current
three-layer pruning controllers;4/256 and5/256 also cannot reach50% compression
within the existing20-action menu. This Dense campaign does not change those rules.

No research condition was skipped. Seeds202/303, test evaluation, pruning,
manuscript changes, push and merge were not started.

## Later status, collection and analysis

The following commands are for a later session; they were **not** run after
acceptance. Read the approved login and actual remote root from the private local
access file, avoiding the unrelated default SSH account:

```sh
FLICKR_PYTHON=/private/tmp/gnn-v2-validation-py312-20261027/bin/python
FLICKR_LOCAL_ROOT="$GNN_ARTIFACT_ROOT/flickr-dense-sweep-20261031"
POLUS_LOGIN=$("$FLICKR_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["login"])' "$FLICKR_LOCAL_ROOT/access-private.json")
POLUS_FLICKR_ROOT=$("$FLICKR_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["root"])' "$FLICKR_LOCAL_ROOT/access-private.json")
ssh "$POLUS_LOGIN" 'bjobs -a -w $(seq 1611757 1611788)'
ssh "$POLUS_LOGIN" 'bpeek 1611788'
ssh "$POLUS_LOGIN" 'bhist -l 1611788'
```

After all jobs finish, choose fresh local collection/analysis directories:

```sh
COLLECTION="$GNN_ARTIFACT_ROOT/flickr-dense-collection-NEW-TIMESTAMP"
"$FLICKR_PYTHON" research/experiments/flickr_dense_sweep_20261031/scripts/collect.py \
  --login "$POLUS_LOGIN" --remote-root "$POLUS_FLICKR_ROOT" --output "$COLLECTION"
MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="src:$GNN_ARTIFACT_ROOT/lookahead-implementation-20261031/test-deps" \
"$FLICKR_PYTHON" -m gnn_pruning.experiments.flickr_analysis \
  --root "$COLLECTION/raw" --protocol "$COLLECTION/raw/protocol.json" \
  --scheduler "$COLLECTION/scheduler.json" --output "$COLLECTION/analysis" --figures
```

Collector fails closed on missing scheduler success; analyzer fails closed on
missing/invalid32 results and writes completeness evidence instead of ranking.
Inspect partial artifacts on failure; do not silently resubmit accepted/ambiguous
claims. Finalize validation-only architecture selection before any later test
evaluation. Additional seed runs are recommendations only, never auto-launched.

# Flickr Dense: collected and verified, 1 November 2026

**32/32 COMPLETED and audited PASS; 0 RUNNING, missing or failed.** LSF jobs
1611757–1611788 all have archived `Done successfully.` evidence. Current `bjobs`
no longer lists these historical jobs; completion is established from `bhist`,
original acceptances, execution job identities and completed output inventories.
No job was resubmitted and no new model was trained or pruned.

## Collection and independent verification

The new collection is separate from preparation and historical experiments:
`${GNN_ARTIFACT_ROOT}/flickr-dense-collection-20261101T140723Z/`.
Its `results.tar.gz` is **105,074,830 bytes**; its SHA256 was checked against the
server before extraction:
`a1a54f9e72b28f311cc3b350bd391e0f1c00db6a88106a1b290c1087c921f74c`.
The machine-readable byte count in [PROVENANCE.json](PROVENANCE.json) is canonical.
Raw logs, checkpoints, inputs, source archives and scheduler histories stay outside
Git. The server campaign remains at
`${POLUS_TASK_ROOT}/results/flickr-dense-sweep-20261031/`.
Access/account values remain private and external.

[AUDIT.json](AUDIT.json), [per-run checks](audit_runs.csv),
[raw file hashes](RAW_FILES.json) and [metric differences](METRIC_RELOAD_DELTAS.json)
record the following checks:

- All 481 collected files inventoried: 286,813,300 uncompressed bytes; all original
  per-run output hashes match. All 794 source archive files match the pinned
  scientific source `3995c6057f3460005b44bc57d98c6d5e7df7f9e2`.
- Original protocol, input manifest, graph bytes/tensors and deterministic disjoint
  split match their pins. Test-node labels remain −1; quarantined labels were not
  loaded or transferred by collection/verification.
- All 32 final checkpoints strictly reload for their declared architectures;
  model parameter counts match an independent layer-size formula. Each final
  tensor state matches its preserved best partial checkpoint. Configuration
  hashes match the original submission manifest; no architecture reuse occurred.
- Finite, sequential epoch histories agree with saved epoch counts, the first
  minimum validation-loss epoch and patience/epoch-limit stopping. Final train
  and validation metrics were independently recomputed from frozen checkpoints,
  using real forward passes and no optimization. Class/prediction counts match.
  Maximum absolute differences: accuracy **0**, macro-F1 **5.55e−17**, loss
  **3.58e−7**. Predeclared audit tolerances are 1e−12 for accuracy/F1 and 5e−5
  for float32 loss across the native POWER8 and local x86 software stacks.
- All job IDs, execution hosts, four requested tasks/threads and 20 allocated
  exclusive slots agree with archived scheduler evidence. All 12 bound large
  jobs ran on C4. Actual host counts: C2 **10**, C3 **7**, C4 **15**.

These checks establish collection/model/metric consistency, not independent-seed
confirmation or convergence. No test-label evaluation or inference timing
benchmark was performed.

## Recorded results and cost

[RESULTS.csv](RESULTS.csv) contains all 32 train/validation results, model sizes,
checkpoint identities, epochs, stopping, actual hosts, process RSS and measured
fit times. [ANALYSIS.md](ANALYSIS.md) is the existing analysis pipeline's basic
validation-only summary; saved loss curves and quality/size/cost figures are in
[FIGURES](FIGURES/). Cost figures separate actual execution hosts.

Twenty-six conditions reached the 500-epoch cap; six stopped by patience50.
These operational stopping events do not establish convergence. The same single
training/split seed101 is used throughout. Saved provisional recommendations in
[FINALIST_RECOMMENDATIONS.json](FINALIST_RECOMMENDATIONS.json) follow the already
implemented validation macro-F1 → validation accuracy → parameter-count rule;
they are not a final architecture choice or authorization for more runs.
Depth2/4/5 still require qualification of the unchanged three-layer pruning
controllers before any subsequent pruning campaign.

Aggregate measured fitting time is **13.982 job-hours**; scheduler RUN time is
**14.176 job-hours**, CPU time **44.313 CPU-hours**. Summed PEND time is
**66.063 job-hours**; concurrent queue waiting is not elapsed campaign duration.
Process peak RSS is at most **3.707 GiB**. Scheduler rounded peak-memory strings
are retained separately in `audit_runs.csv`. Hardware-dependent times are not
used as a global architecture speed ranking.

## Verification results and reproduction

| Check | PASS | FAIL | SKIPPED |
|---|---:|---:|---:|
| Collected run audits, including real forward/reload checks |32|0|0|
| Focused existing Flickr pytest |26|0|0|

The full suite was not repeated for this collection-only change. Its earlier
submission-stage result was 630 PASS /0 FAIL /6 SKIPPED, not a new test result.
The initial current-session pytest invocation could not start because the old
temporary environment had been partially cleaned (`No module named pytest`).
Offline restoration attempts also recorded missing wheel dependencies. A new
isolated local environment was created using saved packages plus missing public
dependencies; no server/system environment was changed. The successful test log,
failed environment attempts and audit logs remain external with hashes in
`PROVENANCE.json`. [verification-environment.txt](verification-environment.txt)
pins the successful local package set.

From the canonical repository, with `COLLECTION` pointing to the external group
above and `PYTHON` to an environment with the recorded packages:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON" \
  research/experiments/flickr_dense_sweep_20261031/collection_20261101/verify.py \
  --collection "$COLLECTION" \
  --pins research/experiments/flickr_dense_sweep_20261031/PROVENANCE.json \
  --output "$COLLECTION/audit-NEW"
MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON" \
  -m gnn_pruning.experiments.flickr_analysis \
  --root "$COLLECTION/raw" --protocol "$COLLECTION/raw/protocol.json" \
  --scheduler "$COLLECTION/scheduler.json" --output "$COLLECTION/analysis-NEW" --figures
MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON" \
  -m pytest tests/test_flickr_dense_sweep.py -q
```

Use fresh output directories: existing evidence is never overwritten. The audit
does not invoke fitting, pruning or held-out-label access. Original submission
reports/manifests and their hashes remain unchanged; completion evidence is added
here. Historical results, algorithms and the manuscript are untouched.

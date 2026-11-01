# Inspect, collect and verify V5

Access values are exclusively in the ignored external file
`${GNN_ARTIFACT_ROOT}/polus-v5-main-20261102/execution-access-private.json`.
Use its approved `login`, `root`, `source`, `taskroot`; never a historical default
SSH alias or a guessed directory. No route/environment modification is needed.

On POLUS, after loading those values into your shell (not committed):

```bash
bjobs -w 1612108 1612109
bhist -l 1612108 1612109
cat "${V5_ROOT}/acceptance-release.json"
cat "${V5_ROOT}/release/completed.json"
cat "${V5_ROOT}/release/blocked.json"
cat "${V5_ROOT}/acceptance-research.json"
cat "${V5_ROOT}/acceptance-verify.json"
cat "${V5_ROOT}/verification/completed.json"
cat "${V5_ROOT}/verification/blocked.json"
```

A missing optional status file means that phase has not written it yet, not that
it passed. For production IDs use acceptance-research.json, then `bjobs -w <IDs>`
or `bhist -l <IDs>` if bjobs history expired. Logs are under `${V5_ROOT}/logs`;
INTENT/response/receipt triples under `submissions`. `release/blocked.json` and
`verification/blocked.json` require investigation; **never rerun a submitter**
without reconciling every existing scheduler ID and immutable claim. No production
condition is implied by acceptance of the preflight/release jobs.

Local completed collection (use the existing verification Python, no installation):

```bash
PYTHONPATH=src python -m gnn_pruning.experiments.v5_campaign_collect \
  --access "${GNN_ARTIFACT_ROOT}/polus-v5-main-20261102/execution-access-private.json" \
  --destination "${GNN_ARTIFACT_ROOT}/polus-v5-main-collection-<NEW_TIMESTAMP>"
```

The collector first verifies the reporting job's retained LSF success and the
252-condition VERIFIED marker, queries relative remote hashes, transfers into a
new raw group and verifies every file SHA/size. It never submits jobs, changes
remote evidence, overwrites a verified collection or uses test labels.
For a completed technical-only collection use `--stage preflight`; this requires
both preflight LSF jobs DONE and does not imply a production launch.

Independent local recheck/analysis after a successful collection:

```bash
PYTHONPATH=src MPLBACKEND=Agg OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  python -m gnn_pruning.experiments.v5_campaign_audit \
  --root "${COLLECTION}/raw" --output "${COLLECTION}/independent-AUDIT.json"
PYTHONPATH=src MPLBACKEND=Agg python -m gnn_pruning.experiments.v5_campaign_analysis \
  --root "${COLLECTION}/raw" --audit "${COLLECTION}/independent-AUDIT.json" \
  --output "${COLLECTION}/independent-analysis"
```

Keep raw checkpoints/traces/logs, source archives and large audit JSON outside
Git. Only compact verified reports/tables/figures/provenance may later be copied
into this research directory. No push or merge is authorized by this task.

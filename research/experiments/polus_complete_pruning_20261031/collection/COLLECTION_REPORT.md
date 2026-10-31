# Collection report — POLUS complete pruning campaign

Collection/audit date:2026-10-31 (Europe/Moscow).

**Collected18/18 dense checkpoints and252/252 pruning condition artifact sets.**
All55 submitted jobs1611561–1611615 are **DONE** in retained LSF history and their
stdout logs report successful completion. FAILED:0; CANCELLED:0; RUNNING:0;
missing history:0. No completion claim is inferred from submission receipts alone.

## Campaign identity and immutable evidence

The authority is the explicit matrix inside executable snapshot
`55d0fc242b543c5725276144bd8e970eedc48ecd`, tied to the actual deployment protocol and
55 original INTENT/ACCEPTED receipts. It is not reconstructed from directory names
or confused with V1–V4. Cora/CiteSeer/PubMed, GCN/GraphSAGE,3layers/hidden32,
dense seeds101/202/303,generated60/20/20,max500epochs/patience50,targets30/50%,
seven existing method/controller packages; no fine-tuning. Six dense seed101
checkpoints are verified historical native reuses;12 are new fits. The28-run
Cora101 pilot is part of252 and is counted once, not an extra primary sample.
Native synthetic preflight smoke is metadata, not a research observation.

- Source archive SHA256:`7ef42db5eeddd3403157c7a2cb534c990bffd09bafe315a19614f6586aff59b6`.
- Deployment protocol SHA256:`5e400a59c27fe98506f16eb0055098de780edd26f0b209695be3a397f1edf256`.
- Original submission evidence SHA256:`42171aac9656c2b4225150842068cec7e643d396c111d366e0aebc2252ecec94`.
- Frozen input manifest SHA256:`414fbd317890e12c3be85fbe3bc21b8e09326662a8b66b6de5305b8a9fab265f`.
- Full per-file hashes/sizes:[artifact_manifest.csv](artifact_manifest.csv).
- Collection receipt/hashes:[collection_receipt.json](collection_receipt.json).
- All55 status/accounting rows:[job_status.csv](job_status.csv).

## Transfer

**1,226 expected available files →1,226 collected →1,226 SHA256/size verified**,
213,152,398 bytes (203.28MiB). All1,145 explicitly required output paths
are present. Missing:0; corrupted:0; unexpected:0; duplicate inventory paths:0;
duplicate result identities:0. Byte-identical file contents, including empty
stderr or identical final masks, are not counted as duplicate scientific runs.

Rsync used `--partial --checksum` and only read original remote files. A second
checksum **dry-run reported zero changes**, establishing idempotence. No remote
archive/result was created, overwritten or deleted. The717-file source snapshot
was reused from the independently hash-verified original local source archive.
Raw transfer inventory, rsync logs and private scheduler text remain external.

New local external root:
`${GNN_ARTIFACT_ROOT}/polus-complete-collection-20261031T110939Z/`:

- `raw/`: untouched collected campaign artifacts, including18 dense and252 pruned
  checkpoints, original frozen graph tensors/splits in checkpoints,252 reports,
  deployment/Q-learning/Lookahead records, training histories,55 stdout/stderr
  pairs, source/input hashes, configuration and submission metadata.
- `source/` and `source-55d0fc242b543c5725276144bd8e970eedc48ecd.tar.gz`: pinned717-file source.
- `private/`: original status queries, full bhist text, transfer logs and audit logs;
  access-specific values stay here and are not committed.
- `remote-file-inventory.json`, `transfer-manifest.json`: original remote hashes
  and local verification outcomes.
- `audit-01/`, `audit-02/`, `audit-03/`, `tests/`: independently generated audit/test evidence.

Original remote root remains
`${POLUS_TASK_ROOT}/results/polus-complete-pruning-20261031/`.
No historical checkpoint/archive/report or SITITO manuscript was modified.

## Scheduler evidence and resources

`bjobs` had already expired these55 completed entries. A broad `bacct` request
exceeded its20-second bound; that failed query is preserved. `bhist -l` returned
complete records for all55 jobs, including success, CPU/memory and state duration;
all55 scheduler stdout logs corroborate success. No accounting data is missing.

Aggregate active LSF wall:**1925s (32.08min)**, below the845-minute
hard reservation. CPU accounting:6960.2s. Exclusive C4 actually allocated
**20 slots per job**, although Python used/requested4 threads/slots: actual
allocated slot-hours are10.694, compared
with2.139 four-slot active hours.
Do not report the latter as actual exclusive allocation. Peak sampled LSF memory
is715MB; per-job values/units are preserved in the accounting table.

Sum of per-job PEND durations:39296s (10.916h), including dependency
waits. This sum is **not elapsed campaign duration** and is not active computation.
Native recorded pruning search totals:952.762614s; twelve new dense fits:
191.522374s. The original fitting wall for six reused checkpoints is unavailable;
current-campaign reuse is zero fitting, not a zero-cost original training run.
These components do not sum to scheduler wall because import, checkpoint/data
I/O, final evaluation/inference timing and launcher work are additional.

No bsub, research-model training, scientific pruning search, environment installation or network/credential
change was performed during collection. Read-only inference was local. See the
[independent audit report](AUDIT_REPORT.md) before scientific use.

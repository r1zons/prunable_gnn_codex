# Experiment campaign registry

**POLUS-V4-DIAGNOSTIC-20261029** — completed;92/92 results independently verified
and analysed (13,200 episodes/137,000 transitions/44,474 Q entries/300 snapshots).
All1,285 source inventory files unchanged;10 audit tests pass. [Verified report,
tables and four figures](polus_v4_diagnostic_20261029/results/REPORT.md).
Menu12 Q ties matched Greedy quality in12/12 cases at44–56x search cost. Other
factors have heterogeneous effects; native historical baseline n=1 (seed101),
C-vs-B native paired n=3. Local V3 references remain separate; no general winner.
All four LSF jobs report success,23/23 tasks each.
84 new Q tasks (seven single-factor variants, four model/target cells, three policy
seeds) +8 deterministic menu-matched Greedy controls; accepted LSF1607753–1607756.
251 tests/2 CUDA skips;36 local real-data gate cells;36/36 native capacity cells
completed in job1607752. Exact V3 baseline replay retained; no full baseline rerun.
Optional5003/5004 skipped uniformly by cost-only capacity rule. Main limits47/45/75/68
minutes plus15-minute gate =250 aggregate minutes. Executable source61e6934;
V1/V2/V3 evidence and SITITO manuscript unchanged. [Protocol and receipts](polus_v4_diagnostic_20261029/README.md).
Raw roots: `${GNN_ARTIFACT_ROOT}/polus-v4-diagnostic-20261029/` and
`${POLUS_TASK_ROOT}/results/polus-v4-diagnostic-20261029/` (main outputs under `campaign-primary/`).
New collected snapshot:`${GNN_ARTIFACT_ROOT}/polus-v4-collection-20261029T111126Z/`.
[Collection receipt and locations](polus_v4_diagnostic_20261029/collection/README.md).

**LOCAL-STRUCTURED-V3-SPRINT-20261027** — completed V3 sprint, followed by a separately
authorized primary collection. Local96 tasks +2 fixed probes; remote paired subset32
tasks +2 probes,4 LSF C4 jobs DONE;132/132 physical checkpoints verified. Separate
source/input/seed/platform IDs; no R2/V1 changes. [Protocol and launch metadata](local_structured_v3_sprint_20261027/README.md).
Raw group: `${GNN_ARTIFACT_ROOT}/local-structured-v3-sprint-20261027/`.
Launch-only session ended without monitoring: [original acceptance receipt](local_structured_v3_sprint_20261027/launch_receipt.json).
Primary processing:9,600 episodes/110,830 transitions/75,467 Q entries verified;
34 paired platform runs match actions/weights. Reachability guarantees target hits;
added Q value over Greedy unproven. [Report and compact tables](local_structured_v3_sprint_20261027/results/REPORT.md).
Collected raw snapshot: `${GNN_ARTIFACT_ROOT}/local-structured-v3-collection-20261027T054655Z/`.

**LOCAL-STRUCTURED-V2-Q-RANDOM-PILOT-20261027** — bounded validation-only diagnostic:
8 Q30/100-episode budget runs +4 representative lambda checks;80 Random masks
(seed1000–1019 per condition), all on frozen Cora dense seed101.92 physical models
verified;920 Q episodes replayed;28,841 old files unchanged. **Q REQUIRES ALGORITHM
REVIEW; Random ADEQUATE as repeated-mask baseline**. No production/R2 change,
research hyperparameter selection, test-metric analysis or broader campaign.
[Evidence](local_structured_v2_q_random_pilot_20261027/README.md). Raw group:
`${GNN_ARTIFACT_ROOT}/local-structured-v2-q-random-pilot-20261027/`.

**LOCAL-STRUCTURED-V2-CORA-MATRIX-20261027** — completed diagnostic extension:
18 new CPU runs (GCN50, GraphSAGE30/50, six methods each), **18/18 PASS**; with six
reused GCN30 cells the matrix is **24/24**, without repeating GCN30. Frozen POLUS
seed101 inputs with original row-sorted Cora provenance. **160 V2 passed / 2 CUDA
skipped**;28,699 prior files unchanged; no code fix or tuning. GO to a separately
scoped validation-only Q pilot. [Evidence](local_structured_v2_cora_matrix_20261027/README.md).
Raw group: `${GNN_ARTIFACT_ROOT}/local-structured-v2-cora-matrix-20261027/`.

2026-10-27 bounded diagnostic: **LOCAL-STRUCTURED-V2-CORA-SMOKE-20261027** — six
local CPU pruning runs from the archived POLUS seed101 checkpoint, with graph
reconstructed from recorded raw provenance; not pooled with V1. All6 PASS;
160 V2 tests passed/2 CUDA skips;28,628 old files unchanged. No implementation
change or new training. [Evidence and limits](local_structured_v2_smoke_20261027/README.md).
Raw group: `${GNN_ARTIFACT_ROOT}/local-structured-v2-smoke-20261027/`.

2026-10-27 audit fixes are implementation-only: architecture rejection, mandatory
graph pin and V2 protection from legacy split regeneration. Python3.12 synthetic
verification:160 V2 passed/2 CUDA skipped;39 V1 regression passes.
[Evidence](../implementation/LOCAL_STRUCTURED_V2_AUDIT_FIXES.md); raw logs are a new
`local-structured-v2-fixes-20261027` artifact group, not a real-data campaign.

Implementation-only update (2026-10-27): LOCAL-STRUCTURED-V2 Stages I–III use synthetic
pytest checks, not a new experiment campaign. See
[implementation plan](../methodology/IMPLEMENTATION_PLAN.md). No V1 artifacts
or registered campaign hashes are replaced; no V2 quality/timing result existed at that stage (subsequent smoke above).

`GNN_ARTIFACT_ROOT` defaults locally to the adjacent `../prunable_gnn_artifacts`. Raw artifacts are excluded from Git. Compact reports, summaries, configs and evidence live under `research/experiments/`.

| ID | Status | Platform / seeds | Scope | Canonical compact evidence | Raw evidence |
|---|---|---|---|---|---|
| `LOCAL-CPU-20261025` | complete, frozen | Mac CPU / 101, 202, 303 | 18 dense; 252 method-target | [`local_cpu_2026`](local_cpu_2026/README.md) | `${GNN_ARTIFACT_ROOT}/local-cpu-20261025/raw/` |
| `POLUS-CPU-101-C4-V2` | complete, frozen | polus-c4 CPU / 101 | 6 dense; 84 method-target; jobs 1598695–1598700 | [`polus_cpu_2026`](polus_cpu_2026/README.md) | `${GNN_ARTIFACT_ROOT}/polus-pilot-20261024/` |
| `KAGGLE-PILOT-2026` | protocol only; not executed on Kaggle | planned / 101 then 202,303 | reviewed protocol source | [`kaggle_pilot_2026/PLAN.md`](kaggle_pilot_2026/PLAN.md) | none |
| `HISTORICAL-COMPACT` | historical/debug only | local / 42,43; partial 44 | older 30-epoch outputs; do not pool | [`../audits/2026-10-22_local-codex.md`](../audits/2026-10-22_local-codex.md) | existing ignored `runs/` tree |
| `HPC-WAVE-COURSE` | excluded | POLUS / unrelated | 3-D wave equation, not GNN pruning | exclusion recorded in local audit | `/Users/perfect/Documents/ChatGPT/Математические дисциплины/cmc_msu_hpc_course_2026/` |

## Artifact integrity

| Artifact | Bytes | SHA256 |
|---|---:|---|
| `polus-pilot-20261024/archives/pilot-cpu-101-c4-v2-results.tar.gz` | 28,354,373 | `2b97d6923efa6cd8836830fdb7e13872c336da1700b514ab538118d47763fa62` |
| `polus-pilot-20261024/archives/pilot-cpu-101-c4-v2-evidence.tar.gz` | 32,196 | `66a905939510edb50e1a64ae814069c91a052fb362e61c6d7fe3cd88ea83aa3f` |
| `local-cpu-20261025/raw/audit/artifact_manifest.csv` | manifest | `f80545f83282e7fe1909f794f525a9baa43dbeb116f21a7ea9d5c9a00990376a` |
| `polus-environment-20261023/wheel/torch-2.0.1+polus.power8.1-cp310-cp310-linux_ppc64le.whl` | 55,591,834 | `30660e31f8c88159461ea647b2f175bdd23e501cad19b3a464fb451ddec8b830` |
| `polus-environment-20261023/wheel/setup-evidence.tar.gz` | 239,351 | `d334511a0950678081431a1d7482ed9ef34a948b9c8413dd8a3e9cba68687251` |
| `polus-environment-20261023/polus-research.zip` | 408,964 | `b5d59fb06f5124f8f74123fd687c40206eb560ef22c29df91d19f50b951439c8` |

The full local raw copy was compared file-for-file with `/Users/perfect/Documents/ChatGPT/Диплом/local_pilot_20261025`; no difference was found. A deterministic digest of the sorted per-file SHA256 listing is `66d43135c2eeaf5ac1bff216c1d775bc2093521d57c99c179f6b167bc2f2c27b`.

## Interpretation boundary

The two completed campaigns used the same research commit and seed-101 split hashes, but separate CPU stacks produced distinct dense checkpoints and non-identical post-prune outcomes. POLUS rows are therefore replication/platform evidence, not an additional independent seed. See [`polus_cpu_2026/PLATFORM_COMPARISON.md`](polus_cpu_2026/PLATFORM_COMPARISON.md).

Never overwrite a registered campaign. A corrected protocol or implementation creates a new ID, directory, artifact root and provenance record.

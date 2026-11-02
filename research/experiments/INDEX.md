# Experiment campaign registry

**Reading rule (2026-11-02):** entries below are dated events. Later collection
or analysis entries supersede earlier submission/blocker states for the same
campaign; retain all original evidence. Current overview: [STATUS](../STATUS.md).
Current paper direction: [new article plan](../publications/adaptive_strategies_2026/STATE_AND_PLAN.md).

**CITATION-V5-MAIN-20261102** —18 scientific jobs1612111–1612128 ACCEPTED for252
conditions; reporting1612129 ACCEPTED. Scientific sourcef050a5a; orchestration0251954.
Completion/collection/full audit **NOT VERIFIED in the current local summary**.
Last stored scheduler snapshot02:52:58 MSK is historical, not a current query.
Do not confuse preflight/test fixtures or the completed October31 campaign with
V5 production results. [Local receipt-backed status](polus_v5_main_20261102/LOCAL_STATUS_20261102.md),
[protocol](polus_v5_main_20261102/PROTOCOL.md), [operations](polus_v5_main_20261102/OPERATIONS.md).


**FLICKR-V5-COLLECTION-20261102** —56/56 COMPLETED, LSF1612130–1612185 historical
DONE;1,951 files/504 completion pins/922 source files verified,56 metadata PASS.
All target_band, Q30 episodes,54 four-thread/two12-thread C4 profiles. Collection
only; no independent model reload/inference or tests, no new jobs or citation edits.
[Report, recorded56-row results and reproducible integrity checks](flickr_submission_20261102/collection_20261102/REPORT.md).
Raw:`${GNN_ARTIFACT_ROOT}/flickr-v5-collection-20261102T094740Z/`.


**FLICKR-V5-PLACEMENT-CORRECTION-20261102** — same56 accepted IDs;21 eligible
PEND jobs unbound via bmod -mn;33 already RUN/DONE not targeted (including one
state-transition skip);two Q4/256 C4-only profiles preserved. Final33 DONE/3 RUN/
20 PEND;0 unexpected pending host restrictions,0 duplicate/new jobs. Only scheduler
metadata/configuration work,no tests or scientific execution; citation untouched.
[Report,original/final56 states,bmod responses,future host guard](flickr_submission_20261102/placement_correction_20261102/REPORT.md).

**FLICKR-STRUCTURAL-V5-SUBMISSION-20261102** — **56/56 ACCEPTED, LSF1612130–1612185**;
REJECTED0/UNRESOLVED0/NOT SUBMITTED0. Independent jobs/no citation dependency;
54 old4-thread allocations and two measured12-thread/C4/20-slot Q4/256 profiles.
Executable686cf3f; original frozen inputs and structural_budget_v5 unchanged.
Explicit historical-evidence reuse; no fresh PASS timestamps, tests, verification
workloads, auxiliary jobs, monitoring, collection, analysis, push or merge.
Citation campaign/source untouched. [Report,manifest,receipts](flickr_submission_20261102/SUBMISSION_REPORT.md).
Raw:`${GNN_ARTIFACT_ROOT}/flickr-submission-20261102/`.

**FLICKR-CPU-THREAD-SCALING-20261101 / collection20261102** — **COMPLETED8/8,
audited8 PASS/0 FAIL/0 SKIPPED**, native C4 LSF1612046 DONE.1026 raw files/876 source
pins verified;2616 stored transitions/2640 cache requests, all Q arithmetic/RNG/
masks/checkpoints and8 validation forwards independently checked. Exact cross-
thread trajectories/metrics and historical4/50 match.22 local tests PASS.
**12 threads recommended**: worst-target forecast133.05min,8 gives143.38min,
20 gives178.03min and fails170min acceptance. Forecast-only56-row proposal fits;
full56 still requires configurable production resource plumbing/fresh gate and is
not submitted. No new search/training/test labels, historical mutation or merge.
[Report,8-run table,figures,proof and forecast-only assessment](flickr_cpu_thread_scaling_20261101/collection_20261102/REPORT.md).
Raw:`${GNN_ARTIFACT_ROOT}/flickr-cpu-thread-scaling-collection-20261102T211303Z/`.

**FLICKR-CPU-THREAD-SCALING-20261101** —3 bounded jobs ACCEPTED;1612044/1612045
C2/C3 topology-only DONE; **1612046 benchmark RUN on C4**, confirmed by LSF/native
host and20 physical-core allocation (160 SMT threads). All measurements4/8/12/20
and targets30/50 sequential fresh processes on that same C4. Scientific30episodes,
technical3+2-action deployment prefix, algorithms/cache/inputs unchanged; source
10d7782.158 regression/13 focused PASS (overlap),0 FAIL/0 SKIPPED.0/8 native profiles
collected/verified; forecasts/equivalence/thread choice pending, full56 NO-GO and
0 submitted. C4 clarification satisfied without duplicate runs; no push/merge.
[Report, C4 confirmation and machine-readable manifests/status](flickr_cpu_thread_scaling_20261101/SUBMISSION_REPORT.md).
Raw:`${GNN_ARTIFACT_ROOT}/flickr-cpu-thread-scaling-20261101/`.

**POLUS-CUDA-PACKAGE-FEASIBILITY-20261101** — read-only, production upgrade
**NO-GO**, isolated candidate plausible/unverified. RocketCE Torch2.0.1 cu118
payload hash/ELF/POWER8 compiler/sm_60 evidence verified; stock Rocket Python
POWER9 ISA and exact CPU Python/zlib solve blockers identified. Alternative
Python3.10.13 technical105-package plan plus PyG2.4 wheel resolves, not installed
or runtime-qualified. RHEL7.5/R440 current; NVIDIA R450/R520 ppc64le EL8 binaries
are not official POWER8/EL7 qualification.5 dry-runs:3 success/2 dependency FAIL,
0 envs/jobs;273 source/36 historical campaign files unchanged. No push/merge.
[Report, binary inspection, complete candidate lock, admin/rollback plan](polus_cuda_package_feasibility_20261101/REPORT.md).
Raw:`${GNN_ARTIFACT_ROOT}/polus-cuda-package-feasibility-20261101/`.

**FLICKR-CUDA-AUDIT-20261101** — environment **NO-GO**, source521f20a.
P100 hardware found on C2/C3/C4, but existing research Torch2.0.1 is CPU-only;
15 visible prefixes/12 runnable Pythons yielded no compatible CUDA build.
Normal180min accessible, gpulong ACL excludes account; no GPU allocation tested.
Current runner CPU specialization/reporting and cross-device cache boundaries
documented.112 local focused PASS/0 FAIL/2 CUDA SKIPPED; native CUDA/CPU-GPU
comparison/performance NOT RUN.0 jobs/installs, no scientific changes/push/merge.
[Report, component audit, sanitized inventory and minimal setup proposal](flickr_structural_pruning_20261101/cuda_audit_20261101/REPORT.md).
Raw:`${GNN_ARTIFACT_ROOT}/flickr-cuda-audit-20261101/`.

**FLICKR-STRUCTURAL-PRUNING-20261101** — implementation/technical validation done,
**research SUBMITTED0/56, BLOCKED56: resource NO-GO**. Four frozen Dense seed101
architectures ×30/50% ×seven new budget-V5 policies; old controllers immutable.
146 focused/776 full PASS,0 FAIL,6 full skips. Eight separate technical jobs DONE,
56 outputs verified; native exact-execution bridge28 PASS. Revised Q4/256 safety
forecast193.021min exceeds normal180min; longer queues exclude this account.
No scientific result, partial matrix, push or merge. [Report and56-row manifest](flickr_structural_pruning_20261101/SUBMISSION_REPORT.md).
Raw:`${GNN_ARTIFACT_ROOT}/flickr-structural-pruning-20261101/`.

**FLICKR-DENSE-SWEEP-20261031 / collection20261101** — **COMPLETED32/32 and audited
PASS32/FAIL0/SKIPPED0**, LSF1611757–1611788 DONE.481 collected files hash-inventoried;
32 frozen checkpoints/physical parameter counts/source/input/split/config pins and
best states verified, train/validation metrics independently recomputed; no test
labels, training, pruning or new jobs.26 focused pytest PASS.26 max-epoch and six
patience stops remain screening evidence from seed101. Compact collection/table,
four figure pairs, audit and provenance: [report](flickr_dense_sweep_20261031/collection_20261101/README.md).
Raw new group:`${GNN_ARTIFACT_ROOT}/flickr-dense-collection-20261101T140723Z/`.
Original submission receipts below remain immutable.

**FLICKR-DENSE-SWEEP-20261031** — **SUBMITTED32/32, ACCEPTED LSF1611757–1611788**.
No research completion/status monitoring or result collection after acceptance.
GCN/GraphSAGE × layers2/3/4/5 × hidden32/64/128/256, seed101, common frozen60/20/20,
Adam LR0.001/dropout0.5/WD0.0005/500epochs/patience50; no pruning/fine-tuning/reuse,
test labels absent from uploaded inputs.12 large C4 jobs,20 unbound;4threads,
exclusive/single host,180min and12/13GiB. Native preflight3 jobs1611753–1611755 DONE,
18 short fits/reloads verified; max5/256 C4 GCN6.876s/epoch andSAGE8.151s/epoch.
26 focused PASS; full630 PASS/0 FAIL/6 skips. Source3995c60, schedulerfd99853.
[Submission report and later commands](flickr_dense_sweep_20261031/SUBMISSION_REPORT.md),
[manifest](flickr_dense_sweep_20261031/JOB_MANIFEST.csv),
[protocol](flickr_dense_sweep_20261031/PLAN.md),
[provenance](flickr_dense_sweep_20261031/PROVENANCE.json).
Raw:`${GNN_ARTIFACT_ROOT}/flickr-dense-sweep-20261031/`;
remote:`${POLUS_TASK_ROOT}/results/flickr-dense-sweep-20261031/`.
No historical pooling, manuscript edit, additional seeds, push or merge.

**POLUS-COMPLETE-SCIENTIFIC-ANALYSIS-20261031** — offline analysis of all 252
audited outcomes, no new experiment or historical pooling. Three paired dense
training/split seeds per stratum; sample SD, seed directions, actual compression
and measured native search cost. Lookahead matches Greedy masks/tensors 35/36 at
median 7.38× search; exceptional PubMed/GCN/202/50% gain +4.286 pp accuracy comes
with +0.931 pp compression and 11.69× search. Magnitude matches Greedy 18/18 at
50%. Recorded Lookahead previews corroborate all 72 Greedy decisions at identical
concrete parents. Native V2 Q: 36/36 horizon exhaustion, 87.441% mean compression,
466/720 unseen-value selections; no matched 30/50% quality comparison justified.
19 focused tests PASS / 0 FAIL / 0 SKIPPED; 12 deterministic core output files.
[Main analysis](polus_complete_pruning_20261031/analysis_20261031/SCIENTIFIC_ANALYSIS.md),
[Lookahead case](polus_complete_pruning_20261031/analysis_20261031/LOOKAHEAD_CASE_STUDY.md),
[Q mechanisms](polus_complete_pruning_20261031/analysis_20261031/Q_LEARNING_DIAGNOSTICS.md),
[H1–H4](polus_complete_pruning_20261031/analysis_20261031/HYPOTHESIS_EVALUATION.md),
[tables, figures and reproduction](polus_complete_pruning_20261031/analysis_20261031/README.md).
Source `55d0fc2`, audited base `8f24513`; local branch
`analysis/polus-complete-scientific-20261031`. No manuscript edit, push or merge.
External logs: `${GNN_ARTIFACT_ROOT}/polus-complete-scientific-analysis-20261031T121639Z/`.

**POLUS-COMPLETE-COLLECTION-20261031** — completed collection/independent audit,
source55d0fc2:18 dense/252 result sets,55 jobs DONE,1,226 files hash/size verified,
270 strict reload/projection/frozen-metric checks.92 PASS/160 WARNING/0 FAIL;
246 physically pruned +6 declared infeasible dense outcomes;96 outside symmetric
band. Greedy unselected-candidate observations, original reused-fit wall and
standardized dense inference latency remain explicitly missing.38 focused tests;
full585 PASS/0 FAIL/6 skips. GO for descriptive analysis with controller/coverage
limits; no new experiment, source-algorithm edit, push or merge.
[Collection](polus_complete_pruning_20261031/collection/COLLECTION_REPORT.md),
[audit](polus_complete_pruning_20261031/collection/AUDIT_REPORT.md),
[252-row verified dataset](polus_complete_pruning_20261031/collection/verified_results.csv),
[84-cell summary](polus_complete_pruning_20261031/collection/summary.csv).
Raw:`${GNN_ARTIFACT_ROOT}/polus-complete-collection-20261031T110939Z/`.

**POLUS-COMPLETE-PRUNING-20261031** — **SUBMITTED55/55 LSF jobs1611561–1611615**;
running/completed not queried.18 dense conditions (six verified reuses,12 new fits),
252 pruning tasks; Cora/CiteSeer/PubMed,GCN/GraphSAGE,seeds101/202/303,L3/H32,
500epochs/patience50,generated60/20/20; seven existing method/controller packages.
Source55d0fc2;547 pytest PASS/0 FAIL/6 skips;14 native smoke PASS.845min maximum
aggregate active wall; automatic Cora101 technical pilot and strict checkpoint
validation gate bulk work. No collection/analysis, historical overwrite, manuscript
edit, push or merge. [Submission report, exact IDs, dependencies and later commands](polus_complete_pruning_20261031/SUBMISSION_REPORT.md).

**QUESTIONS-LOCAL-READINESS-20261031** — completed integration/readiness check,
not a full experiment matrix. Native HeterophilousGraphDataset; generated60/20/20
primary split kept separate from ten official50/25/25 folds. GCN/GraphSAGE,3layers,
hidden32, seed101, original200-epoch/patience50 training; validation-only reporting,
one physical channel-removal smoke each. Focused32 PASS; full527 PASS/0 FAIL/6 skips.
Data/software GO; full Questions POLUS quality campaign NO-GO: severe imbalance,
minority collapse/very low recall at fixed argmax, convergence not established.
[Readiness report, resource observations and hashes](questions_readiness_20261031/REPORT.md).
Raw root: `${GNN_ARTIFACT_ROOT}/questions-readiness-20261031/`.
No wider fit, reward/algorithm edit, historical-result change or POLUS operation.


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

# Canonical GNN pruning research status

**Updated:** 2026-10-31

## Complete POLUS campaign preparation — 2026-10-31

A separately authorized 18-dense/252-pruning campaign is in preflight on branch
`campaign/complete-pruning-20261031`; no Questions, Flickr or OGB. Current SSH
access recovered, native activation/runtime and scheduler limits verified;
six archived seed101 checkpoints independently match the frozen archive and
reproduce validation metrics. Twelve new fits are planned for seeds202/303.
Historical algorithms/controllers remain unchanged. See [protocol and gates](experiments/polus_complete_pruning_20261031/README.md).
This preparation entry does not establish LSF submission or experiment completion.

## Questions dataset integration and limited readiness — 2026-10-31

Native PyG Questions integrated and locally validated:48,921 nodes,301 features,
153,540 undirected edges ->307,080 unique reciprocal entries; ten official50/25/25
folds preserved. Primary protocol remains generated60/20/20, seed101. Two fixed
3-layer hidden32 CPU baselines completed with unchanged training/early stopping;
GCN val accuracy/F1/AUC=96.770%/.4918/.5694, GraphSAGE=96.821%/.5075/.7514.
GCN predicts no positives; GraphSAGE recalls5/316. Physical one-channel smokes and
strict reload pass for both. Focused32 PASS; full527 PASS/0 FAIL/6 explicit skips.
**Data integration GO; Questions full POLUS quality campaign NO-GO at these
baselines pending convergence/imbalance protocol.** No tuning, remote jobs or
algorithm/history/manuscript changes. [Report and provenance](experiments/questions_readiness_20261031/REPORT.md).

## V4 offline verification and controlled analysis — 2026-10-29

All92 primary results and20 historical references independently verified from the
existing archive:13,200 primary episodes,137,000 transitions and magnitude choices,
44,474 final Q entries,300 policy snapshots;112 physical models reload/project
correctly and their validation accuracy/F1 recompute.10 offline audit tests passed.
Original1,285 collection files and launch evidence hashes remain unchanged. No new
training, pruning search, rollout, timing experiment or algorithm/manuscript edit.

All92 models reach the exact band, enforced by reachability;89 stop at target_band,
three menu12 runs reach it on max_steps20. Menu12 Q ties matched Greedy in12/12
comparisons while costing44–56x more search. More episodes, decay, incremental
reward and gamma0 show no uniform quality advantage across cells. Some small D/F
gains coexist with different compression/widths. Controlled A/B/D/E/F comparisons
use native POLUS seed101 only; C versus B uses three native paired seeds. Local
V3 seeds5001/5002 remain cross-platform references, never pooled as native controls.
One frozen training checkpoint per model; validation-only, no final winner claim.

See [verified V4 report, tables and figures](experiments/polus_v4_diagnostic_20261029/results/REPORT.md).
Analysis root: `${GNN_ARTIFACT_ROOT}/polus-v4-analysis-20261029T112436Z/`.
Primary active LSF time67.72min (76.80min including capacity), below250min reserved;
queue waiting is reported separately. Raw traces and full per-episode rows remain
external; compact verified analysis is versioned on the existing V4 branch.

## V4 raw results collected — 2026-10-29

All92 primary tasks completed (84 Q +8 Greedy),23/23 in each LSF job1607753–1607756;
all four LSF logs report successful completion. A new remote snapshot was downloaded
after recovering access for the researcher's current physical network. Archive
SHA256,1,285 file hashes,92 completed-marker output hashes and launch-manifest/source
identity were checked. No scientific aggregation/interpretation, model re-execution
or new experiment occurred. Raw collection is separate from the original campaign.
See [collection receipt and artifact locations](experiments/polus_v4_diagnostic_20261029/collection/README.md).
External root: `${GNN_ARTIFACT_ROOT}/polus-v4-collection-20261029T111126Z/`.
Primary scientific verification/analysis remains pending; earlier launch receipts
are preserved unchanged.

## V4 controlled POLUS diagnostic submitted — 2026-10-29

Separate branch/worktree `research/v4-tabular-diagnostic-20261029`, executable source
`61e693440b53d3cac3136600f3208a73c02492db`. **84 new Q tasks +8 deterministic Greedy
controls accepted**, LSF jobs1607753–1607756. This records submission, not completed
scientific results. No primary-job outcomes have been collected or interpreted.

251 tests passed/2 CUDA skips;36 local real-Cora smoke/control cells and four V3
reference comparisons PASS. Capacity job1607752 completed36/36 technical cells;
16 existing V3 baselines/1,600 stored episodes replay exactly through V4 without
new forward/search. Capacity source `a363861` remains separately identified;
resource-only correction preserves the scientific algorithm byte-for-byte and
the task-execution AST, with another four native new-source reference checks PASS.

Cost-only allocation:47/45/75/68-minute main job limits +15-minute capacity limit
=250 minutes aggregate, below300. Uniform optional seeds5003/5004 are skipped
in every Q variant:396-minute reservation would exceed the bound. Full primary
matrix retained; no accuracy-based scope change. Initial equal-wall capacity block
was resolved before any main acceptance. Queue waiting is recorded separately.

V1/V2/V3 source/evidence,101 selected historical raw files and SITITO manuscript
unchanged. No training/fine-tuning, environment/package/network/credential changes.
Baseline platform labels retained; only V3 POLUS seed101 is a matched remote
baseline, while local5001/5002 remain separate developmental references.
See [V4 protocol, gates and acceptance receipts](experiments/polus_v4_diagnostic_20261029/README.md).
External local root: `${GNN_ARTIFACT_ROOT}/polus-v4-diagnostic-20261029/`;
remote: `${POLUS_TASK_ROOT}/results/polus-v4-diagnostic-20261029/`.

## SITITO manuscript v1 frozen for GitHub — 2026-10-28

The researcher explicitly authorized versioning and publishing the completed
27 October manuscript bundle. A byte-identical snapshot is preserved under
`research/publications/sitito_2026/releases/v1_20261027/`; navigation and status:
[manuscript releases](publications/sitito_2026/RELEASES.md).
DOCX/DOC/PDF, LaTeX/BibTeX, figures, exact result tables, claim/source hashes,
official-requirement audit and submission checklist are included. The manuscript
is ready for scientific editing; author/declaration placeholders and official
template/direct-journal-rule verification remain open. Source pins, validation-only
claims and single-checkpoint limits are preserved. No new scientific experiment,
model loading, POLUS operation or changes to archived evidence occurred.

## V3 primary collection and processing — 2026-10-27

Researcher authorized a subsequent collection/processing stage after the completed
launch-only session. **98/98 local and34/34 POLUS runs verified** at executable
source `7e2c760`; all four LSF jobs DONE.132 physical checkpoints,9,600 training
episodes,110,830 transitions and75,467 Q entries audited;34 additional native
POWER8 reload/metric checks PASS.11 collector tests passed. Original893 local-batch
files unchanged;352 remote files transferred and hash-verified. No new experiment,
algorithm change, fine-tuning or test-set evaluation.

Exact-reachability Q/Random/Greedy each hit12/12 local targets; H4a6/12,
R2/H1/H2/H3 each0/12. Constrained Q matches Greedy quality at50% and has lower mean
validation accuracy at30% across three policy seeds; added learning value remains
unproven. All34 paired platform runs match actions/retained IDs/checkpoint tensors;
fixed forward differences are small with no argmax disagreement. Platforms and
policy seeds remain separate; one frozen training checkpoint per model.
See [primary report and compact data](experiments/local_structured_v3_sprint_20261027/results/REPORT.md).
External collection: `${GNN_ARTIFACT_ROOT}/local-structured-v3-collection-20261027T054655Z/`.

## V3 ablation sprint submission — 2026-10-27

Researcher explicitly authorized isolated H1/H2/H3/H4a/H4b variants and constrained
Random/Greedy controls, pinned frozen Cora inputs, local detached execution and a
bounded POLUS C4 batch. R2 remains a direct unchanged reference. Exact width/horizon
reachability is label-free; no H5 extension. Independent V3/execution tests and R2
regressions pass. Runtime/smoke and submission gates are recorded under
[the sprint protocol](experiments/local_structured_v3_sprint_20261027/README.md).
The launch-only session ended after submission attempts/receipts without monitoring
or analysis. Subsequent collection was separately authorized as recorded above.

Submission receipt: local **ACCEPTED**, PID `8440`,98 tasks; POLUS **ACCEPTED**, accepted LSF IDs `1602237, 1602238, 1602239, 1602240`,34 paired tasks. See [launch_receipt.json](experiments/local_structured_v3_sprint_20261027/launch_receipt.json). Acceptance records are preserved; no full-batch outcomes were collected or monitored.

## V2 validation-only Q / Random pilot — 2026-10-27

Completed on core `f110c5d`:8 Q budget conditions +4 targeted lambda conditions,
80 Random masks;92 physical checkpoints and920 episode traces verified. **Q:
REQUIRES ALGORITHM REVIEW**;100 episodes/max20 still overcompress strongly and
representative lambda0.5/2/5 checks do not change deployment. Sparse unvisited-Q0
choices, delayed terminal credit and state aliasing are documented; no core/R2
change applied. **Random: ADEQUATE as a repeated-mask baseline**, with large
mask/width variability; favorable single masks do not represent the mean.
Validation labels only; no final scientific settings chosen.160 V2 tests passed,
2 CUDA skips;28,841 prior files unchanged. Functional24-cell GO remains a technical
gate, not scientific qualification of the learned policy.
See [pilot evidence and next-step recommendation](experiments/local_structured_v2_q_random_pilot_20261027/README.md).

## V2 Cora functional matrix completed — 2026-10-27

Authorized extension at `11ab841`: **18/18 new runs PASS** (GCN50, GraphSAGE30/50,
six methods each); **24/24** functional cells with the unchanged earlier GCN30
smoke. Both frozen POLUS seed101 checkpoints are linked independently to archived
raw-input/job provenance and reuse the verified original Cora graph ordering.
**160 V2 tests passed / 2 CUDA skipped**; all physical models, saved metrics,
adaptive transitions and Q tables independently verified; **28,699** prior files
unchanged. No implementation fix, GNN training, tuning or inference benchmark.
**GO to a separately scoped validation-only Q-learning pilot** with these pinned
inputs. Lambda2/max3/3 episodes remain diagnostic; target misses are reported
algorithm outcomes. No pilot or broader campaign was executed here.
See [matrix evidence and exact artifact index](experiments/local_structured_v2_cora_matrix_20261027/README.md).

## V2 Cora smoke update — 2026-10-27

Authorized bounded real-data smoke at `e2389ca`: **6/6 CLI methods PASS** on local
CPU using the archived POLUS seed101 GCN32/32 checkpoint and manifest-reconstructed
original Cora edge ordering. **160 V2 passed / 2 CUDA skipped**; all checkpoints,
metrics and adaptive/Q traces independently verified;28,628 prior files unchanged.
No implementation fix, training, dataset download or campaign occurred.
**GO for a bounded Cora CPU pilot with this frozen input**; diagnostic lambda2/max3/
Q3 episodes are not approved research settings. Local checkpoint historical graph
identity remains unverified; no contemporaneous processed graph hash exists for
the selected POLUS run either, whose graph is reconstructed via recorded raw lineage.
See [full smoke evidence](experiments/local_structured_v2_smoke_20261027/README.md).

## V2 implementation update — 2026-10-27

Independent implementation audit follow-up: fixes on base `cf61c80` reject
dense architecture/config mismatches, require an original-graph hash, and block
legacy evaluate-pruned for V2 checkpoints. Python3.12.9: **160 V2 passed / 2 CUDA
skipped; 39 selected V1 regressions passed**. See [fixes and measured evidence](implementation/LOCAL_STRUCTURED_V2_AUDIT_FIXES.md).
This audit-fix entry predates the separate Cora smoke documented above.

LOCAL-STRUCTURED-V2 is separate from the frozen V1 campaigns below.
Implementation branch: `implementation/local-structured-v2-20261027`, based on
research draft `9076b06` with consolidated V1 evidence `d422263` preserved by merge.
Audit verified directly at `5c4259664dfb5ae05e84deadacfcdc9486abb7ca`; the local
audit file has the identical Git blob `309cfcc5b56f6d141aa343cf5dd5a430e2a89cf2`.
See [revised canonical specification](methodology/LOCAL_STRUCTURED_V2_SPEC.md)
and [three-stage implementation plan](methodology/IMPLEMENTATION_PLAN.md).
Stage I Local Surgery: Red **40 failed / 31 passed / 2 CUDA skipped**,
then Green **71 passed / 2 CUDA skipped**; legacy subset **31 passed**.
See [measured evidence and limitations](implementation/LOCAL_STRUCTURED_V2_STAGE_I.md).
R2 decisions are explicit in SPEC §15. All three prototype stages completed
sequential Test-First gates: II Red c49832c→Green4df84c5; III Red08e4fb8→Green.
Final V2 suite: **127 passed / 2 CUDA skipped**, V1 no-fit regression subsets:
**39 passed**. See [III report](implementation/LOCAL_STRUCTURED_V2_STAGE_III.md)
and [prototype guide](../docs/local_structured_v2.md). Existing prune CLI routes
six separate V2 identifiers; full checkpoint/split/config provenance is recorded.
Stage I revalidated on Python3.11.13:71 passed/2 skipped, then on user-selected
conda gnn_pruning (Python3.8.20):71 passed/2 skipped. The unnecessary temporary
.venv was removed; conda environments unchanged. Subsequent work uses gnn_pruning.
CUDA/POLUS remain unverified; no dense training, large experiment or server run
is authorized. Tiny synthetic Q-policy unit tests are part of Stage III. Historical evidence is unchanged.

**Article target:** 2026-11-02 (researcher-designated; official venue path/deadline still requires confirmation)

## Start here

The only canonical repository is:

`/Users/perfect/Desktop/code/MSU/MSU_SQI_25_DIPLOMA/test_codex/prunable_gnn_codex`

The reviewed experimental implementation is pinned to `72293149d8fff186dd268789e7b8fea5848c9360`. Consolidation work is on `consolidation/gnn-research-20261025`; inspect the branch HEAD and working tree before further edits.

Read next:

1. [`experiments/INDEX.md`](experiments/INDEX.md) for campaign status, paths and hashes.
2. [`experiments/local_cpu_2026/audit/AUDIT_REPORT.md`](experiments/local_cpu_2026/audit/AUDIT_REPORT.md) for the three-seed local audit.
3. [`experiments/polus_cpu_2026/README.md`](experiments/polus_cpu_2026/README.md) for the POLUS pilot and platform caveats.
4. [`publications/sitito_2026/PLAN.md`](publications/sitito_2026/PLAN.md) for the paper plan.

Do not use chat history as the source of record when these files and the hashed artifacts answer the question.

## Canonical evidence now available

- Local CPU campaign, 2026-10-25: 18 dense conditions and 252 method-target results, seeds 101/202/303, all strict checkpoint reloads and exact structural-reduction checks passed.
- POLUS CPU campaign `pilot-cpu-101-c4-v2`, 2026-10-24: six LSF jobs 1598695–1598700, six dense conditions and 84 method-target results, seed 101, code commit `72293149d8fff186dd268789e7b8fea5848c9360`.
- The local and POLUS campaigns share the seed-101 split hashes but produced different dense checkpoints and post-prune values. They are separate platform executions and must not be pooled.
- POLUS seed 101 is a repeat of seed 101 on another platform, not a fourth statistically independent seed.
- The unrelated `cmc_msu_hpc_course_2026` tree contains wave-equation work and contributes no GNN result.

## Scientific guardrails

- Compare methods at achieved parameter reduction, not requested target alone.
- Adaptive Layerwise did not reach either 0.30 or 0.50 target in the local campaign; achieved reduction was 0.0062–0.0440. Treat it as a failed matched-compression comparator, not as a valid 30%/50% point.
- Q-learning uses a different local non-monotonic surgery capability from static cascade baselines. Differences cannot be attributed to learning alone.
- Group Lasso is a pruning-time scoring/regularization pass, not dense training with a Group-Lasso objective.
- Test metrics are descriptive reporting only. Do not use them for protocol changes, method selection or winner claims.
- Timing evidence is platform-specific. Parameter reduction is not measured acceleration, and the current Q-learning speed proxy is not hardware-aware.

## Storage and local overrides

Large/raw artifacts live outside Git in the adjacent directory:

`/Users/perfect/Desktop/code/MSU/MSU_SQI_25_DIPLOMA/test_codex/prunable_gnn_artifacts`

Portable tools may set `GNN_REPO_ROOT` and `GNN_ARTIFACT_ROOT`. POLUS uses `POLUS_TASK_ROOT=$HOME/gnn-polus-20261023`; SSH target and Mac route details remain in the user's existing secure configuration or ignored `research/local-overrides.yaml`. The tracked example contains placeholders only.

Do not move, delete, regenerate or overwrite the external artifact tree. The original source folder `/Users/perfect/Documents/ChatGPT/Диплом` remains an additional backup and was not deleted.

## Immediate work to 2026-11-02

1. Build paper tables/figures from the local three-seed condition-level aggregate; do not pool across datasets/models or with POLUS.
2. Present Adaptive Layerwise target failure and Q-learning target/stop behavior explicitly.
3. Use the POLUS run as cross-platform/operational evidence only unless a separately declared analysis needs it.
4. Freeze the claim-to-evidence table before final prose and verify the official submission route/template/deadline.
5. Obtain supervisor approval; do not launch replacement experiments merely because a result is inconvenient.

## Working convention for chats

Use one ChatGPT chat for research decisions, paper structure and interpretation. Use one Codex/Work chat rooted at this canonical repository for files, code and verification. At the start of either chat, point to this file and `experiments/INDEX.md`; record every material decision or new campaign there before relying on it elsewhere.

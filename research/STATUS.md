# Canonical GNN pruning research status

**Updated:** 2026-10-25

## V2 implementation update — 2026-10-27

Independent implementation audit follow-up: fixes on base `cf61c80` reject
dense architecture/config mismatches, require an original-graph hash, and block
legacy evaluate-pruned for V2 checkpoints. Python3.12.9: **160 V2 passed / 2 CUDA
skipped; 39 selected V1 regressions passed**. See [fixes and measured evidence](implementation/LOCAL_STRUCTURED_V2_AUDIT_FIXES.md).
This is not a new campaign; Cora smoke testing has not been run.

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

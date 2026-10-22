# Historical benchmark reanalysis — second local Codex report

**Status:** CODEX_REPORTED_LOCAL; attached local Codex output received 2026-10-22. Original CSV/checkpoints have not been independently inspected by ChatGPT. This is not a new confirmed publication result.

## Principal finding

The current legacy `fast_debug` compact benchmark **does not support** a general claim of Q-learning superiority over static structured pruning **at matched achieved parameter reduction**.

The matching-rule sensitivity is substantial:
- Validation-based selection within compression bands gave Q minus static **+0.61 test-accuracy points** (30–35% band, 24 paired units), **+2.10 points** (50–55%, 17 paired units), but **−3.59 points** (45–50%, 7 paired units).
- Nearest-result matching within 2.5 percentage points of reduction instead gave **−0.23** (around 30%, 22/30 units) and **−0.19** (around 50%, 19/30 units).
- No evidence supports robust effects above 70%; zero Q-learning outcomes reached requested 80% or 90% in these local runs.

## Experimental inventory

- 159 completed configurations: seed 42 and seed 43 complete (120 configs total); seed 44 partial (39/60, no PubMed).
- All compared configurations within a dataset/model/seed group share equal dense parameters/accuracies, splits, and tensor-equivalent dense checkpoint states, according to local hash checks.
- Historical static `achieved_sparsity` used channel removal, unlike RL parameter reduction. Reanalysis uses `1 - P_pruned/P_dense`; old static plots are not directly comparable.
- Structured global and layerwise magnitude are duplicates in this benchmark (240 paired matching outputs).
- The legacy Q-learning adaptation has greater local structural surgery freedom than the static cascade, confounding attribution of differences to learning.
- Data from different Q-learning accuracy budgets and stop reasons were formerly connected into a single curve; broad pooled summaries also mix architecture/dataset and replicate budget variants.

## Constraint and target outcomes

- Across 714 Q-learning outcomes, **115 = 16.1%** exceeded the configured validation-accuracy-loss threshold after taking an action (budgets are a **post-action stop trigger**, not strict feasibility guarantees).
- Violations: budget .05 = 50/240; .07 = 43/240; .10 = 22/234.
- Target-reaching outcomes: .1 = 119/119; .3 = 113/119; .5 = 75/119; .7 = 17/119; .8 = 0/119; .9 = 0/119.
- The agent often terminates below high compression targets, but this does not guarantee respecting accuracy constraints.

## Training quality concern

- Configurations are `fast_debug`, train at most 30 epochs, no fine-tuning.
- Of 40 dense architecture/seed conditions, 18 reached the training cap and 13 had their best epoch at epoch 29.
- This is a **convergence uncertainty**, not conclusive proof of undertraining. It challenges treating legacy outputs as a full publication-grade benchmark.

## Paper-level decision and guardrails

1. Do not claim general RL superiority, strict quality-loss constraints, or hardware-aware/real-speedup optimization.
2. Legacy results may inform a limited exploratory analysis, especially early stopping, target misses and actual compression.
3. Do **not** automatically complete the 21 missing old seed-44 configurations; first compare the value of this with a small new canonical study under a better training protocol.
4. For a comparative paper, freeze validation-based matching and selection criteria **before** looking at new canonical test results; use equal dense models, splits and actual parameter reduction.
5. A canonical study should use 3 complete independent seeds at minimum (5 if feasible) and an appropriate dense training budget; preserve environment/Git SHA/configs/checkpoint identities.
6. Design work and draft writing should proceed in parallel. No new full-grid benchmark is authorized by this audit.
7. Conference priority: submit by the working cutoff 2026-11-02.

## Recommended manuscript artifacts

- Achieved reduction and stopping/threshold-violation table.
- Individual dataset/model/seed accuracy vs actual parameter-reduction plot.
- Q-learning requested-vs-achieved reduction, with stop reasons.
- Predeclared compression-matched comparison with actual coverage.
- Dense baseline quality/convergence table.

**Next action:** ask local Codex to prepare and estimate one narrow, reproducible canonical experiment with a sufficiently generous dense-training budget. Do not initiate expensive runs before the expected runtime and comparative design are reviewed.

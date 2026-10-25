# Canonical local GNN pruning audit — 2026-10-25

## Outcome

- Full requested matrix completed: 18 dense conditions and 252 method-target results (3 datasets × 2 models × 3 seeds × 7 methods × 2 targets).
- Quality audit: 252/252 strict checkpoint reloads passed; 252/252 parameter reductions recomputed exactly; every variant reused its condition's dense checkpoint and byte-identical split; no result failed the audit.
- Git commit: `72293149d8fff186dd268789e7b8fea5848c9360`; source checkout was clean after execution.
- This is a **local CPU** execution from the original repository, not a POLUS or Kaggle execution. Scientific config fields match the reviewed Kaggle configs; only base/config path, data/output path, experiment name, and seed (202/303) differ.
- Historical `fast_debug` outputs were excluded from all tables below.

## Provenance decision

`cmc_msu_hpc_course_2026` contains the 3-D wave-equation assignment. Its only POLUS manifest lists MPI/hybrid wave jobs, and the tree contains no GNN dataset/model/pruning trace. Therefore it contributes zero GNN results.

The old compact benchmark remains a separate provenance group: 120 complete configurations for seeds 42/43, 39 partial seed-44 pipelines (no PubMed), all with 30 epochs and patience 10. It is not pooled with this canonical run.

## Dense training

| Dataset | Model | val accuracy mean±sd | test accuracy mean±sd* | epochs range |
| --- | --- | ---: | ---: | ---: |
| citeseer | gcn | 0.7574±0.0083 | 0.7603±0.0048 | 59–60 |
| citeseer | graphsage | 0.7634±0.0061 | 0.7497±0.0128 | 57–61 |
| cora | gcn | 0.8946±0.0049 | 0.8674±0.0080 | 69–87 |
| cora | graphsage | 0.8823±0.0011 | 0.8637±0.0092 | 64–77 |
| pubmed | gcn | 0.8771±0.0031 | 0.8723±0.0076 | 415–500 |
| pubmed | graphsage | 0.8915±0.0040 | 0.8901±0.0048 | 137–177 |

\* Test values are descriptive reporting only. No model, method or protocol choice used test accuracy.

All 18 histories are complete. Seventeen conditions stopped by the 500-epoch ceiling or patience-50 rule; PubMed/GCN seed101 reached the 500-epoch ceiling (best checkpoint epoch index 479), so it retains some convergence uncertainty rather than being silently extended.

## Method/target results

| Method | Target | achieved mean [min,max] | within ±0.025 | val drop mean / max | test drop mean±sd* | budget violations† |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| adaptive_layerwise | 0.30 | 0.0216 [0.0062, 0.0440] | 0/18 | 0.0764 / 0.1474 | 0.0778±0.0481 | 10 |
| adaptive_layerwise | 0.50 | 0.0216 [0.0062, 0.0440] | 0/18 | 0.0764 / 0.1474 | 0.0778±0.0481 | 10 |
| global_magnitude | 0.30 | 0.3189 [0.3143, 0.3253] | 12/18 | 0.0361 / 0.1201 | 0.0349±0.0370 | n/a |
| global_magnitude | 0.50 | 0.5074 [0.5021, 0.5149] | 18/18 | 0.1672 / 0.4954 | 0.1628±0.1317 | n/a |
| grasp | 0.30 | 0.3189 [0.3143, 0.3253] | 12/18 | 0.0509 / 0.1575 | 0.0509±0.0418 | n/a |
| grasp | 0.50 | 0.5074 [0.5021, 0.5149] | 18/18 | 0.1846 / 0.4233 | 0.1867±0.1314 | n/a |
| group_lasso | 0.30 | 0.3189 [0.3143, 0.3253] | 12/18 | 0.0434 / 0.1774 | 0.0371±0.0474 | n/a |
| group_lasso | 0.50 | 0.5074 [0.5021, 0.5149] | 18/18 | 0.1325 / 0.2625 | 0.1326±0.0799 | n/a |
| q_learning_tabular | 0.30 | 0.3238 [0.3101, 0.3474] | 11/18 | 0.0167 / 0.0530 | 0.0201±0.0179 | 0 |
| q_learning_tabular | 0.50 | 0.4998 [0.3438, 0.5310] | 1/18 | 0.0392 / 0.0684 | 0.0441±0.0245 | 0 |
| random | 0.30 | 0.3189 [0.3143, 0.3253] | 12/18 | 0.0503 / 0.2346 | 0.0458±0.0522 | n/a |
| random | 0.50 | 0.5074 [0.5021, 0.5149] | 18/18 | 0.1179 / 0.3173 | 0.1170±0.0746 | n/a |
| snip | 0.30 | 0.3189 [0.3143, 0.3253] | 12/18 | 0.0378 / 0.1460 | 0.0351±0.0351 | n/a |
| snip | 0.50 | 0.5074 [0.5021, 0.5149] | 18/18 | 0.1330 / 0.3564 | 0.1357±0.1194 | n/a |

\* Test-drop aggregates are descriptive and pool heterogeneous dataset/model conditions; use `aggregate_by_condition_method_target.csv` for paper figures and inference.

† The 0.07 validation-drop budget is configured only for Adaptive Layerwise and tabular Q-learning. Static methods have `n/a`, not zero-budget compliance.

## Main quality findings

1. **Adaptive Layerwise is not a valid matched-compression comparator in this implementation.** All 36 rows missed both targets: achieved reduction was only 0.0062–0.0440. It stopped with `max_accuracy_drop_exceeded` in 20/36 rows and `max_steps_reached` in 16/36. The same final model was produced for targets 0.30 and 0.50 within each condition.
2. **Q-learning respected the 0.07 validation-drop budget in all 36 rows**, but target matching was imperfect. At target 0.30, all 18 runs reported `target_sparsity_reached`, with 11/18 within ±0.025. At target 0.50, 13/18 reached the target and 5/18 selected `agent_stop`; only 1/18 was within ±0.025 because discrete steps often overshot above 0.525 or stopped short (minimum 0.3438).
3. **Static cascade methods have identical achieved parameter counts for a given condition/target**, because they select the same number of first-hidden-group channels and differ only in which channels they keep. At target 0.30, architecture granularity yielded 0.3143–0.3253 reduction (12/18 within ±0.025); at target 0.50 all 90 static rows were within tolerance.
4. **Surgery capabilities differ.** Random, Global Magnitude, SNIP, GraSP and Group Lasso use first-hidden-group cascade surgery; Adaptive uses monotonic sequential cascade surgery; Q-learning uses local non-monotonic surgery and can produce unequal hidden widths. Differences cannot be attributed to learning alone.
5. **Group Lasso is a pruning-time regularized scoring pass**, not dense training under a Group-Lasso objective. It should not be described as a fully trained Group-Lasso baseline.

## Matrix and artifacts

- `method_target_matrix.csv`: all 252 rows, metrics, checks, checkpoints and surgery modes.
- `condition_matrix.csv`: all 18 dense conditions, split/checkpoint hashes and convergence checks.
- `aggregate_by_condition_method_target.csv`: paper-ready 3-seed summaries without pooling conditions.
- `quality_issues.csv`: every target miss and budget violation.
- `artifact_manifest.csv`: relative paths, sizes and SHA-256 hashes.
- Per-condition directories contain resolved config, split, dense checkpoint, 14 pruned checkpoints, metrics, Q-learning/adaptive traces and logs.

## Article priorities to 2026-11-02

1. Freeze this local provenance group and do not rerun/select based on test ranking.
2. Base the primary comparison on achieved parameter reduction, per dataset/model, with seeds paired by split and dense checkpoint.
3. Treat Adaptive Layerwise as an implementation/protocol mismatch, not a 30%/50% result. Either report it explicitly as failed-to-reach-target or define and predeclare a corrected separate experiment; do not replace these rows silently.
4. Report Q-learning stopping/overshoot behavior and surgery asymmetry. Avoid a general “RL is superior” claim without a no-learning baseline in the same local-surgery action space.
5. Use local CPU timing only as local runtime evidence; do not market it as Kaggle/POLUS/GPU speedup or hardware-aware optimization.

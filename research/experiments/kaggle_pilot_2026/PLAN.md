# Kaggle canonical pruning benchmark — pilot protocol

**Status:** approved design, not executed  
**Recorded:** 2026-10-22  
**Evidence ID:** `EXP-CANONICAL-01`  
**Immediate publication target:** scientifically defensible submission by 2026-11-02

This benchmark is a new provenance group. Never pool its estimates, plots or uncertainty with historical `fast_debug` coursework runs.

## 1. Evidence motivating the new benchmark

The following findings are `CODEX_REPORTED_LOCAL` and preserved in `research/audits/`; they are not newly reproduced by this plan:

- only historical seeds 42 and 43 are complete;
- historical dense training used a 30-epoch debug budget and has unresolved convergence uncertainty;
- historical static and Q-learning sparsity fields used different definitions;
- structured global and layerwise magnitude produced identical compact-benchmark results;
- Q-learning often stopped below requested compression and its post-action accuracy check could exceed the configured budget;
- static cascade surgery and Q-learning local surgery exposed different structural capabilities;
- historical evidence does not establish general Q-learning superiority.

## 2. Approved pilot matrix

| Dimension | Values |
| --- | --- |
| Platform | Kaggle Notebooks |
| Pilot seed | 101 |
| Datasets | Cora, CiteSeer, PubMed |
| Models | GCN, GraphSAGE |
| Architecture | 3 layers, hidden width 32 |
| Dense budget | Maximum 500 epochs; validation-loss early stopping patience 50 |
| Candidate methods | Random; structured Global Magnitude; SNIP; GraSP; Group Lasso; Adaptive Layerwise; tabular Q-learning |
| Requested compression | 0.30 and 0.50 for direct comparison |
| Q-learning accuracy-drop budget | 0.07 on validation accuracy |
| Fine-tuning | Disabled; primary endpoint is raw post-prune behavior |
| Repeated study | Seeds 101, 202 and 303 after the seed-101 pilot passes |

The structured Magnitude baseline is `global_magnitude`; `layerwise_magnitude` is excluded because the historical structured implementation produced identical results. Confirm the current implementation path before creating configs.

The conceptual pilot contains six dense conditions and up to 84 method-target evaluations: `3 datasets × 2 models × 7 methods × 2 targets`. One dense checkpoint per dataset/model/seed condition must be reused for all paired methods and targets.

## 3. Comparison rules

1. Use the identical split and dense checkpoint for every method in a dataset/model/seed condition.
2. Keep requested sparsity and achieved sparsity separate.
3. Define achieved structural sparsity only as `1 - pruned_parameter_count / dense_parameter_count`.
4. Use validation loss for dense early stopping and validation metrics for method configuration, adaptive decisions and protocol checks.
5. Do not use pilot test results to select a method, target, architecture or protocol change. The pilot is not a winner-selection study.
6. Compare quality against achieved parameter reduction. Do not imply equivalence solely because requested targets match.
7. Report target misses, accuracy-budget violations, stopping reasons and failed/incompatible runs.
8. Report each method's structural-surgery path, layers affected, independent-width capability, gradient/training requirements and pruning overhead.
9. Do not attribute a difference to learning alone when methods have different feasible model transformations.
10. Report Random as a stochastic method with its seed. Do not treat repeated targets from one seed as independent replicates.
11. Keep historical and canonical files, summaries and plots in separate roots and provenance manifests.

## 4. Pilot acceptance gate

The pilot passes only if all of the following are reviewed without using test results to choose a winner:

- all candidate pruners resolve and either complete structurally or are explicitly classified as incompatible;
- structural methods reduce parameter count and produce reloadable checkpoints;
- dense runs record epoch-by-epoch train/validation loss and quality, best epoch, stop reason and total epochs;
- shared split and dense-checkpoint identities are verifiable across paired methods;
- achieved parameter reduction is recomputable from saved parameter counts;
- Q-learning traces identify target, achieved reduction, validation drop, deployment result and stop reason;
- expected Kaggle runtime, storage and accelerator availability are documented;
- required artifacts can be downloaded and checksummed;
- no protocol decision depends on pilot test-set ranking.

After this gate, freeze the protocol and extend it to at least three complete seeds. If a candidate method is incompatible, document the exact capability mismatch before excluding or adapting the comparison; do not silently replace it.

## 5. Required artifacts

Each Kaggle run must preserve:

- Git commit SHA and dirty/clean status;
- notebook version or checksum and exact invocation order;
- resolved configs and config checksums;
- dataset identity, split file and split checksum;
- dense and pruned checkpoints with checksums;
- seed, run ID and paired-condition ID;
- Python, PyTorch, PyG, CUDA, driver and Kaggle accelerator information;
- dense convergence history and early-stopping record;
- per-phase validation/test accuracy and macro-F1;
- requested sparsity, achieved parameter reduction, channel diagnostics and parameter counts;
- pruning time, Q-learning overhead, inference timing protocol and raw timing summaries when available;
- Q-table, RL training trace, deployment trace and action-space diagnostics for Q-learning;
- CSV/JSON summaries, stdout/stderr log and a machine-readable artifact manifest.

Test metrics may be emitted by the existing pipeline, but they are quarantined from pilot decisions until the protocol and repeated study are frozen.

## 6. Kaggle execution restrictions

- The researcher manually starts every experiment.
- Codex must never start training, pruning, smoke tests or benchmarks without explicit permission.
- Codex may prepare and review notebook/config code only when separately requested.
- Do not install dependencies or alter the Kaggle environment during documentation work.
- Do not use Google Colab for this benchmark.
- Do not start unrelated algorithm development, HPC benchmarking or later-publication experiments.
- Stop after the pilot and review artifacts/runtime before authorizing additional seeds.

## 7. Unverified assumptions and pre-implementation checks

- Kaggle's available Python/PyTorch/PyG/CUDA combination is compatible with the pinned repository revision.
- GraSP, Group Lasso and Adaptive Layerwise can execute on both GCN and GraphSAGE under the required structured protocol.
- All candidate methods can be compared at requested targets 0.30 and 0.50 without changing their scientific definitions.
- A 500-epoch ceiling and patience 50 are sufficient to diagnose convergence for all six dense conditions.
- Kaggle runtime and storage are sufficient for the pilot matrix.
- The official conference deadline/submission path matches the researcher-designated 2026-11-02 target.
- Fine-tuning remains disabled for this protocol; changing that would define a different experiment.
- Seeds 202 and 303 are reserved for the confirmatory extension after seed 101 passes the pilot gate.

These assumptions must be checked before notebook execution. A failed assumption is a protocol issue to document, not permission to modify pruning algorithms silently.

## 8. Pilot deliverables

- compatibility matrix for every dataset/model/method;
- dense convergence table and curves;
- runtime/storage estimate for extending to at least three seeds;
- artifact manifest and checkpoint/split pairing report;
- achieved-versus-requested parameter-reduction table;
- Q-learning target, validation-budget and stopping summary;
- explicit recommendation to freeze, amend or stop the repeated protocol;
- no one-seed winner declaration.

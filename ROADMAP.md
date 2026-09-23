# ROADMAP.md

## Implemented Foundation

- Config layering and YAML-driven experiments.
- PyG dataset registry, seeded exact-ratio splits, and explicit DBLP author projection.
- GCN and GraphSAGE node classifiers.
- Dense training with validation-loss early stopping, checkpointing, evaluation, timing, and memory metrics.
- Structural channel pruning and layer reconstruction for GCN and GraphSAGE.
- Dense -> prune -> post-prune -> optional post-finetune pipeline.
- Repeated suites with independent seeds and aggregate CSV reporting.
- Static/sensitivity pruners: random, global magnitude, layerwise magnitude, SNIP, GraSP, L1 threshold, group lasso, movement, and hard-concrete.
- Adaptive layerwise pruning with trace artifacts.
- Experimental tabular Q-learning with graph-aware bucketed state, feasible-action filtering, explicit stop action, training/deployment separation, local non-monotonic structural surgery, and diagnostics.
- Compact benchmark configs plus collection, summary, and plotting scripts.

## Stabilization Priorities

1. Keep checkpoint/split fairness and seed aggregation covered by regression tests.
2. Keep model selection and adaptive decisions validation-based; reserve test metrics for final descriptive reporting.
3. Compare methods by achieved sparsity as well as requested sparsity.
4. Validate structural checkpoint reload and post-finetune behavior for non-uniform hidden widths.
5. Run controlled multi-seed studies before drawing scientific conclusions.

## Deferred Infrastructure

- Optional MLflow integration.
- Additional model families such as GAT and GIN.
- Subgraph/minibatch training for very large graphs.
- Hardware-aware latency objectives based on measured latency rather than parameter-reduction proxies.
- Input-graph sparsification, which is separate from the current model-channel pruning scope.

## Research Status

The tabular Q-learning module is an experimental baseline, not a finalized thesis contribution. The final Master's thesis direction must be selected only after the audited infrastructure produces reproducible multi-seed evidence. DQN, PPO, and other new algorithms are intentionally outside the current cleanup scope.

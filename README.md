# gnn_pruning

`gnn_pruning` is a config-driven PyTorch Geometric research framework for structural pruning of node-classification GNNs.

## Implemented Scope

- Datasets: Cora, Citeseer, PubMed, Flickr, Reddit, and an explicit homogeneous-author adapter for heterogeneous DBLP.
- Models: GCN and GraphSAGE with configurable depth, hidden width, and dropout.
- Reproducible exact-ratio train/validation/test splits.
- Dense training with validation-loss early stopping and checkpoint compatibility checks.
- Dense -> prune -> post-prune -> optional post-finetune evaluation pipeline.
- Structural channel surgery for GCN and GraphSAGE, including reloadable non-uniform hidden widths.
- Registered pruners: random, global magnitude, layerwise magnitude, SNIP, GraSP, L1 threshold, group lasso, movement, hard-concrete, adaptive layerwise, and tabular Q-learning.
- Repeated suites with per-seed rows and seed-level mean, sample standard deviation, and 95% Student-t confidence intervals.
- CSV/JSON/Markdown artifacts, diagnostics, benchmark collection, summaries, and plotting scripts.

MLflow is not currently wired into the runtime pipeline even though it remains a possible reporting extension.

## Structural Pruning Boundary

This project prunes **model hidden channels** and rebuilds affected GNN layers. A structured result must have fewer model parameters than its dense source model. Zero-valued masks alone do not count as final pruning.

The input graph is not sparsified: nodes and edges are unchanged. Model structural pruning and input-graph sparsification are separate research problems.

## Setup

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .[dev]
```

## Core CLI

```bash
PYTHONPATH=src python -m gnn_pruning --help
PYTHONPATH=src python -m gnn_pruning show-config --config configs/experiments/example.yaml
PYTHONPATH=src python -m gnn_pruning train --config configs/experiments/example.yaml
PYTHONPATH=src python -m gnn_pruning evaluate --config configs/experiments/example.yaml
PYTHONPATH=src python -m gnn_pruning run-dense --config configs/experiments/example.yaml
PYTHONPATH=src python -m gnn_pruning run-pipeline --config configs/experiments/pipeline_pubmed_gcn.yaml --progress
PYTHONPATH=src python -m gnn_pruning run-suite --config configs/suites/default_small.yaml --progress
```

The pipeline writes a shared dense baseline followed by method/sparsity variants:

```text
<run.output_dir>/
  resolved_config.yaml
  splits.yaml
  dense_checkpoint.pt
  metrics_train.json
  metrics_eval.json
  pipeline_results.csv
  summary_pipeline.md
  pruning/<method>/sparsity_<level>/...
```

Within one pipeline run, every pruning method starts from the same dense checkpoint and split. Different architectures, seeds, splits, or materially different configs fail checkpoint compatibility and are retrained rather than silently reused.

For structured runs, `achieved_sparsity` is the dense-to-pruned parameter-count reduction, making static and adaptive methods comparable. One-shot channel-pruning ratios remain available as `details.achieved_channel_sparsity`; they are not substituted for actual model compression.

`run-suite` derives seeds as `base_seed + run_index` and writes `suite_runs.csv` plus `suite_aggregate.csv`. Aggregation groups stable experimental conditions, not seeds, paths, achieved outcomes, rewards, or stopping reasons. Single-run rows are explicitly marked and do not receive standard-deviation or confidence-interval values.

## Tabular Q-Learning

`q_learning_tabular` is an experimental registered pruner used through the existing pipeline. Its implementation is split between `src/gnn_pruning/rl/` and the pruner orchestration in `src/gnn_pruning/pruning/methods.py`.

Current protocol:

1. Train a tabular Q-function over episodes; every episode resets to a copy of the same dense model.
2. Filter structurally infeasible actions before epsilon-greedy selection.
3. Use validation accuracy, never test accuracy, for adaptive rewards and accuracy-drop stopping.
4. Reset to the dense model again and run a deterministic greedy deployment rollout.
5. Return and report the deployment model, not the final exploratory episode model.

The state includes bucketed graph statistics and pruning progress. Actions prune a hidden layer by a configured ratio or select `STOP`. Local structural mode supports safe non-monotonic layer choices while preserving already-pruned downstream widths.

Example:

```bash
PYTHONPATH=src python -u -m gnn_pruning run-pipeline \
  --config configs/experiments/rl_comparison_pubmed_graphsage_l3_h16.yaml \
  --progress
```

Important limitations:

- This is tabular Q-learning, not DQN/PPO.
- Episodes do not fine-tune the model.
- The speed proxy is `compression_gain`, derived from parameter-count reduction. It is not measured inference latency.
- The agent is therefore not hardware-aware or latency-optimized.
- Discretized state aliasing, short training horizons, and target/accuracy trade-offs remain experimental limitations.
- This module is not claimed as the final Master's thesis contribution; the research direction remains open.

Q-learning variants save:

```text
q_table.json
rl_trace.json
deployment_trace.json
action_space_diagnostics.json
pruning_metrics_q_learning_tabular.json
```

## Benchmark Utilities

Compact benchmark:

```bash
PYTHONPATH=src python -m gnn_pruning run-suite \
  --config configs/suites/compact_pruning_benchmark.yaml \
  --progress

PYTHONPATH=src python scripts/summarize_compact_pruning_benchmark.py \
  --csv runs/compact_pruning_benchmark/suite_runs.csv
```

Collect completed artifacts and plot them:

```bash
PYTHONPATH=src python scripts/collect_benchmark_artifacts.py \
  --runs-root runs/compact_pruning_benchmark \
  --include-runs run_000 run_001 \
  --out-dir exports/compact_pruning_benchmark_artifacts \
  --zip

PYTHONPATH=src python scripts/plot_compact_pruning_benchmark.py \
  --csv exports/compact_pruning_benchmark_artifacts/combined_pipeline_results.csv \
  --out-dir exports/compact_pruning_benchmark_plots
```

Additional lightweight study tooling includes:

- `configs/suites/pubmed_stress_sweep.yaml`
- `scripts/summarize_pubmed_stress_sweep.py`
- `scripts/run_flickr_graphsage_sweep.py`
- `scripts/summarize_flickr_graphsage_sweep.py`
- `scripts/audit_experiment_correctness.py`

Plots and summaries based on final test metrics are post-hoc descriptive analyses. Configuration selection and adaptive pruning decisions must use validation metrics. Comparisons use achieved sparsity as the compression coordinate; requested sparsity remains a condition label, especially when adaptive methods stop early.

## Device and Timing

Use the supported config field:

```yaml
device:
  device: auto
```

Inference timing uses warmup passes, repeated timed passes, high-resolution CPU timers, and CUDA synchronization where applicable. Small-graph latency can be noisy; parameter count is the more stable structural-compression measure.

## Tests

```bash
PYTHONPATH=src pytest -q
```

The test suite uses synthetic or small debug graphs for pipeline, structural surgery, checkpoint, aggregation, and Q-learning regressions. It does not launch full benchmark suites.

## DBLP Adapter

DBLP is heterogeneous and is never silently treated as a homogeneous graph. The implemented `author_homogeneous` strategy projects `author -> paper <- author` relations into a co-author graph with author features and labels:

```yaml
data:
  name: dblp
  dblp_strategy: author_homogeneous
```

This drops non-author node types and relation semantics and can create dense co-author connectivity.

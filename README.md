# gnn_pruning

`gnn_pruning` is a modular, config-driven research framework for benchmarking one-shot pruning methods on graph neural networks (GNNs) for node classification.

## Current scope

This scaffold includes:
- package layout for core subsystems
- YAML-based layered config system (base + dataset + model + preset + user overrides)
- dataset factory with explicit backend mapping and DBLP adapter placeholder
- exact-ratio seeded split generation with split artifact saving
- baseline dense node-classification models: GCN and GraphSAGE
- dense training/evaluation workflow with early stopping and checkpointing
- full pruning pipeline orchestration (dense → prune → finetune across methods/sparsity levels)
- repeated benchmark suites with deterministic per-run seeds and aggregate reporting
- CLI entrypoints via `python -m gnn_pruning`
- pytest setup with smoke/config/data/model/training tests

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .[dev]
```

## Config layout

```text
configs/
  base/default.yaml
  datasets/*.yaml
  models/*.yaml
  presets/*.yaml
  experiments/example.yaml
  experiments/presentation_flickr.yaml
  experiments/presentation_reddit.yaml
  experiments/flickr_graphsage_l*_h*.yaml
  suites/default_small.yaml
  suites/default_medium.yaml
  suites/extended_optional.yaml
```

## Run CLI

```bash
python -m gnn_pruning --help
python -m gnn_pruning show-config --config configs/experiments/example.yaml
python -m gnn_pruning train --config configs/experiments/example.yaml
python -m gnn_pruning evaluate --config configs/experiments/example.yaml
python -m gnn_pruning run-dense --config configs/experiments/example.yaml
python -m gnn_pruning run-pipeline --config configs/experiments/pipeline_pubmed_gcn.yaml
python -m gnn_pruning run-suite --config configs/suites/default_small.yaml
python scripts/run_large_datasets.py
python scripts/run_flickr_graphsage_sweep.py
python scripts/summarize_flickr_graphsage_sweep.py --csv runs/flickr_graphsage_sweep/sweep_results.csv
```

`train` writes artifacts to `run.output_dir`:

```text
resolved_config.yaml
splits.yaml
dense_checkpoint.pt
metrics_train.json
```

`evaluate` writes:

```text
resolved_config.yaml
metrics_eval.json
```

`run-dense` writes a structured dense experiment directory with:

```text
resolved_config.yaml
splits.yaml
dense_checkpoint.pt
metrics_train.json
metrics_eval.json
summary.md
dense_results.csv
```

`run-pipeline` writes a full pruning benchmark run directory including:

```text
resolved_config.yaml
splits.yaml
dense_checkpoint.pt
metrics_eval.json
pipeline_results.csv
summary_pipeline.md
pruning/<method>/sparsity_<level>/...
```

`run-suite` repeats `run-pipeline` for `num_runs` with seeds derived from `base_seed + run_index` and writes:

```text
suite_runs.csv
suite_aggregate.csv
run_000/
run_001/
...
```

## Large dataset + architecture studies

- `configs/experiments/presentation_flickr.yaml` and `configs/experiments/presentation_reddit.yaml` run the same random/global-magnitude comparison setup used by presentation runs (dense/post_prune/post_finetune, sparsity 0.5/0.9).
- `configs/experiments/flickr_graphsage_l*_h*.yaml` provides a dedicated Flickr GraphSAGE depth/width sweep.
- `scripts/run_large_datasets.py` runs Flickr/Reddit comparisons and merges pipeline CSV outputs.
- `scripts/run_flickr_graphsage_sweep.py` runs all Flickr GraphSAGE sweep configs and writes `runs/flickr_graphsage_sweep/sweep_results.csv`.
- `scripts/summarize_flickr_graphsage_sweep.py` prints grouped summaries and best post-finetune/tradeoff rows.

## Run tests

```bash
pytest
```

# gnn_pruning

`gnn_pruning` is a modular, config-driven research framework for benchmarking one-shot pruning methods on graph neural networks (GNNs) for node classification.

## Current scope

This scaffold includes:
- package layout for core subsystems
- YAML-based layered config system (base + dataset + model + preset + user overrides)
- dataset factory with explicit backend mapping and DBLP adapter placeholder
- exact-ratio seeded split generation with split artifact saving
- CLI entrypoints via `python -m gnn_pruning`
- pytest setup with smoke/config/data tests

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
```

## Run CLI

```bash
python -m gnn_pruning --help
python -m gnn_pruning show-config --config configs/experiments/example.yaml
python -m gnn_pruning run --config configs/experiments/example.yaml
```

The `run` command writes artifacts to `run.output_dir`:

```text
resolved_config.yaml
splits.yaml
```

## Run tests

```bash
pytest
```

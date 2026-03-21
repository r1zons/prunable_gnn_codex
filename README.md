# gnn_pruning

`gnn_pruning` is a modular, config-driven research framework for benchmarking one-shot pruning methods on graph neural networks (GNNs) for node classification.

## Current scope

This initial scaffold includes:
- package layout for core subsystems
- a basic CLI entrypoint via `python -m gnn_pruning`
- pytest setup with a smoke test

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .[dev]
```

## Run CLI

```bash
python -m gnn_pruning --help
python -m gnn_pruning run --config configs/example.yaml
```

## Run tests

```bash
pytest
```

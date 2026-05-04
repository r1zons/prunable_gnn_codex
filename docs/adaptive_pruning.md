# Adaptive pruning baseline

`adaptive_layerwise` is a **rule-based adaptive baseline**, not a full RL agent.

It exists as a bridge toward future RL pruning by using RL-style concepts (`state`, `action`, `reward`) while keeping execution deterministic and easy to benchmark.

## Why this baseline exists

- Provides adaptive behavior beyond static one-shot ranking.
- Keeps compatibility with structural pruning rules (channels are physically removed).
- Produces step-wise diagnostics and an `adaptive_trace.json` artifact for analysis.

## Reward shape

Current reward is:

`reward = alpha * compression_gain + beta * speedup - gamma * accuracy_drop`

At intermediate pruning steps, speedup is currently set to `0.0` if not measured.

## Run commands

Debug run (fast local validation):

```bash
python -u -m gnn_pruning run-pipeline --config configs/experiments/adaptive_graphsage_citeseer_debug.yaml --progress
```

Full Citeseer adaptive-vs-static comparison:

```bash
python -u -m gnn_pruning run-pipeline --config configs/experiments/adaptive_graphsage_citeseer.yaml --progress
```

Small multi-dataset suite (Cora/Citeseer/PubMed):

```bash
python -u -m gnn_pruning run-suite --config configs/suites/adaptive_comparison_small.yaml --progress
```

Convenience script:

```bash
python scripts/run_adaptive_comparison.py --debug-only
python scripts/run_adaptive_comparison.py --full
```

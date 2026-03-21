# AGENTS.md

## Project Overview

This project is a modular research framework for benchmarking pruning methods on Graph Neural Networks (GNNs), using PyTorch Geometric.

Primary task:
- Node classification

Baseline models:
- GCN
- GraphSAGE

Future models:
- GAT
- GIN

The framework must support:
- dataset loading (PyG)
- exact-ratio random splits
- dense training and evaluation
- one-shot pruning
- structural compression (NOT mask-only pruning)
- post-pruning fine-tuning
- benchmarking (accuracy, F1, sparsity, timing, memory)
- CSV + MLflow reporting

---

## Core Design Principles

### 1. Modularity (CRITICAL)
- Each subsystem must be isolated:
  - data
  - models
  - training
  - pruning
  - surgery
  - evaluation
  - reporting
- No tight coupling between modules
- Use registries for:
  - datasets
  - models
  - pruners

---

### 2. Config-Driven Design
- All experiments must be driven by YAML config
- No hardcoded hyperparameters inside code
- Support config merging:
  - base + dataset + model + preset

---

### 3. Reproducibility
- All randomness must use a seed
- Default seed = 42
- Splits must be reproducible
- Save:
  - config snapshot
  - splits
  - checkpoints
  - metrics

---

### 4. Structural Pruning (VERY IMPORTANT)

⚠️ NEVER treat zero-masked weights as real pruning.

Rules:
- Masking is allowed ONLY as an intermediate step
- Final model MUST be structurally smaller
- Must support:
  - removing channels
  - rebuilding layers
  - updating dependent layers
- Parameter count MUST decrease after structured pruning

If unsure:
→ prefer rebuilding layers over masking

---

### 5. Separation of Pruning Phases

All pruning methods MUST follow this structure:

1. `score(...)` → compute importance
2. `plan(...)` → decide what to prune
3. `apply(...)` → modify model (with surgery if needed)

Do NOT mix scoring and model modification logic.

---

### 6. Fair Benchmarking

Within one experiment run:
- All pruners use the SAME:
  - dataset split
  - dense checkpoint
  - sparsity levels

Pipeline:
1. Train dense model
2. Save checkpoint
3. For each pruner:
   - load checkpoint into fresh model
   - prune
   - evaluate
   - fine-tune
   - evaluate again

---

### 7. Metrics (must be consistent)

Always report:

#### Quality
- accuracy
- macro F1

#### Sparsity / compression
- requested sparsity
- achieved sparsity
- parameter count
- parameter bytes
- checkpoint size

#### Runtime
- inference time (mean/std)
- pruning time
- fine-tuning time

#### Memory
- parameter memory
- runtime memory (if available)

---

### 8. Timing Rules

- Use warmup passes before timing
- Use multiple runs (mean + std)
- CUDA must be synchronized before timing
- CPU uses high-resolution timers

---

### 9. Device Support

- Must run on CPU
- Must support CUDA if available
- Device handled centrally (no scattered `.to(device)` logic)

---

### 10. Code Quality Rules

- Keep functions small and readable
- Avoid duplicated logic
- Prefer explicit over implicit behavior
- Add docstrings for all public classes/functions
- Use type hints where possible

---

### 11. Testing Requirements

Every major module must have tests:

Minimum:
- dataset loading
- split generation
- model forward pass
- training smoke test
- pruning plan correctness
- structural compression reduces parameters
- checkpoint save/load
- CSV schema

Use small datasets (Citeseer + fast_debug preset) for tests.

---

### 12. CLI Design

All functionality must be accessible via CLI:

Examples:
- train
- evaluate
- prune
- finetune
- run-pipeline
- run-suite

CLI must:
- accept config path
- not require code modification

---

### 13. Logging and Outputs

Each run must produce:
- checkpoints
- config snapshot
- metrics JSON
- CSV row(s)
- summary report

MLflow:
- must be optional (config flag)
- must not break project if disabled

---

### 14. Dataset Handling Rules

- Use PyTorch Geometric datasets
- Cache processed datasets by default

⚠️ DBLP:
- It is a heterogeneous dataset
- MUST NOT be treated as homogeneous silently
- Must go through explicit adapter logic

---

### 15. Extensibility (VERY IMPORTANT)

The code must make it easy to:
- add new pruning methods
- add new models (GAT, GIN)
- integrate RL-based pruning later

To achieve this:
- use clear interfaces
- avoid hardcoding assumptions
- keep pruning logic decoupled from models

---

### 16. What to Avoid

DO NOT:
- mix training and pruning logic
- hardcode dataset-specific behavior in core modules
- implement pruning as permanent masks only
- create monolithic scripts
- break reproducibility
- assume GPU is always available

---

### 17. First Priority

Always ensure this works:

- Citeseer dataset
- GraphSAGE model
- fast_debug preset
- full pipeline runs end-to-end

Only after that:
→ expand features

---

## Summary for Codex

When implementing code:
- follow modular architecture
- respect pruning + surgery separation
- ensure structural compression
- keep everything config-driven
- prioritize correctness over optimization

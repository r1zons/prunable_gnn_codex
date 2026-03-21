# ROADMAP.md

## Phase 0: Setup
- Create project structure
- Add dependencies
- Setup CLI

## Phase 1: Config System
- YAML config loader
- Presets
- Config merging

## Phase 2: Data Layer
- Dataset factory
- Split generation
- DBLP adapter placeholder

## Phase 3: Models
- GCN
- GraphSAGE
- Model registry

## Phase 4: Training
- Trainer
- Evaluator (accuracy, F1)
- Checkpointing

## Phase 5: Dense Pipeline
- Full training pipeline
- CLI command

## Phase 6: Benchmarking
- Timing utilities
- Memory metrics

## Phase 7: Pruning Abstraction
- BasePruner
- PruningPlan
- Registry

## Phase 8: Structural Pruning
- Channel pruning
- Layer surgery

## Phase 9: Basic Pruners
- Random
- Global magnitude
- Layer-wise magnitude

## Phase 10: Full Pipeline
- Train → Prune → Finetune → Evaluate

## Phase 11: Multi-run Experiments
- Repeated runs
- Aggregation (mean/std)

## Phase 12: MLflow
- Logging
- Artifacts

## Phase 13: Advanced Pruners
- SNIP
- GraSP
- L1 / Group Lasso
- Movement
- Hard-concrete

## Phase 14: DBLP Adapter
- Implement proper handling

## Phase 15: Reporting
- CSV tables
- Summary reports

## Final Goal
- Clean, modular, extensible research framework
- Ready for RL-based pruning integration

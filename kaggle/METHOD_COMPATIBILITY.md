# Kaggle pilot method compatibility

**Inspection type:** static source audit only; no method was executed.  
**Models inspected:** GCN and GraphSAGE, three layers, equal hidden widths.  
**Runtime status:** unverified until the researcher manually launches seed 101.

| Method | Registered structured support | Decision data | Structural behavior | Pilot status |
| --- | --- | --- | --- | --- |
| `random` | GCN + GraphSAGE | Seeded random scores | Scores first hidden group; cascade applies the kept indices through deeper hidden outputs | Statically compatible |
| `global_magnitude` | GCN + GraphSAGE | Dense weights | Scores first hidden group and its immediate consumer; cascade surgery | Statically compatible; this is the only magnitude baseline |
| `snip` | GCN + GraphSAGE | Training labels/indices and one gradient pass | First hidden-group saliency; cascade surgery | Statically compatible |
| `grasp` | GCN + GraphSAGE | Training labels/indices and second-order autograd | First hidden-group saliency; cascade surgery | Statically compatible; Kaggle memory/runtime remains unverified |
| `group_lasso` | GCN + GraphSAGE | Training labels/indices and one regularized gradient pass | First hidden-group saliency; cascade surgery | Statically compatible; this is pruning-time scoring, not Group-Lasso dense retraining |
| `adaptive_layerwise` | GCN + GraphSAGE | Validation accuracy | Sequential magnitude decisions with monotonic layer order; cascade surgery | Statically compatible; rule-based, not RL |
| `q_learning_tabular` | GCN + GraphSAGE | Validation accuracy | Training episodes plus greedy deployment; local non-monotonic surgery allows unequal hidden widths | Statically compatible; distinct feasible model space |

## Comparison limitation

The fixed baseline implementations pass `pruning.target_sparsity` to first-hidden-group channel selection. Adaptive Layerwise and Q-learning use parameter-count reduction as their rollout stopping coordinate. Therefore the common numeric levels 0.30 and 0.50 are **requested comparison levels**, not proof that every method receives an identical optimization constraint.

Every result must be plotted and compared using recomputed achieved parameter reduction:

```text
1 - pruned_parameter_count / dense_parameter_count
```

The pilot runner records the input-target semantics, surgery mode, final widths, actual parameter reduction, target misses, stopping reasons, and accuracy-budget violations. A method failure aborts the selected condition; it is never silently skipped.

## Context plumbing

SNIP, GraSP and Group Lasso receive `data` and `train_idx`. Adaptive Layerwise and Q-learning receive `data` and `val_idx`. The pruning workflow now also passes the resolved raw `pruning` section into `PruningContext`, making the explicitly configured Group Lasso regularization strength available to its existing scoring code.

No test labels are supplied to a pruner. The evaluator still writes final test metrics as raw reporting artifacts, but the pilot acceptance summary contains validation metrics only.

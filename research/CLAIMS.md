# Scientific claims register

**Status date:** 2026-10-20. Claims here are **not** interchangeable with publication-ready conclusions.

| ID | Statement | Status | Source / why not yet final |
| --- | --- | --- | --- |
| CL-001 | Coursework compared random, global/layerwise magnitude, SNIP and tabular Q-learning on GCN and GraphSAGE with Cora, CiteSeer and PubMed. | DOCUMENTED_HISTORICAL | 2026 coursework, methods/results sections. |
| CL-002 | The coursework reports accuracy-versus-achieved-compression curves and studies accuracy-drop limits 0.05, 0.07, 0.10 for Q-learning. | DOCUMENTED_HISTORICAL | 2026 coursework, figs. 2–9. This experiment is **already done**, not a proposed novelty. |
| CL-003 | Q-learning sometimes stops before reaching requested sparsity; achieved compression varies with the permitted accuracy loss. | DOCUMENTED_HISTORICAL | Coursework, section 10.4–10.5. Need actual run traces and seed-level variation before a general claim. |
| CL-004 | For a comparable allowed accuracy drop, Q-learning compresses better than fixed baselines. | DOCUMENTED_HISTORICAL — **comparative advantage NOT independently established** | Defense presentation, slide 12; inspect underlying plots, baseline protocol, empirical coverage and uncertainty. Do not equate early stopping with superior channel selection. |
| CL-005 | The coursework did not find a substantial, consistent inference-time speedup on its small/medium graph experiments. | DOCUMENTED_HISTORICAL | Coursework conclusion; requires raw latency protocols to generalize beyond tested conditions. |
| CL-006 | The coursework used a **preselected subset of dense model configurations** chosen because they performed well before pruning. | USER_REPORTED | Researcher clarification, 2026-10-20. Recover full search space, selection metric, checkpoints and split identity; do **not** describe them as arbitrary/undertrained models. |
| CL-007 | Current code physically removes hidden channels, and the RL speed proxy equals parameter reduction rather than measured latency. | VERIFIED_CODE | Pinned audited code revision in `RESEARCH_STATE.md`, `src/gnn_pruning/rl/environment.py`. Does not establish performance. |

## Publication claim discipline

For each proposed manuscript conclusion, state (1) its exact quantifier/conditions, (2) matching evidence IDs, (3) what could falsify it, (4) remaining limitations, and (5) wording suitable for the paper.

Avoid without evidence:
- “Q-learning is universally better / statistically non-inferior.”
- “Parameter compression necessarily accelerates GNN inference.”
- “Graph features in the agent state imply cross-dataset generalization.”
- “Previously selected dense models were insufficiently trained.”

## Open verification items (not automatic requests for new experiments)

- Identify the archival CSVs, figures, configs, checkpoints and seed lists underlying figures 2–9.
- Record the architecture-selection protocol for the dense baseline; ensure final test labels were not used for selection.
- Reconcile pre-audit and audited definitions of achieved sparsity.
- Decide which one supported **specific** conclusion will anchor the November paper.

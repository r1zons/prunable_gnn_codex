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

## 2026-10-22 evidence audit update — CODEX_REPORTED_LOCAL

See [local Codex audit summary](audits/2026-10-22_local-codex.md). Original local CSV/checkpoint data were inspected by Codex, **not independently by ChatGPT**. Main corrections:

- CL-002: complete plotted compact data appear to include **n=2 seeds (42, 43), not n=3**; seed 44 is partial. Original plotted mean/SD cannot be described as three-seed estimates.
- CL-004: a claimed Q-learning superiority at matched **achieved parameter reduction** remains unsupported. Historical static and RL `achieved_sparsity` fields were defined differently; high-target RL runs frequently stop short of requested compression.
- CL-003: stopping is heterogeneous; reported totals among 540 Q-learning conditions: target reached 246, agent STOP 178, accuracy threshold exceeded 87, max steps 29. Accuracy budget is a post-action stop rule and may be exceeded, not a strict guarantee.
- CL-006: local candidate grid establishes five architecture choices per dataset in compact runs and a PubMed stress sweep. Full preselection history and validation-versus-test criterion for Cora/CiteSeer remain open. Do not dispute researcher's statement that dense models were preselected based solely on the compact development configuration.
- CL-007: structured static methods use cascade surgery whereas RL can use independent local widths. This design confounds causal attribution to *RL itself*. Historical global/layerwise magnitude are identical in compact structured mode.
- **New CL-008 (CODEX_REPORTED_LOCAL):** Per-architecture/seed shared dense states and splits were checked by tensor/byte hashes across the two completed compact runs. Needs a reproducible manifest to transition to VERIFIED_EXPERIMENT.
- **New CL-009 (CODEX_REPORTED_LOCAL):** Legacy metadata cannot reliably reload 322/540 non-uniform-width Q-learning checkpoints. Current audited serialization is different; historical metrics remain usable as historical evidence, without claiming checkpoint reloadability.

# Research hub — structural pruning of GNNs

**Purpose:** Short entry point for the research behind `prunable_gnn_codex`, so a new ChatGPT, Work, or Codex session can restore context without rereading conversations. This is an evidence index, **not** an automatically validated scientific report.

## Reading order (mandatory)

1. [`RESEARCH_STATE.md`](../RESEARCH_STATE.md) — detailed audited code scope, pinned SHA, limitations, open questions.
2. [`HISTORY.md`](HISTORY.md) — recovered research timeline and prior decisions (secondary historical context).
3. [`CLAIMS.md`](CLAIMS.md) — scientific statements, confidence and evidence needed.
4. [`EVIDENCE.md`](EVIDENCE.md) — historical coursework and experiment provenance.
5. [`publications/sitito_2026/PLAN.md`](publications/sitito_2026/PLAN.md) — current paper's editorial and research tasks.
6. [`AGENTS.md`](../AGENTS.md) — implementation instructions; [`ROADMAP.md`](../ROADMAP.md) — engineering roadmap.

## Research boundaries

- Current implementation: **structural hidden-channel pruning of GCN / GraphSAGE** in a node-classification task. The graph's nodes and edges are not pruned.
- Implemented tabular Q-learning is an **experimental baseline**, not a proven superior method and not a finalized thesis novelty.
- Near-term publication: *Современные информационные технологии и ИТ-образование* / VMK MSU conference (November 2026 target, submission path and deadline to recheck).
- Later, distinct publication candidate: *ПаВТ-2027* on **actual inference performance** of structurally pruned models, potentially including training a compact GNN from scratch. This is a **candidate**, not a finalized scientific result.
- Avoid duplicate publication: the two manuscripts need distinct research questions, experimental evidence, and claims.

## Evidence/status vocabulary

- **VERIFIED_CODE:** inspected at the explicitly pinned commit (not necessarily reproduced experimentally).
- **DOCUMENTED_HISTORICAL:** shown in earlier coursework; raw run provenance may be absent.
- **USER_REPORTED:** methodological fact clarified by the researcher, awaiting artifacts/protocol.
- **VERIFIED_EXPERIMENT:** independently traceable run ID, exact code/config, split, seeds, environment and results.
- **HYPOTHESIS:** testable possibility, not a finding.
- **OPEN:** decision, evidence, or verification not yet complete.

Never silently promote a DOCUMENTED_HISTORICAL, USER_REPORTED or HYPOTHESIS item to VERIFIED_EXPERIMENT.

## Immediate objective: the first paper

1. Recover **which dense models were preselected, by what criterion, and on which data**.
2. Map conclusions and plots of the existing coursework to source runs.
3. Find any claim that cannot be supported, and qualify or test it.
4. Adapt the manuscript to the venue's *current verified* format and review requirements.
5. Add new experiments only to close an identified material evidence gap.

## Updates

Update `CLAIMS.md` and `EVIDENCE.md` when new supporting artifacts appear. Review changes in a branch/PR before merging. Keep large checkpoints, private material and raw logs outside the public Git repository, linking them with stable identifiers and (where useful) hashes. Do not rely on AI-chat memory as the canonical source.

# Research hub — structural pruning of GNNs

**Purpose:** Short entry point for the research behind `prunable_gnn_codex`, so a new ChatGPT, Work, or Codex session can restore context without rereading conversations. This is an evidence index, **not** an automatically validated scientific report.

## Reading order (mandatory)

1. [`RESEARCH_STATE.md`](../RESEARCH_STATE.md) — detailed audited code scope, pinned SHA, limitations, open questions.
2. [`HISTORY.md`](HISTORY.md) — recovered research timeline and prior decisions (secondary historical context).
3. [`CLAIMS.md`](CLAIMS.md) — scientific statements, confidence and evidence needed.
4. [`EVIDENCE.md`](EVIDENCE.md) — historical coursework and experiment provenance.
5. [`publications/sitito_2026/PLAN.md`](publications/sitito_2026/PLAN.md) — current paper's editorial and research tasks.
6. [`experiments/kaggle_pilot_2026/PLAN.md`](experiments/kaggle_pilot_2026/PLAN.md) — approved canonical pilot protocol and execution restrictions.
7. [`AGENTS.md`](../AGENTS.md) — implementation instructions; [`ROADMAP.md`](../ROADMAP.md) — engineering roadmap.

## Research boundaries

- Current implementation: **structural hidden-channel pruning of GCN / GraphSAGE** in a node-classification task. The graph's nodes and edges are not pruned.
- Implemented tabular Q-learning is an **experimental baseline**, not a proven superior method and not a finalized thesis novelty.
- The only immediate objective is a scientifically defensible submission through the VMK MSU conference by the researcher-designated target of **2026-11-02**. The organizer's official deadline and submission path still require independent confirmation.
- Unrelated algorithm development, HPC benchmarking and later publication candidates, including *ПаВТ-2027*, are deferred until after this submission effort.
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

1. Preserve the historical audit and do not treat the two-seed debug benchmark as publication-grade evidence of Q-learning superiority.
2. Prepare the approved Kaggle pilot without mixing its outputs with historical runs.
3. Use the pilot only to verify execution, method compatibility, convergence diagnostics, artifact capture and runtime.
4. Freeze validation-based comparison rules before reviewing final multi-seed test results.
5. Prepare the manuscript and venue materials in parallel; do not defer writing until experiments finish.

## Updates

Update `CLAIMS.md` and `EVIDENCE.md` when new supporting artifacts appear. Review changes in a branch/PR before merging. Keep large checkpoints, private material and raw logs outside the public Git repository, linking them with stable identifiers and (where useful) hashes. Do not rely on AI-chat memory as the canonical source.

## Latest local evidence audit

Read [the 2026-10-22 local Codex audit summary](audits/2026-10-22_local-codex.md) before drafting experimental conclusions. This is locally reported, not independently rerun; in particular, only two seeds are complete and old static/RL sparsity fields have different meanings.

The subsequent decision to run a separate canonical Kaggle pilot is recorded in [the pilot plan](experiments/kaggle_pilot_2026/PLAN.md). It does not upgrade or replace the historical evidence.

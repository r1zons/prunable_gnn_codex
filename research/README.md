# Research hub — structural pruning of GNNs

**Updated: 2026-11-02.** Start here to restore project context without relying on
chat memory. The repository contains different historical protocols and campaigns;
implemented code, accepted jobs, completed outputs and independently checked
scientific claims are distinct states.

## Reading order

1. [Current status](STATUS.md): the opening summary is current; later entries are
   a dated event log, including superseded blockers and submission-only states.
2. [Current article intentions](publications/adaptive_strategies_2026/STATE_AND_PLAN.md):
   new manuscript, adaptive strategy behavior, hypothesis and evidence limits.
3. [Campaign registry](experiments/INDEX.md), then the relevant protocol, manifests,
   receipts, collection/audit and analysis reports. Citation V5 and Flickr V5 are
   separate campaigns; completion of one says nothing about the other.
4. [SITITO release](publications/sitito_2026/RELEASES.md): immutable previous text,
   reusable material for the new article, not the current writing plan.
5. [Historical claims](CLAIMS.md), [historical evidence](EVIDENCE.md),
   [history](HISTORY.md) and [older code snapshot](../RESEARCH_STATE.md).
   Their old open items do not override later verified campaign reports.
6. [Repository instructions](../AGENTS.md) and [engineering roadmap](../ROADMAP.md).

## Current research direction

Compare structural hidden-channel pruning strategies for GCN/GraphSAGE, giving
particular attention to decisions, trajectories and consequences of adaptive
methods. Quality, achieved physical compression and cost remain necessary outcomes.
Additional flexibility helping the method is a hypothesis, not a proven advantage.
Q-learning is one method; Lookahead is implemented; beam/genetic search is not part
of the seven-method V5 matrix. Literature can cover a wider range than experiments.

The graph itself is not sparsified. Parameter reduction is not measured speedup.
Training, pruning and model-selection decisions never use test labels. Preserve
independent Dense seeds versus policy seeds and platform/source provenance.

## Evidence vocabulary

- **ACCEPTED/SUBMITTED:** scheduler receipt, not a completed experiment.
- **COMPLETED:** successful execution and final outputs; state exact evidence.
- **INTEGRITY VERIFIED:** bytes/hashes and recorded metadata agree; does not imply
  independently rerun tensor reconstruction or model inference.
- **INDEPENDENTLY AUDITED:** name the actual checks and limitations of that audit.
- **ANALYZED:** scientific/descriptive analysis exists for that specific provenance.
- **HYPOTHESIS / OPEN:** intentions, associations or missing verification, not facts.

Large checkpoints, raw traces/logs and private access metadata remain under the
adjacent external `prunable_gnn_artifacts` root, outside Git. A clone restores
compact provenance and analysis, not all raw artifacts. Read the per-campaign paths
and hashes before collecting or rerunning anything. Do not overwrite evidence.

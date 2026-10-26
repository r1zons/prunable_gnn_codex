# Independent red-team audit brief — LOCAL-STRUCTURED-V2

**Status:** PREPARED, AUDIT NOT STARTED. This brief is to be used only after researcher has finalized the specification. Audit should be independent of the discussions that generated it.

**Repository:** https://github.com/r1zons/prunable_gnn_codex

**Draft branch (subject to merge):** `research/local-structured-v2-draft-20261026`

**Canonical inputs after final approval:**
1. `research/methodology/LOCAL_STRUCTURED_V2_DRAFT.md` (or successor approved spec).
2. `research/methodology/MASK_SCORING_AND_EQUIVALENCE.md`.
3. `research/backlog/IDEAS.md` (deferred, NOT mandatory article scope).
4. Current checked-out `src/gnn_pruning/pruning/methods.py`, `src/gnn_pruning/rl/environment.py`, `src/gnn_pruning/surgery/structural.py`; matching unit tests and configs.
5. Repository evidence ledgers `RESEARCH_STATE.md`, `research/README.md`, `research/CLAIMS.md`, `research/EVIDENCE.md`, latest `research/audits/` records.
6. Original scientific sources for SNIP, GraSP, Magnitude, structural pruning, and RL pruning. Check original method definitions against current adapted naming and comparisons.

## Work assignment (read-only, no code changes)

Independently review the draft as if preparing a hostile but constructive scientific and engineering peer review. Do not use the prior ChatGPT dialogue as evidence that a choice is scientifically correct. Do not silently reinterpret user approvals as independently verified claims. Identify contradictions, missing definitions, implementation gaps, infeasible configurations, unfair comparisons and risks of false scientific claims.

Prioritize:
- Whether full-mask score defined from unique physically disappearing parameters is mathematically compatible with each method (magnitude, SNIP, signed GraSP, Random). Investigate bias, root weights and coupling of two hidden masks.
- Whether one fixed per-layer channel ranking per width tuple is sufficient for stated claims; provide a counterexample if joint mask interaction matters. Check prefix-sum derivation and complexity.
- Whether correct dense mask-based forward is equivalent to physically rebuilt local PyG GCN/SAGE models; demonstrate how incorrect bias/weight masks violate equivalence.
- Whether Grad/Hessian expressions and GraSP score sign match the cited paper; whether post-training without fine-tuning invalidates or merely changes the scientific interpretation.
- Whether random baseline selection silently benefits from score-oracle search or wider architecture search.
- Whether reachable final width pairs and target tolerance ±1 percentage point exist for each dataset/model; demonstrate cases where they do not, including short horizon and k=1,2,4,8.
- Whether terminal overshoot and symmetric terminal penalties are consistent across greedy and Q STOP, including reward double counting, target-reaching auto-stop in existing code and STOP eligibility.
- Whether greedy testing every action with validation vs Q testing one action creates unequal information/computation budgets; propose minimally invasive reporting/controls.
- Whether validation/test split discipline, dropout scoring RNG, signs, ties, trainable bias treatment, hardware/torch-pyg portability, checkpoint equality, and zero-finetuning policy are sound.
- Whether method IDs/labels make scientifically supported claims (do not call legacy GraSP canonical or current group-lasso full regularization).
- Whether a separated V2 parallel code path can preserve historical V1 results and reproducible CI.
- Whether overplanned ablations/datasets create unacceptable deadline risk (paper priority).

## Required output

1. Executive determination: READY / READY WITH BLOCKERS / NOT READY.
2. Critical, major, minor findings table. Each finding: exact file/line or spec section, falsifiable reproduction or mathematical counterexample, scientific consequence, smallest viable correction, and whether it blocks publication experiments.
3. Contradiction matrix: explicit user decision vs specification vs current code vs scientific source, with evidence confidence.
4. Minimal scientific protocol and implementation acceptance checklist, with prioritized tests.
5. Suggested *minimal* changes to preserve reproducibility and deadline; separately label deferred enhancements.
6. Exact references/DOIs/links for claims; no fabricated citations or claims of experimentally verified results.
7. No code changes, PRs, or execution of full experiments without a new explicit request.

If inputs are still DRAFT, review them as DRAFT and report OPEN items rather than making decisions on the user's behalf.

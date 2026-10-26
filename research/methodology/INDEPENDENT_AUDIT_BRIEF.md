# Independent red-team audit brief — LOCAL-STRUCTURED-V2

**Status:** READY FOR INDEPENDENT REVIEW; AUDIT NOT STARTED.  
**Repository:** https://github.com/r1zons/prunable_gnn_codex  
**Working branch:** research/local-structured-v2-draft-20261026  
**Canonical protocol:** research/methodology/LOCAL_STRUCTURED_V2_SPEC.md (2026-10-26)  
**Read-only research/code audit. Do NOT edit code, run experiments, or merge.**

## Canonical reading order

1. **research/methodology/LOCAL_STRUCTURED_V2_SPEC.md** — primary researcher-approved plan for independent audit; supersedes prior V2 draft on points where they differ.
2. research/methodology/MASK_SCORING_AND_EQUIVALENCE.md — mask score, dense masked forward vs physical local surgery, prefix sums.
3. research/backlog/IDEAS.md — strictly deferred research ideas, including Cascade vs Local ablation, *not obligatory* in present article.
4. Current checked-out src/gnn_pruning/pruning/methods.py; src/gnn_pruning/rl/environment.py; src/gnn_pruning/surgery/structural.py; models, configs, tests and scripts tied to the exact reviewed SHA.
5. Existing RESEARCH_STATE.md, research/README.md, research/CLAIMS.md, research/EVIDENCE.md, research/audits/ and research/experiments/kaggle_pilot_2026/PLAN.md as **historical context**, NOT current V2 experimental instructions.
6. Original scientific works for SNIP, GraSP, structural pruning, and RL pruning, plus official PyTorch and Torch-Pruning docs as implementation reference only.

**Non-negotiable correction:** the current V2 *minimum* datasets are Cora, CiteSeer and PubMed, with optional more datasets if time/resources permit. The computing platforms are **local workstation and POLUS**. **Kaggle and Colab are not planned.** Any Kaggle-only instructions in old files are superseded, not a license to edit or delete historical documents.

## Mission

Conduct a genuinely independent hostile-but-constructive scientific and engineering review. Do not treat the preceding ChatGPT discussions or author's intentions as proof of correctness. Identify contradictions, mathematical mistakes, fairness defects, computational cost, lack of reproducibility, missing operational definitions, and incorrect claims of equivalence.

Prioritize the following checks:

1. Full-mask score from unique physically removed parameters for Magnitude, SNIP and *signed* GraSP, including bias and two-branch SAGEConv; is the proposed ranking/score search well-defined? Prove or refute direction of GraSP score.
2. Validity of one channel ranking per layer, followed by search across target width pairs; construct counterexamples for joint channel interactions; evaluate bias from selecting the lowest compression within a ±1 pp band.
3. Prefix-sum formula versus direct full-mask score; whether masked dense logits match local physically rebuilt PyG GCN/SAGE logits with bias and activation/graph semantics. Mask-only output must never be reported as actual structural compression.
4. HVP formula exactly: original GraSP vs legacy H*W+g computation, sign lost by absolute value; critically assess post-training/no-finetune motivation for GraSP and SNIP.
5. Random baseline: probability law for feasible width pairs and retained channels, no importance oracle; compare method-specific architecture selection opportunities fairly.
6. Exact attainable compressions for GCN and GraphSAGE on intended F,C of Cora/CiteSeer/PubMed; unreachable 0.30/0.50 ±0.01; feasibility and physical parameter counts, minimum width four; account for graph model bias tensors.
7. Q-learning: explicit STOP not forced at target, max 20; old environment auto-terminates; identify code changes needed for new version. Step actions k in {1,2,4,8} and greedy allowed returns to both hidden layers.
8. Terminal score P=-lambda max(0,|s_T-s*|-eps): effects on early STOP, repeated accuracy drop D_t, existing terminal bonuses, lack of strong overshoot constraint in greedy and ability of Q to exceed target; check internal scientific consistency.
9. Fairness between adaptive greedy (up to eight validation evaluations per step), Q-learning (many training episodes) and statics (no validation per mask); log and report whole search budget and expected accuracy confidence.
10. Dataset splits, seed, determinism, train/eval/dropout for gradient score, checkpoint hash equality across local and POLUS. CPU/GPU numerical differences and Python/PyTorch/PyG compatibility; **never pool inference latency across platforms**.
11. New identifiers separate from legacy; Group Lasso removed from V2 core, legacy GraSP not mislabeled as original; pretraining SNIP and GraSP called post-training structural *adaptations*.
12. Publication deadline: minimum three datasets, two models, 30/50% targets, five mandatory pruners and GraSP conditional. Optional larger datasets cannot delay minimum credible publication.
13. Identify contradictions between old Kaggle documents and new local/POLUS plan, recommending provenance-safe minimal corrections (no silent retroactive editing of old evidence).

## Required report (write to a new research/audits/ file in an audit-only documentation branch or return text; do not modify code or configs)

- Overall verdict: READY / READY WITH BLOCKERS / NOT READY.
- Severity-ranked critical/major/minor findings, **each with exact file/section/code line**, proof/counterexample, consequence, minimal fix, experiment-blocking status.
- A contradiction matrix: researcher-approved spec vs present code vs original literature, specifying confidence and what is actually verified.
- Minimal acceptance-test checklist, scientific-protocol gate, runtime/compute feasibility on local and POLUS.
- Immediate blockers vs deferred suggestions, prioritized for earliest publication.
- Verified original citations/DOIs/URLs; no invented evidence or test results.
- Confirm exact Git SHA used for review.

Do not auto-fix scientific choices or infer author consent to new experiments. If the spec remains labeled "ready for audit", review it as draft and return required decisions for author approval. The source code and experiment outputs MUST NOT be modified or executed without a new explicit request.

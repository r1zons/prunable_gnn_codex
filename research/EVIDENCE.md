# Evidence and provenance register

This index records **what exists and how strongly it supports a claim**. It does not upgrade coursework figures to independently reproduced results.

## Sources

| ID | Material | Accessible location | Evidence status |
| --- | --- | --- | --- |
| DOC-001 | 2026 46-page coursework: *Исследование адаптивного структурного прунинга графовых нейронных сетей для оптимизации инференса на основе базовых методов обучения с подкреплением* | Provided in project conversation; **do not assume it is a public/repository file** | DOCUMENTED_HISTORICAL |
| DOC-002 | VMK MSU defense slides, 18 May 2026, 23 slides | https://docs.google.com/presentation/d/1EW7GTrHqamO_lQiv3oC6a_ACqjDy48mCgsDBe7I5ih0/edit | DOCUMENTED_HISTORICAL |
| CODE-001 | Audited structural-pruning implementation | https://github.com/r1zons/prunable_gnn_codex/tree/882b3c97385a43361e417ff02f8e6854899ede85 | VERIFIED_CODE; not a reproduced benchmark |
| DOC-003 | Detailed code/evidence inventory | ../RESEARCH_STATE.md | Source document, updated separately |

## Historical experiments — provenance worklist

| ID | Experiment family | Existing source | Missing before `VERIFIED_EXPERIMENT` |
| --- | --- | --- | --- |
| EXP-COURSE-01 | Accuracy versus achieved sparsity, 3 datasets × GCN/GraphSAGE × static/Q-learning methods | DOC-001 figs. 2–7 | Original plot data / CSV, model configs, seed lists, checkpoint provenance, exact split and evaluation stage |
| EXP-COURSE-02 | Accuracy-drop budget and achieved sparsity for Q-learning | DOC-001 figs. 8–9; DOC-002 slides 6–12 | Validation/test split used for decisions, traces, comparison definitions, seed variability |
| EXP-COURSE-03 | Dense-model preselection before pruning | Researcher's 2026-10-20 clarification | Search space, optimization/selection metric, train logs, selected run IDs and checkpoints |
| EXP-COURSE-04 | Inference latency and parameter reduction | DOC-001 conclusion; DOC-002 supplementary slides | Raw timings, device, warm-up, synchronization, run-to-run variation and scope of timed operations |
| EXP-FUTURE-01 | Pruned GNN vs compact GNN trained from scratch | Candidate for ПаВТ-2027 | Not executed; first define matched size, training budgets and inference measurement |

## Required provenance manifest for newly run experiments

Each new run should be identifiable by:
- run ID; Git commit SHA; resolved YAML config and checksum;
- dataset version, split IDs, architecture and dense checkpoint ID;
- seed; Python / PyTorch / PyG / CUDA versions; device and driver;
- algorithm, selected hyperparameters and selection dataset;
- raw accuracy/F1, requested and achieved **parameter** sparsity, channel diagnostics, pruning/training overhead and latency protocol;
- full artifact locations, checksums where appropriate and analysis script/commit.

Keep historical and post-audit runs separate. If raw data cannot be recovered, label the conclusion *historical* or rerun a **small, explicit** verification experiment; do not invent the missing run properties.

## Local Codex audit received 2026-10-22

The researcher supplied a read-only audit from a Codex session with local filesystem access: [summary](audits/2026-10-22_local-codex.md). Findings are **CODEX_REPORTED_LOCAL**, not independently verified in this GitHub repository. The full source CSVs and checkpoints remain local; the summary does not contain copies of private artifacts.

- Historical full compact runs: seed 42 and seed 43 only; third seed 44 incomplete (39/60 pipelines, missing PubMed).
- `exports/compact_pruning_benchmark_artifacts/combined_pipeline_results.csv` and `exports/compact_pruning_benchmark_plots/plot_summary_by_dataset_model_method.csv` support reconstruction of plotted historical results.
- Historic metric definition conflict: static achieved sparsity (channel ratio) versus Q-learning achieved sparsity (parameter ratio). **Keep originals unchanged**, recompute derived tables.
- No canonical independent `n>=3` repeat series yet. Original `RESEARCH_STATE.md` code SHA is an audited code reference, not necessarily the SHA of historical runs.
- Older checkpoint metadata may not encode non-uniform widths and should not be represented as reliably reloadable artifacts.

**Next artifact:** per-figure row/run manifest and compression-normalized historical reanalysis. It will determine whether a *small* new canonical benchmark is actually needed.

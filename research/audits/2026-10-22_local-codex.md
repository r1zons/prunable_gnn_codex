# Local Codex evidence audit — 2026-10-22

**Status: CODEX_REPORTED_LOCAL, not independently re-run from raw artifacts in ChatGPT.** The researcher provided the read-only Codex High audit report on 2026-10-22. The detailed audit and underlying raw logs remain on the researcher's local machine. Do not cite this document as an independent reproduction of numerical results.

## Verified by local Codex according to its report

- Compact suite intended seeds 42, 43, 44. Seeds **42 and 43 complete** (60 pipelines per run). Seed **44 partial**, 39 completed pipelines, none for PubMed. Existing coursework plots/export aggregate 42+43, so **n=2**, not three completed replicates.
- The completed runs use **matching dense model-state hashes and byte-identical splits** for all compared methods in each dataset/architecture/seed group.
- Historical static `achieved_sparsity` = channel-removal fraction, while Q-learning = parameter-count reduction. Reported mean absolute numeric gap ~0.0025, maximum ~0.0149. Normalize using `1 - params_pruned / params_dense` for all methods before any new plot.
- Historical post-prune test results can be traced to CSV/JSON and Q-learning deployment traces; originals remain in local `runs/`, `exports/`.
- Historical global and layerwise magnitude methods are **behaviorally identical in structured compact runs**; do not count them as two independent structural mechanisms.
- Static runs use a cascade structural pathway while Q-learning local surgery may set independent hidden widths. A between-method difference cannot on its own establish an advantage due to RL learning.
- Q-learning uses validation-based actions, dense-model resets for training episodes, and greedy deployment. `speed_proxy` is parameter reduction, not measured hardware latency.
- Q-learning accuracy-drop budgets are **post-action termination thresholds**, not guaranteed feasibility constraints: locally reported overshoots total **87/540** (40 for budget .05, 31 for .07, 16 for .10).
- Stopping outcomes across 540 post-prune Q-learning rows: target reached 246, agent STOP 178, post-action accuracy drop 87, maximum steps 29. No requested target .8 or .9 was reached in these exported runs.
- Historical Q-learning checkpoint metadata lacks independent per-layer widths; 322/540 final models reportedly have non-uniform widths and are not generally reloadable via the legacy metadata. Current audited code has improved serialization; do not retroactively represent legacy checkpoints as reloadable.
- In the PubMed stress subset, dense checkpoints were chosen via validation-loss early stopping and candidate summaries show validation accuracy. The **complete architecture preselection grid**, including Cora/CiteSeer, could not be located. Validation best-epoch 29/30 in the stress study is a convergence uncertainty, **not proof** of poor training.
- Historical plotted budget analyses contain some **test-derived post-hoc filtering**; inspect exactly where test information influenced figures versus actual choice of policies/models. Do not use final test labels to guide new selection.

## Scientific interpretation

**Defensible now as an exploration:** tabular Q-learning provides an implementable adaptive, structurally compact GNN and exhibits heterogeneous early-stopping and accuracy/compression trade-offs across tested architectures.

**Not yet defensible:** statistically robust superiority, superior compression under equally achieved parameter sparsity, strict quality-budget guarantees, hardware-aware or measured-latency optimization, cross-graph generalization, or fully reproducible old checkpoints.

Avoid the headline "RL is better than static pruning" until the design matches model size, architecture and action-space flexibility appropriately.

## First-paper triage

P0 (before writing main comparative conclusions):
1. Find original CSV and figure IDs used in the actual submitted coursework; preserve read-only legacy artifacts.
2. Recompute historical **derived analysis only** with a uniform parameter-count-based compression metric, separating per-seed runs and reporting n=2 honestly.
3. Test whether the existing observations survive compression-matched comparisons (exclude out-of-budget runs when claiming feasibility; if interpolation is used, document its assumptions and never extrapolate).
4. Recover whether dense model **selection** used validation-only information. Separate this from within-run checkpoint selection, which is validation-loss-based.
5. Confirm conference registration/submission deadline and authorship; researcher had not registered when asked.

P1 (only if needed for stronger numerical claims):
6. On current code, use a deliberately small, paired, preselected set of architectures with >=3 **complete** independent seeds, matching splits/checkpoints and validation-only selection.
7. Document architecture search cost and action-space asymmetry; ideally add a no-learning baseline in the same local-surgery action space if claiming an RL-specific effect.

P2 (after first submission):
8. Restore reliable environment + checkpoint round-trip provenance for new experiments; consider timing and compact-from-scratch comparison as a scientifically **distinct** PaVT-2027 line.

## Locally reported paths

- `runs/compact_pruning_benchmark/run_{000,001,002}/`
- `exports/compact_pruning_benchmark_artifacts/combined_pipeline_results.csv`
- `exports/compact_pruning_benchmark_artifacts/artifact_manifest.csv`
- `exports/compact_pruning_benchmark_plots/plot_summary_by_dataset_model_method.csv`
- `exports/compact_pruning_benchmark_plots/qlearning_summary.csv`
- `runs/pubmed_stress_sweep/`
- `configs/suites/compact_pruning_benchmark.yaml`

**Next:** Ask local Codex for a read-only (or derived-output-only) per-figure/per-seed provenance manifest and reanalysis. Do not start a full benchmark re-run before that result is reviewed.

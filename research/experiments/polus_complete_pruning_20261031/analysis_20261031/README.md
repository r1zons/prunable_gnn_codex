# Offline scientific analysis — 2026-10-31

Read [SCIENTIFIC_ANALYSIS.md](SCIENTIFIC_ANALYSIS.md) first. Supporting reports:
[Lookahead case](LOOKAHEAD_CASE_STUDY.md), [native Q diagnostics](Q_LEARNING_DIAGNOSTICS.md),
[H1–H4](HYPOTHESIS_EVALUATION.md). [FIGURES](FIGURES/) contains eight figures in
vector PDF and 250 dpi PNG. Raw evidence remains outside Git.

The analysis branch starts at audit commit
`8f24513da09c0291a562fbb31a688da2698671b9`; experimental executable source is
`55d0fc242b543c5725276144bd8e970eedc48ecd`. Exact input, output, code and test hashes
are in [ANALYSIS_MANIFEST.json](ANALYSIS_MANIFEST.json). The containing Git commit
identifies the analysis revision; a commit cannot contain its own hash.

## Reproduce from the canonical checkout

These commands only read saved CSV/JSON/traces. They never instantiate a GNN,
run inference, update Q values, train, prune, contact POLUS or measure model time.
Use Python 3.12.9 for the recorded table/test reproduction; figures additionally
used Matplotlib 3.9.4 and NumPy 1.26.4. No packages were installed for this task.
The report renderer also ran successfully with Python 3.14.3.

Set `GNN_ARTIFACT_ROOT` to the existing adjacent `prunable_gnn_artifacts` directory.
Choose a **fresh** output directory. The table generator refuses to overwrite a
completed analysis; reports and plots can be regenerated there.

```sh
campaign=research/experiments/polus_complete_pruning_20261031
analysis="$campaign/analysis_20261031"
collection="$campaign/collection"
raw="$GNN_ARTIFACT_ROOT/polus-complete-collection-20261031T110939Z/raw"
fresh=/private/tmp/polus-scientific-reproduction-20261031

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m gnn_pruning.experiments.complete_scientific \
  --collection "$collection" --raw "$raw" --output "$fresh"
python "$analysis/scripts/reports.py" --collection "$collection" --analysis "$fresh"
python "$analysis/scripts/figures.py" --collection "$collection" --analysis "$fresh"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest tests/test_complete_scientific.py -ra
```

The table generator rehashes all 1,226 raw files and all 19 audited deliverables,
matches all 252 rows against the existing audit, and checks pairing/input identity.
The successful final test log is preserved externally at
`polus-complete-scientific-analysis-20261031T121639Z/focused-tests-release.log`:
**19 PASS / 0 FAIL / 0 SKIPPED**. No model-dependent full suite was run in this
analysis task; earlier audit-suite results are historical evidence only.
Two independent Python 3.12 table generations produced 12 byte-identical outputs.
Development drafts/logs remain external and are not substituted for the final log.

Verify the committed compact deliverables, code and primary input hashes:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python "$analysis/scripts/manifest.py" \
  --repo . --analysis "$analysis" --verify-only
```

This verification checks the stored hashes without accessing raw checkpoints.
Re-running the table generator additionally verifies the complete raw inventory.
The manifest can be regenerated with `--raw "$raw" --logs PATH_TO_ANALYSIS_LOG_ROOT`
and `--repeat-a PATH --repeat-b PATH` for the two preserved table generations;
it then rechecks the complete raw archive and source archive as well.

## Table definitions

- `METHOD_SUMMARY.csv`: 84 dataset/model/target/method cells, exactly three dense
  seeds per cell. Quality and compression are fractions; paired differences below
  are percentage points. `*_sd` is sample SD, not SE or a confidence interval.
- `PAIRED_COMPARISONS.csv`: all 756 individual B−A contrasts, paired by the same
  checkpoint, graph and split; ratios are B/A. Warnings/no-ops remain present.
- `PAIRED_SUMMARY.csv`: 252 method-pair cells with n=3, SD and seed directions.
- `PARETO_DIAGNOSTICS.csv`: all 252 outcomes, with separate all-outcome and
  physically pruned/in-band frontiers; ineligible flags remain missing.
- `GREEDY_LOOKAHEAD_TRAJECTORIES.csv`: 36 matched final-mask/action comparisons.
- `GREEDY_PREVIEW_BRIDGE.csv`: 72 Greedy decisions corroborated from existing
  Lookahead previews at identical concrete parents, including nonselected branches.
  This extends the audit's standalone-Greedy coverage without changing the audit.
- `Q_RUN_DIAGNOSTICS.csv`: 36 Q deployments. `Q_DEPLOYMENT_DECISIONS.csv`: 720
  recorded decisions reconstructed from final Q values. `Q_EPISODE_SUMMARY.csv`:
  1,080 process records, never additional statistical replicas.
- `Q_TARGET_PAIR_CHECK.csv`: 18 target-pair identities, with Q tables compared as
  state-to-action maps independent of JSON serialization order.
- `LOOKAHEAD_CASE_EVIDENCE.json`: selected recorded traces/paths only;
  `ANALYSIS_RESULTS.json`: validation and diagnostic totals.

Blanks/nulls mean unmeasured or undefined; zero means an observed zero. No test
outcome enters new comparisons, plots, hypotheses, case selection or ranking.
Historical V2/V3/V4 reports are context only and supply no primary observations.

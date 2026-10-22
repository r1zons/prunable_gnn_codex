# Kaggle seed-101 pruning pilot

This directory prepares, but does not start, `EXP-CANONICAL-01`. The researcher must manually launch every Kaggle run.

Read [`METHOD_COMPATIBILITY.md`](METHOD_COMPATIBILITY.md) before execution. It records the statically verified method requirements and the non-equivalent baseline/Q-learning structural action spaces.

## Before upload

1. Review the local documentation, implementation and notebook-pin commits.
2. Push them to `r1zons/prunable_gnn_codex` only after explicit approval.
3. Verify that implementation commit `e574f4043135b011e0f86163f5481b36947945d4` is publicly fetchable from GitHub.

The notebook is pinned to the implementation commit above and verifies the checked-out SHA before invoking the pilot runner. The later notebook-pin commit changes only launch documentation and the embedded SHA.

## Kaggle setup

1. Create a new Kaggle Notebook and enable Internet access so GitHub and PyG dependencies can be downloaded.
2. Select a GPU accelerator if available. CPU execution remains supported but may be slower.
3. Upload or import `kaggle/prunable_gnn_pilot.ipynb`.
4. Set `SELECTED_CONDITIONS` in the first code cell. Use one condition for independently saved outputs or `all` for sequential unattended execution.
5. Manually choose **Save Version** and **Save & Run All**.

Valid independent conditions:

- `cora_gcn_l3_h32`
- `cora_graphsage_l3_h32`
- `citeseer_gcn_l3_h32`
- `citeseer_graphsage_l3_h32`
- `pubmed_gcn_l3_h32`
- `pubmed_graphsage_l3_h32`

The notebook clones the pinned commit into `/kaggle/working/prunable_gnn_codex`, installs only the pilot dependencies, installs the repository editable without dependency re-resolution, records `pip freeze`, and runs `scripts/run_kaggle_pilot.py`.

`mlflow` is deliberately omitted from the pilot dependency file because no source module imports it and the pilot does not use MLflow. The exact Kaggle package set is captured in `pip_freeze.txt`; PyTorch is supplied by the selected Kaggle image and is not replaced.

## Retrieval

Download `/kaggle/working/prunable_gnn_kaggle_pilot_results.zip` and its `.sha256` file from the Notebook output after completion or failure. Clone, dependency, pipeline, and artifact-validation failures all trigger a best-effort archive of the logs and artifacts produced so far.

The runner never overwrites a condition directory. A failed condition keeps completed variant artifacts and writes `failure.json`, `partial_results_inventory.csv`, and `artifact_manifest.csv` before returning an error. Download the partial archive for diagnosis, then move the failed condition aside before a deliberate retry.

`validation_summary.csv` reports both requested and recomputed achieved parameter reduction. It labels a requested-target comparison `matched` only when the achieved-reduction span across methods is at most 0.025; otherwise it labels the comparison `unmatched` without invalidating or deleting any method result. A disagreement between a reported achieved reduction and `1 - pruned_parameters / dense_parameters` remains a fatal data-integrity error.

## Selection boundary

The seed-101 validation summary is for compatibility, convergence, artifact and runtime review only. Raw pipeline files may contain test metrics because the existing evaluator reports all splits, but the runner's gate summary excludes test metrics. Do not select a method or amend the protocol from pilot test rankings.

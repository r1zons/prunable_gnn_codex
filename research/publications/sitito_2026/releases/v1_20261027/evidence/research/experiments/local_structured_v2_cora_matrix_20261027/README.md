# LOCAL-STRUCTURED-V2 Cora functional smoke matrix — 2026-10-27

**Verdict: GO to a separately scoped validation-only Q-learning pilot.**
All **18 new** model/method/target combinations passed independently verified
physical-checkpoint checks. Together with the six unchanged GCN30 records from
[the initial smoke](../local_structured_v2_smoke_20261027/README.md), the functional
matrix covers **24/24** cells. GCN30 was **not rerun**. There were no blocked cells,
implementation fixes, source/test changes, or algorithm/methodology changes.

This is a bounded CPU functional test. Its settings and metrics are diagnostic,
not approved scientific hyperparameters, confirmatory results or paper evidence
of superiority. GO qualifies the runtime/artifact path for the validation-only
pilot; it neither executes that pilot nor establishes Q-convergence or target-hit.

## Runtime, code and scope

- Existing branch: `implementation/local-structured-v2-20261027`, draft PR48.
- All new CLI runs record tested SHA `11ab841b4960d9591dd13feb240b7b7e5639c6a3`
  with `tracked_worktree_dirty=false`. Code under `src/`, tests and methodology
  is unchanged. The evidence commit follows that tested SHA.
- Existing interpreter: `/private/tmp/gnn-v2-validation-py312-20261027/bin/python`;
  Python **3.12.9**, PyTorch **2.2.2**, PyG **2.6.1**, pytest **8.3.5**,
  NumPy **1.26.4**, local Mac CPU. No environment created or modified.
- Each CLI has `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`,
  `OPENBLAS_NUM_THREADS=2`; verification uses two Torch CPU threads.
- New runs only: GCN50, GraphSAGE30 and GraphSAGE50, six methods each.
  No dense training, fine-tuning, dataset download, server access/LSF submission,
  hyperparameter tuning, inference benchmark, V1 change or new CI.

## Diagnostic settings

All runs use Cora, three conv layers with hidden widths32/32, dropout0.5,
seed101, CPU, min4 hidden channels, target tolerance0.01.
The fixed diagnostic values are **lambda2.0, max_steps3, episodes3, gamma0.9,
alpha0.1, epsilon0.2**. No result was used to tune or exclude a method.
Exact byte-identical successful inputs are in [configs/](configs/).

## Frozen checkpoint and graph identity

Both selected dense checkpoints are archived **POLUS-CPU-101-C4-V2 seed101**
artifacts, executed here on the local CPU as a new diagnostic provenance group.
Neither local/POLUS results nor V1/V2 protocols are silently pooled.

| Dense model | Parameters | Archived job | Dense validation accuracy | Dense validation macro-F1 |
|---|---:|---:|---:|---:|
| gcn 1433→32→32→7 | 47175 | 1598695 | 0.8983364140 | 0.8852755860 |
| graphsage 1433→32→32→7 | 94279 | 1598696 | 0.8835489834 | 0.8671309679 |

- **gcn** checkpoint: `/Users/perfect/Desktop/code/MSU/MSU_SQI_25_DIPLOMA/test_codex/prunable_gnn_artifacts/polus-pilot-20261024/raw/results/pilot-cpu-101-c4-v2/jobs/cora_gcn_l3_h32/seed_101/cora_gcn_l3_h32/dense_checkpoint.pt`
  - Checkpoint SHA256: `276b2ff0f7790a84afc2755e89d88f1a170865a4397e4355daf7552c7075c129`.
  - Split: `/Users/perfect/Desktop/code/MSU/MSU_SQI_25_DIPLOMA/test_codex/prunable_gnn_artifacts/polus-pilot-20261024/raw/results/pilot-cpu-101-c4-v2/jobs/cora_gcn_l3_h32/seed_101/cora_gcn_l3_h32/splits.yaml`.
  - Split file SHA256: `59a0209a8e2a73a5e70a720d327f8c024d36d184a6d00be820ced0e1a3eb08bc`; ordered indices hash: `942a10b128f7138e118b3f2e2f96eb87a62f856c10d82010ac995e40e2b4c675`.
- **graphsage** checkpoint: `/Users/perfect/Desktop/code/MSU/MSU_SQI_25_DIPLOMA/test_codex/prunable_gnn_artifacts/polus-pilot-20261024/raw/results/pilot-cpu-101-c4-v2/jobs/cora_graphsage_l3_h32/seed_101/cora_graphsage_l3_h32/dense_checkpoint.pt`
  - Checkpoint SHA256: `09b289dcbaddc1afb1c3660f7fa5d60fb1109c3c3059bdae14d8c4de03034774`.
  - Split: `/Users/perfect/Desktop/code/MSU/MSU_SQI_25_DIPLOMA/test_codex/prunable_gnn_artifacts/polus-pilot-20261024/raw/results/pilot-cpu-101-c4-v2/jobs/cora_graphsage_l3_h32/seed_101/cora_graphsage_l3_h32/splits.yaml`.
  - Split file SHA256: `59a0209a8e2a73a5e70a720d327f8c024d36d184a6d00be820ced0e1a3eb08bc`; ordered indices hash: `942a10b128f7138e118b3f2e2f96eb87a62f856c10d82010ac995e40e2b4c675`.

For each model independently, checkpoint/split/resolved config/training metrics,
prepared config/provenance and environment hashes match its job artifact manifest;
that manifest is byte-identical to the original frozen results archive. Successful
job logs1598695/1598696 match the frozen evidence archive and the independently
isolated `data-by-condition/pilot-cpu-101-c4-v2/<condition>` training roots.
Both models reload strictly with F1433, hidden32/32, C7,3 layers, dropout0.5;
SAGE uses mean aggregation, root weights, no project or output normalization.
Dense validation metrics reproduce their respective archived metrics.
The GCN source is exactly the checkpoint used for the initial GCN30 smoke.

The existing manifest-backed Cora preparation is reused **without rewriting it**:
- Data root: `/Users/perfect/Desktop/code/MSU/MSU_SQI_25_DIPLOMA/test_codex/prunable_gnn_artifacts/local-structured-v2-smoke-20261027/verified_data`.
- Graph tensor hash: `2b2bc996be89812f80ce7d09599dab983f804f87ddd37a09b3ce0b5689ec9a0d`.
- Processed cache SHA256: `7449c091f060648c48c46d7d092246b56d22f4c66ae2bcfff0cf1f8dd01d8c29`.

All eight raw Cora files were rechecked against the recorded raw manifest,
which matches the original source bundle. Input archive hashes were rechecked
against the registry. The graph has x[2708,1433], y[2708], edge_index[2,10556],
original node order and historical PyG2.4 **row-sorted** edges. Installed PyG2.6.1
would column-sort freshly processed edges; the already verified historical cache
is used, not regenerated. Both frozen split signatures match the same disjoint,
complete train/val/test partition. No frozen split or graph tensor was changed.

**Provenance limitation:** neither original dense run recorded a contemporaneous
processed-tensor digest. Identity is reconstructed from recorded raw hashes,
verified preparation/job/config lineage and deterministic historical processing;
it is not a claim that such an old tensor digest exists. Local GCN/GraphSAGE
seed101/202/303 checkpoints were inventoried and file-hash checked, but their
historical graph identity is not independently attested; none was silently
substituted. See [provenance.json](provenance.json) for every path/hash and limitation.

## Results and measured costs

[summary.csv](summary.csv) contains all24 cells with the requested columns plus
artifact/config/log paths and `executed_this_task`. Its six GCN30 rows are reused
previous evidence; [summary_new.csv](summary_new.csv) contains only the18 new runs.
All accuracies/compressions are fractions in CSV/JSON; values below are percentages.
`successful_deletions` is empty for static methods because it is an adaptive-step
counter, not the number of removed channels. Static search validation evaluations
are0; this excludes the separate four dense/final validation/test reporting calls.
Adaptive validation evaluations include search resets, candidates and Q episodes.

`search_seconds` measures the existing CLI's **plan** phase. Dense importance
scoring and final materialization are separate columns; adaptive plan includes its
internal scoring, surgeries and validation calls. `cli_seconds` includes Python
imports, data loading, search, checkpoint writes and post-search reporting.
Neither measure is inference latency, and this smoke makes no speed-ranking claim.

### New runs (all exit0, PASS)

| Model | Target % | Method | Achieved % | Val acc % | Val macro-F1 % | Terminal | Deletions | Val evals | Search s | CLI s |
|---|---:|---|---:|---:|---:|---|---:|---:|---:|---:|
| gcn | 50 | random_local_v2 | 50.0774 | 81.3309 | 77.6009 | static_plan_complete | — | 0 | 0.0356 | 5.999 |
| gcn | 50 | magnitude_local_v2 | 49.7213 | 81.3309 | 72.8138 | static_plan_complete | — | 0 | 0.0663 | 4.384 |
| gcn | 50 | snip_pt_local_v2 | 49.8739 | 77.0795 | 66.3663 | static_plan_complete | — | 0 | 0.0640 | 4.506 |
| gcn | 50 | grasp_pt_local_v2 | 50.2809 | 43.2532 | 20.7872 | static_plan_complete | — | 0 | 0.0641 | 4.320 |
| gcn | 50 | adaptive_greedy_local_v2 | 49.7213 | 81.3309 | 72.8138 | target_reached | 2 | 17 | 0.2632 | 4.873 |
| gcn | 50 | q_learning_local_v2 | 18.6455 | 90.0185 | 87.9656 | max_steps | 3 | 14 | 0.1561 | 4.842 |
| graphsage | 30 | random_local_v2 | 29.2090 | 64.3253 | 67.7574 | static_plan_complete | — | 0 | 0.0359 | 4.479 |
| graphsage | 30 | magnitude_local_v2 | 29.0149 | 81.3309 | 75.8155 | static_plan_complete | — | 0 | 0.0609 | 4.538 |
| graphsage | 30 | snip_pt_local_v2 | 29.0149 | 72.8281 | 67.2335 | static_plan_complete | — | 0 | 0.0619 | 4.596 |
| graphsage | 30 | grasp_pt_local_v2 | 29.1443 | 53.7893 | 42.1594 | static_plan_complete | — | 0 | 0.0633 | 4.700 |
| graphsage | 30 | adaptive_greedy_local_v2 | 31.0886 | 86.6913 | 84.5848 | target_reached | 2 | 17 | 0.6732 | 5.058 |
| graphsage | 30 | q_learning_local_v2 | 18.6531 | 88.3549 | 86.6126 | max_steps | 3 | 14 | 0.9133 | 5.432 |
| graphsage | 50 | random_local_v2 | 50.0907 | 81.5157 | 77.3272 | static_plan_complete | — | 0 | 0.0367 | 4.405 |
| graphsage | 50 | magnitude_local_v2 | 49.7417 | 84.8429 | 82.4898 | static_plan_complete | — | 0 | 0.1236 | 4.804 |
| graphsage | 50 | snip_pt_local_v2 | 49.8414 | 84.2884 | 82.0645 | static_plan_complete | — | 0 | 0.0866 | 4.650 |
| graphsage | 50 | grasp_pt_local_v2 | 50.3399 | 48.0591 | 30.6294 | static_plan_complete | — | 0 | 0.0944 | 4.684 |
| graphsage | 50 | adaptive_greedy_local_v2 | 49.7417 | 84.8429 | 82.4898 | target_reached | 2 | 17 | 0.7143 | 5.174 |
| graphsage | 50 | q_learning_local_v2 | 18.6531 | 88.3549 | 86.6126 | max_steps | 3 | 14 | 0.7799 | 5.315 |

Q stops at its diagnostic3-step horizon, not upon crossing target or accuracy loss;
its missed nominal targets are expected procedure outcomes and correctly recorded
target violations. Greedy GraphSAGE30 also exceeds the ±1pp band despite its valid
`target_reached` rule; the resulting violation flag/gap is preserved. These outcomes
are not implementation failures or matched-compression comparisons. No accuracy
or compression difference is interpreted as algorithm superiority.

## Independent verification

The new [verification_tools/verify_matrix.py](verification_tools/verify_matrix.py)
uses no V2 surgery or search code for physical reconstruction or reward/Q checks.
[verification.json](verification.json) records all18 PASS results, precise model and
output paths, hidden widths, counts, metrics, statistics and saved file hashes.

- Reloaded each actual `pruned.pt` strictly; real full-Cora logits are finite
  with shape[2708,7]. Every tensor equals an independent direct projection of
  the original dense checkpoint by saved original channel IDs. GraphSAGE checks
  both lin_l/lin_r weight branches and the lin_l bias; GCN checks weights/biases.
- Independently counted all parameters with `P_GCN(a,b)=1433a+ab+7b+a+b+7` and
  `P_SAGE(a,b)=2(1433a+ab+7b)+a+b+7`. Actual counts, byte sizes, achieved
  compression, absolute target gaps and violation flags match saved reports.
- Recomputed dense/final validation and final test accuracy/macro-F1 from actual
  predictions using NumPy/sklearn. Test metrics are post-termination reporting;
  they were not used for tuning, selection or exclusion.
- Checked frozen checkpoint/split/graph/tensor/config hashes, model identity,
  seed, architecture, exact split indices and raw/resolved config snapshots.
- Every selected Greedy/Q transition and Q learning transition was independently
  projected from dense original IDs and run on real Cora validation nodes.
  Verified surviving IDs, widths, current accuracy, dense-relative drop, incremental
  reward, once-only terminal penalty, STOP eligibility/counter and terminal priority.
- Replayed sparse Q Bellman updates from training traces and compared every saved
  Q value; checked three dense resets, fresh dense deployment, legal-action greedy
  deployment, actual final-model reporting, entries/visited states/slots/sparsity
  and exact validation evaluation counts. Greedy candidate/evaluation counts match.
- Verified the unmodified search code masks test labels to−1 before score/plan;
  gradient scoring uses train indices, reward uses only the validation callback,
  full-label test metrics follow termination. The passed suite includes both-model
  integration, train-only scoring and the GCN test-label permutation regression.
  No instrumentation altered the normal real-data CLI.
- SHA256 comparison found **28,699 previous files unchanged**, including V1 results, both original datasets/checkpoints/splits, the initial six-run smoke and all its raw logs/configs/verified cache.

## Existing pytest suite

Exit0: **160 passed,2 skipped,39 upstream Torch AST warnings in23.51s**.
Both skips are CUDA preservation checks at `tests/test_local_surgery_v2.py:257`;
CUDA is unavailable. No xfail/collection error. Exact invocation:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /private/tmp/gnn-v2-validation-py312-20261027/bin/python -m pytest tests/test_local_surgery_v2.py tests/test_static_local_v2.py tests/test_adaptive_local_v2.py tests/test_local_v2_integration.py -p no:cacheprovider --basetemp=/private/tmp/local-v2-cora-matrix-pytest-20261027 --tb=short -r fEs
```

Structured command/result: [pytest_result.json](pytest_result.json); raw stdout stays
external at the indexed path. No new failing regression test or V1 rerun was needed
because no implementation defect or code change was found.

## Reproduction commands and artifact index

[cli_runs.json](cli_runs.json) records every exact argv, working directory,
environment, exit code, wall cost, config path, raw log path and output directory.
All18 real runs invoked `python -m gnn_pruning prune`; no internal replacement
entry point or dense training was used. The wrapper refused duplicate run IDs
and each CLI wrote a fresh unique result directory.

External group:
`/Users/perfect/Desktop/code/MSU/MSU_SQI_25_DIPLOMA/test_codex/prunable_gnn_artifacts/local-structured-v2-cora-matrix-20261027`

Each indexed run directory contains `pruned.pt`, `metrics.json`,
`config_snapshot.json`, `trace.json`, `q_table.json`.
[artifact_manifest.csv](artifact_manifest.csv) gives exact absolute locations,
byte sizes and SHA256 for raw checkpoints/logs/configs/manifests/verifier scripts
and all30 prior-run output files referenced by the completed matrix. The large
prior-file baseline itself is external and indexed by its own digest.
Raw artifacts are kept outside Git; compact summaries, exact configs, command
metadata, provenance and independent verification are committed.

The included verification tools are evidence helpers, not new experiment/CI
infrastructure. `preflight.py` records the original clean tested SHA and snapshot;
`run_matrix.py` is an archival record of the authorized18 invocations. Do not rerun
those scripts merely to recreate completed evidence. The read-only model verifier
can be used to inspect existing artifacts; it writes a new verification report
only within this diagnostic group's external directory.

## Failures, conclusions and next boundary

There were **no input/provenance blockers, runtime failures or verified technical
defects** in the18 new runs. Missed targets/terminal reasons follow the specified
algorithm and diagnostic horizon. There was no minimal code fix to make.

The functional matrix now supports moving to a **validation-only Q-learning
pilot** with these pinned graph/checkpoint/split inputs. That pilot still needs
its own scope and scientific choices; this task does not approve lambda2,
3 episodes or max3 as experimental settings. Do not turn these diagnostic quality
or timing rows into paper claims, hardware speedups, convergence evidence or a
large campaign. Keep test labels out of all forthcoming pilot decisions.

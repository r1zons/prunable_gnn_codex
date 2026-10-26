# LOCAL-STRUCTURED-V2 real Cora smoke — 2026-10-27

**Verdict: GO for a bounded broader Cora CPU pilot using the pinned input below.**
All four required methods and optional Random/GraSP passed end-to-end. This is a
functional diagnostic, not a scientific campaign, hyperparameter approval, paper
result or evidence of method superiority. It does not authorize a larger campaign.

## Code and environment

- Canonical checkout: `/Users/perfect/Desktop/code/MSU/MSU_SQI_25_DIPLOMA/test_codex/prunable_gnn_codex`.
- Branch: `implementation/local-structured-v2-20261027`; tested code commit:
  `e2389ca3442fd72d747438d39e15a77a62ad0f36` (the requested baseline).
- Starting worktree clean; all six successful CLI reports record this SHA and
  `tracked_worktree_dirty=false`. The subsequent evidence commit changes only
  research documentation/configs; implementation and tests remain unchanged.
- Existing interpreter `/private/tmp/gnn-v2-validation-py312-20261027/bin/python`:
  Python **3.12.9**, PyTorch **2.2.2**, PyG **2.6.1**, NumPy1.26.4, pytest8.3.5.
  CPU; OMP/MKL/OpenBLAS thread limits2. No environment created or changed.
- Existing [draft PR48](https://github.com/r1zons/prunable_gnn_codex/pull/48)
  remains draft/open; no push or merge performed.

## Frozen checkpoint and graph provenance

Local Cora GCN L3/H32 checkpoints for seeds101/202/303 were located in
`${GNN_ARTIFACT_ROOT}/local-cpu-20261025/raw/results/seed_<seed>/cora_gcn_l3_h32/`,
with `dense_checkpoint.pt` and `splits.yaml`; all six file hashes match the frozen
local artifact manifest. Details: [checkpoint_inventory.json](checkpoint_inventory.json).
The local101 checkpoint SHA256 is
`a41a765bc68611e4896398cc365d6775c3c0a75e363451a8c397cc93171510d6`.
Its recorded training root is the repository's dataset cache, but its historical
graph tensor identity is not independently hash-attested. It was not silently
approved from that current cache and was not selected.

Instead, selected the existing **POLUS-CPU-101-C4-V2 seed101** Cora GCN checkpoint,
executed here on the **local Mac CPU**. No POLUS access or submission occurred.
This is a separate smoke provenance group, not pooled with V1 or local training.
The checkpoint contains `gcn`, Cora, seed101, F1433→32→32→C7,
three layers, dropout0.5; strict state loading succeeds and dense P=47,175.

- Frozen checkpoint: `/Users/perfect/Desktop/code/MSU/MSU_SQI_25_DIPLOMA/test_codex/prunable_gnn_artifacts/polus-pilot-20261024/raw/results/pilot-cpu-101-c4-v2/jobs/cora_gcn_l3_h32/seed_101/cora_gcn_l3_h32/dense_checkpoint.pt`
- Frozen split: `/Users/perfect/Desktop/code/MSU/MSU_SQI_25_DIPLOMA/test_codex/prunable_gnn_artifacts/polus-pilot-20261024/raw/results/pilot-cpu-101-c4-v2/jobs/cora_gcn_l3_h32/seed_101/cora_gcn_l3_h32/splits.yaml`
- Checkpoint SHA256: `276b2ff0f7790a84afc2755e89d88f1a170865a4397e4355daf7552c7075c129`.
- Split file SHA256: `59a0209a8e2a73a5e70a720d327f8c024d36d184a6d00be820ced0e1a3eb08bc`.
- Ordered split-content hash: `942a10b128f7138e118b3f2e2f96eb87a62f856c10d82010ac995e40e2b4c675`; matches checkpoint signature.
- Graph tensor SHA256: `2b2bc996be89812f80ce7d09599dab983f804f87ddd37a09b3ce0b5689ec9a0d`.

The checkpoint manifest was compared byte-for-byte with the frozen results
archive (registry SHA256 `2b97d6923efa6cd8836830fdb7e13872c336da1700b514ab538118d47763fa62`).
Graph recovery uses all eight Cora raw hashes from the recorded POLUS input manifest,
verified against the archived source bundle (registry SHA256
`b5d59fb06f5124f8f74123fd687c40206eb560ef22c29df91d19f50b951439c8`).
The hash-matched `prepare-condition-data.py` and `submit-pilot.sh`, successful job1598695
log, prepared config and pinned historical dataset factory link these inputs to training.
An independent Cora NumPy reconstruction follows the original
[PyG2.4 reader](https://raw.githubusercontent.com/pyg-team/pytorch_geometric/2.4.0/torch_geometric/io/planetoid.py)
and [dataset processor](https://raw.githubusercontent.com/pyg-team/pytorch_geometric/2.4.0/torch_geometric/datasets/planetoid.py).

**Ordering caveat:** PyG2.4 row-sorted the edge tensor; installed PyG2.6.1
column-sorts it. The reconstructed graph has exactly matching x/y and edge set,
but preserves the original row ordering in a dedicated offline processed cache.
It has 2,708 nodes, 10,556 directed edges, 1,433 features and7 classes.
The expected graph hash comes from that manifest-backed reconstruction, before
checking the CLI input. It is not approved from an arbitrary current cache.
No dataset was downloaded, original cache changed, or frozen split regenerated.

**Remaining limit:** no contemporaneous processed-tensor digest was recorded during
dense training. The graph identity is reconstructed from trustworthy archived raw
provenance and deterministic processing, not a claim that such an old tensor digest
exists. Dense validation accuracy0.8983364140480592 and macro-F1≈0.8852755859531409
reproduce the archived metrics (F1 rounding difference1e-16), as corroboration only.
See [graph_provenance.json](graph_provenance.json) for hashes and evidence paths.

## Existing V2 pytest suite

From the canonical checkout, exact command:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /private/tmp/gnn-v2-validation-py312-20261027/bin/python -m pytest tests/test_local_surgery_v2.py tests/test_static_local_v2.py tests/test_adaptive_local_v2.py tests/test_local_v2_integration.py -p no:cacheprovider --basetemp=/private/tmp/local-v2-smoke-pytest-20261027 --tb=short -r fEs
```

Exit0: **160 passed, 2 skipped, 39 upstream Torch AST deprecation warnings in26.30s**.
Both skips are CUDA device-preservation cases at `tests/test_local_surgery_v2.py:257`;
CUDA is unavailable. No xfail/collection errors. Log:
`${GNN_ARTIFACT_ROOT}/local-structured-v2-smoke-20261027/pytest-v2.txt`.
No V1 algorithm changes required an additional V1 test invocation; previous V1
artifacts were independently byte-verified below.

## Diagnostic runs

Each config derives from `configs/experiments/local_structured_v2/example.yaml`.
All use GCN hidden32/32, seed101, target0.30, CPU, min4, tolerance0.01.
Adaptive diagnostic values: **lambda2.0, max_steps3; Q episodes3, gamma0.9**
(alpha0.1, epsilon0.2 from the template). These values are **not approved research
hyperparameters or paper results**. STOP/reward/terminal rules are unchanged.
The CLI wall time below includes interpreter imports and final reporting; it is
not inference latency or a fair algorithm speed comparison.

| Method | Model | Seed | Target | Achieved | Val accuracy | Val macro-F1 | Terminal reason | CLI seconds | Status |
|---|---|---:|---:|---:|---:|---:|---|---:|---|
| magnitude_local_v2 | GCN32/32 | 101 | 0.30 | 0.29019608 | 0.87430684 | 0.84174949 | static_plan_complete | 4.518 | PASS / exit0 |
| snip_pt_local_v2 | GCN32/32 | 101 | 0.30 | 0.29019608 | 0.79667283 | 0.68553126 | static_plan_complete | 4.309 | PASS / exit0 |
| adaptive_greedy_local_v2 | GCN32/32 | 101 | 0.30 | 0.31075782 | 0.88909427 | 0.87668263 | target_reached | 4.406 | PASS / exit0 |
| q_learning_local_v2 | GCN32/32 | 101 | 0.30 | 0.18645469 | 0.90018484 | 0.87965619 | max_steps | 4.522 | PASS / exit0 |
| random_local_v2 | GCN32/32 | 101 | 0.30 | 0.29216746 | 0.87245841 | 0.85218737 | static_plan_complete | 4.601 | PASS / exit0 |
| grasp_pt_local_v2 | GCN32/32 | 101 | 0.30 | 0.29151033 | 0.45471349 | 0.29841150 | static_plan_complete | 4.617 | PASS / exit0 |

Greedy achieved0.31075782, gap0.01075782: target violation=true, despite its correct
`target_reached` stopping rule (s≥target−tolerance). Q achieved0.18645469,
gap0.11354531: target violation=true, stopping at its diagnostic horizon.
Those are correctly reported finished procedures, not matched-compression results.
The lower GraSP/SNIP validation metrics do not establish an implementation defect;
no tuning, method exclusion, algorithm-ranking or superiority claim follows.

Every invocation used the normal CLI, with the actual interpreter and absolute
checkpoint/config paths. Exact argv, environment, exit codes and elapsed times are
saved in the external `cli_runs.json`. For example:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /private/tmp/gnn-v2-validation-py312-20261027/bin/python -m gnn_pruning prune \
  --checkpoint '/Users/perfect/Desktop/code/MSU/MSU_SQI_25_DIPLOMA/test_codex/prunable_gnn_artifacts/polus-pilot-20261024/raw/results/pilot-cpu-101-c4-v2/jobs/cora_gcn_l3_h32/seed_101/cora_gcn_l3_h32/dense_checkpoint.pt' \
  --config '/Users/perfect/Desktop/code/MSU/MSU_SQI_25_DIPLOMA/test_codex/prunable_gnn_artifacts/local-structured-v2-smoke-20261027/configs/magnitude_local_v2.yaml'
```

## Independent artifact verification

All six `pruned.pt` files reload strictly and run a real full-Cora forward pass
with finite [2708,7] logits. The verifier independently slices the frozen dense
weights/biases by original kept IDs and compares every resulting tensor exactly.
Parameter counts match both the actual models and the independent GCN formula;
parameter bytes/checkpoint bytes, achieved compression, absolute gap and target
violation flags match the reports. Dense/final validation and post-termination
test accuracy/macro-F1 were recomputed with NumPy/sklearn and match saved values.
All checkpoint/split/graph/tensor/config hashes, split indices and architecture
provenance match; config snapshots match actual CLI inputs.

Greedy:16 candidate evaluations,17 validation evaluations,2 selected deletions,
`target_reached`. Q:3 episode resets plus fresh dense deployment,3 deployed
successful deletions,14 validation evaluations,7 unique states,5 Q entries,
26 legal state-action slots, sparsity0.8076923076923077, `max_steps`.
All terminal rewards/penalties were recomputed; the Q-table was independently
replayed from serialized Bellman updates (legal unseen entries0, terminal bootstrap0),
and deployment actions match the saved table's greedy legal policy.
The saved model matches the actual last deployment transition; no substitution
of an earlier best model. Trace paths and hashes are in [verification.json](verification.json).

Search isolation was checked against `workflow_v2`: test labels are replaced by−1
before score/plan/apply; scoring gets train indices and adaptive reward gets only a
validation callback. Final full-label metrics run after search terminates.
The existing passed synthetic test-label permutation regression verifies unchanged
Q-policy, retained IDs and final tensors. The real CLI was not monkeypatched.

All **28,628** protected pre-existing files under V1 local/POLUS artifacts, environment
bundle, old V2-fix evidence, repository `runs/` and original `data/` retain identical
SHA256 values. No earlier result or frozen checkpoint/split was overwritten.

## Exact artifact locations

External root:
`/Users/perfect/Desktop/code/MSU/MSU_SQI_25_DIPLOMA/test_codex/prunable_gnn_artifacts/local-structured-v2-smoke-20261027`

- `magnitude_local_v2`: `/Users/perfect/Desktop/code/MSU/MSU_SQI_25_DIPLOMA/test_codex/prunable_gnn_artifacts/local-structured-v2-smoke-20261027/outputs/local_structured_v2/magnitude_local_v2/target_0.3/aeda1fa0d41d`
- `snip_pt_local_v2`: `/Users/perfect/Desktop/code/MSU/MSU_SQI_25_DIPLOMA/test_codex/prunable_gnn_artifacts/local-structured-v2-smoke-20261027/outputs/local_structured_v2/snip_pt_local_v2/target_0.3/1933cda61d37`
- `adaptive_greedy_local_v2`: `/Users/perfect/Desktop/code/MSU/MSU_SQI_25_DIPLOMA/test_codex/prunable_gnn_artifacts/local-structured-v2-smoke-20261027/outputs/local_structured_v2/adaptive_greedy_local_v2/target_0.3/b6ce93b48435`
- `q_learning_local_v2`: `/Users/perfect/Desktop/code/MSU/MSU_SQI_25_DIPLOMA/test_codex/prunable_gnn_artifacts/local-structured-v2-smoke-20261027/outputs/local_structured_v2/q_learning_local_v2/target_0.3/e63322e857a8`
- `random_local_v2`: `/Users/perfect/Desktop/code/MSU/MSU_SQI_25_DIPLOMA/test_codex/prunable_gnn_artifacts/local-structured-v2-smoke-20261027/outputs/local_structured_v2/random_local_v2/target_0.3/19bd1f141a66`
- `grasp_pt_local_v2`: `/Users/perfect/Desktop/code/MSU/MSU_SQI_25_DIPLOMA/test_codex/prunable_gnn_artifacts/local-structured-v2-smoke-20261027/outputs/local_structured_v2/grasp_pt_local_v2/target_0.3/7a050e79fcad`

Each run directory contains `pruned.pt`, `metrics.json`, `config_snapshot.json`,
`trace.json`, `q_table.json`. The external root also contains `runtime.json`,
`checkpoint_inventory.json`, `graph_provenance.json`, `cli_runs.json`, each CLI log,
`pytest-v2.txt`, `preflight.py/.txt`, `run_smoke.py`, `verify_smoke.py`,
`verification.json/.txt`, and the 28,628-file `protected_files_before.json`.
`verified_data/` contains only the isolated manifest-backed Cora input.
Tracked [configs/](configs/) are byte-identical copies of the successful inputs.
[summary.csv](summary.csv) and [verification.json](verification.json) index every
checkpoint/metrics path and output hash.

## Failures, corrections and scope

- Initial diagnostic preflight assertion exposed PyG2.4 versus2.6 edge ordering;
  corrected by reconstructing the original graph order in isolated input data.
  Environment/data compatibility issue; no model, pruning rule or target changed.
- The first Magnitude CLI attempt exited1 during config resolution, before graph
  loading or pruning: the typed schema accepts scalar `hidden_channels: 32`, while
  the operator-supplied list `[32,32]` raised `TypeError` in its positive-value check.
  Used the scalar syntax from the existing example (three layers still mean two
  hidden32 layers); no schema/API extension needed. Failed config/log/argv are
  preserved externally as `magnitude_local_v2.initial-list-widths.yaml`,
  `magnitude_local_v2.initial-config-failure.log` and `initial-config-failure.json`
  (the YAML is within external `configs/`).
- No implementation defect requiring a fix was verified. No source/test changes,
  algorithm changes, regression-test addition, or refactoring occurred.
- No GNN training, fine-tuning, dataset download, new environment, CI change,
  POLUS submission, test-driven tuning or large campaign occurred.
- GO is a CPU functional gate using this frozen provenance. Scientific pilot
  hyperparameters, latency methodology, other data/checkpoints, GraphSAGE, CUDA
  and broader campaign decisions remain outside this smoke. Original local dense
  checkpoints still need their own reliable historical graph link before reuse.

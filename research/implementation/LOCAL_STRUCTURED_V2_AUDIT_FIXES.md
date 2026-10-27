# LOCAL-STRUCTURED-V2 — audit fixes

2026-10-27. Audit corrections based on PR48 head
`cf61c80687b196b894fe6d93248e06b06006e500`, published in the existing PR48 branch
at the researcher's explicit request.
This is implementation verification, not an experimental campaign.

## Corrections

- **M1:** reject dense architecture/config mismatches before loading the dataset:
  layer count, normalized hidden widths, dropout, and explicitly configured
  input/output dimensions. Present architecture fields in run_signature must agree
  with model_config. Older signatures without those fields remain supported via
  model_config. Save the checked dense model_config in provenance.
- **M2:** require a valid graph_sha256 and compare the loaded x/edge_index/y tensors
  before any scoring or search. The expected hash must be obtained from the original
  dense graph or its verified manifest; hashing an arbitrary current cache does not
  establish its relationship to dense training. See the updated prototype guide.
- **M3:** reject V2 checkpoints in the legacy evaluate-pruned command before its
  split path. Existing V2 metrics.json already reports the frozen-split metrics.
  Separate V2 reevaluation is not added. V1 evaluation retains its original path.

Static/adaptive algorithms, surgery, reward, stopping rules and canonical R2
methodology are unchanged. The integration fixture now declares its actual dense
architecture and graph hash. The test-label isolation test explicitly repins its
intentional synthetic label permutation; policy/final tensors must still agree.

## Measured verification

- Test-first integration Red on the original implementation, Python3.8.20:
  **21 failed / 26 passed**, exit1. Failures reached dataset loading for a different
  architecture/unpinned graph, or the legacy split path for V2 evaluation.
- Initial integration Green: **47 passed**. Two further compatibility checks
  (V1 evaluation and older signatures) brought integration to **49 passed**.
- Final full V2 suite on **Python3.12.9 / Torch2.2.2 / PyG2.6.1 / pytest8.3.5**:
  **160 passed / 2 CUDA skipped in84.51s**, exit0.
- Selected no-fit V1 regressions on the same Python3.12 runtime:
  **31 passed / 20 deselected**, then **8 passed / 9 deselected**, both exit0.
- All39 warnings in each final invocation are upstream Torch AST deprecations.
  There are no xfails or collection errors. CUDA skips do not qualify GPU support.
- Two archived Cora seed202 dense checkpoints (GCN/GraphSAGE) were read-only checked
  on the existing Python3.8 runtime: matching architecture accepted; declaring
  hidden64 instead of32 rejected. No real-dataset forward/search was executed.
- git diff --check passes. Core static_v2.py, adaptive_v2.py and local_v2.py have
  no changes relative to the audited head.

Final V2 command, from the canonical checkout:

```sh
PYTHONDONTWRITEBYTECODE=1 /private/tmp/gnn-v2-validation-py312-20261027/bin/python -m pytest \
  tests/test_local_surgery_v2.py tests/test_static_local_v2.py \
  tests/test_adaptive_local_v2.py tests/test_local_v2_integration.py \
  -p no:cacheprovider --basetemp=/private/tmp/local-v2-fix-all-py312-20261027 --tb=short -r fEs
```

Raw final logs and runtime/package manifests are outside Git under
`${GNN_ARTIFACT_ROOT}/local-structured-v2-fixes-20261027/`:
v2-py312.txt, v1-core-py312.txt, v1-extra-py312.txt, runtime.json,
python312-packages.txt. The Python3.12 environment is temporary and separate;
existing conda environments and package metadata were not changed.

The three audited defects are covered. The next real-data step is preparing a
matching Cora config with verified checkpoint/split/original-graph hashes, then a
separately authorized CPU smoke test. No GNN training, real-dataset download,
campaign, POLUS job, fine-tuning or timing benchmark was run here.

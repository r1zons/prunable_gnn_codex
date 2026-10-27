# Stage II — Static Local V2

2026-10-27, SPEC R2 (520bb4e). Selected interpreter: conda gnn_pruning,
Python3.8.20, Torch2.2.2, PyG2.6.1, pytest8.3.5. No environment upgrades.

## Red

`python -m pytest tests/test_static_local_v2.py --tb=line -r fEs`

**21 failed in 4.89s**, exit1. Importable V2 API stubs raise NotImplementedError;
no collection/import failures. Tests precede production implementation.
Independent oracles cover all841 physical width-pair counts (both models/bias
variants), unique-coordinate union, full-mask ranking/search/ties/fallback,
CiteSeer30, seeded Random without oracle, analytic bias-cross-Hessian and unused/
stationary gradients, train-only/dropout-off SNIP, finite-difference SNIP/Hg on
GCN/SAGE, source/RNG preservation and rejection of nonfinite saliency.

## Green / review

Red commit: `c49832c`. Unchanged tests: **21 passed in 17.69s**, exit0.
The count oracle physically rebuilds all841 width pairs for GCN/SAGE, bias on/off.
GraSP passed the analytic signed/bias-cross-Hessian/unused/stationary and GCN/SAGE
finite-difference Hg checks in this CPU runtime. It remains conditional for any
new platform/runtime; no empirical quality claim follows from mathematical gates.

V1 command: same no-training subset as Stage I (test_models, test_surgery,
test_pruning, test_graphsage_config_application and selected terminal/registry
Q tests): **31 passed, 20 deselected in 4.18s**, exit0. `git diff --check` passed.
Stage II core gate accepted by self-review: fixed dense rankings, original raw
full-mask criterion, unique union/bias, all nearest ties and lexicographic ties,
local seeded Random; no new optimization criterion, accuracy oracle or prefix
optimization. Production integration/registry/CLI are Stage III work.
No dense training, fine-tuning, real dataset download or campaign was run.

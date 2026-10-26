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

Pending. No training, fine-tuning, real dataset download or campaign is run.

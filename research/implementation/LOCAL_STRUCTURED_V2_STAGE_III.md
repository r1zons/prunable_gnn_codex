# Stage III — Adaptive Local V2 and integration

2026-10-27, after Stage II Green `4df84c5`. Runtime: user-selected conda
gnn_pruning, Python3.8.20 / Torch2.2.2 / PyG2.6.1 / pytest8.3.5.

## Red

`python -m pytest tests/test_adaptive_local_v2.py tests/test_local_v2_integration.py --tb=line -r fEs`

Measured Red: **21 failed in 7.06s**, exit1 (commit `08e4fb8`). Adaptive stubs raise NotImplementedError;
integration fails because V2 methods are not registered/routed. No import errors.
Tests precede adaptive implementation and production registration/routing.
Coverage: integer actions/min4/repeat layers; STOP0/crossing/horizon/once-only P;
dense-relative drop; Greedy candidate rewards; legal-only/unvisited/terminal
Bellman targets; exact widths/horizon state; episode/deployment resets and sparse
Q statistics; six V2 identifiers, strict checkpoint reload, full config/provenance
and mismatch rejection without split regeneration. Only synthetic graphs and
short mocked-validation policy episodes, no neural-network fit.

## Green / review

Initial unchanged Red set: **21 passed in 8.12s**, exit0. Additional boundary
oracles first exposed **3 failures** (external float/bool action handling and
initial no-action terminal penalty), then passed after correction. Tests also
cover exactly20 successful removals, all8 initial actions at width32, test-label
independence of Q policy/final weights and preservation of prior V2 invocations.
Full integration is parameterized over all6 methods × GCN/GraphSAGE.

Self-review confirmed:

- R=0.6Δs−0.4D with fixed dense validation accuracy; STOP-only P, once-only P
  for STOP/max/no-actions/Greedy crossing. Invalid actions leave model/counter unchanged.
- STOP does not consume horizon; Q does not target/accuracy-stop. Terminal state
  flag distinguishes STOP from its preceding aggregate state and keeps Q sparsity
  denominator honest. No retained-channel signatures were introduced.
- Legal-only Bellman max, unseen legal0, terminal bootstrap0, gamma0.9;
  each episode/deployment starts dense, actual final model reported.
- Unique states/Q entries/legal slots/sparsity and evaluation/scoring/surgery costs
  are recorded. Original channel IDs survive iterative surgery and disk reload.
- Six separately registered V2 classes expose score/plan/apply. The legacy
  workflow only adds a3-line V2 dispatch, and __init__ adds registration; V1
  implementations remain untouched. Existing CLI list-pruners lists both versions.
- Required checkpoint/split file hashes, dense signature dataset/model/seed and
  split hash are checked. Nonempty/disjoint/full split coverage is checked;
  no split generation path exists. Optional graph hash can additionally pin tensors.
- Full raw+resolved config hash includes lambda/actions/episodes; canonical dense/
  graph tensor hashes, original IDs, Git SHA/dirty flag and runtime are saved.
  Search gets erased test labels; val/test accuracy and F1 are computed after search.
- Each run gets a new output directory; frozen V1 evidence directories match
  d422263 exactly. No real datasets, GNN fit, fine-tuning, LSF or campaign was run.

V1 regressions: Stage I-style no-fit subset **31 passed, 20 deselected in4.02s**;
additional V1 static/gradient-pruner and mocked checkpoint workflow subset
**8 passed, 9 deselected in3.94s**. These39 passes are not a full-suite claim.
The full suite contains training tests and is not run under this request.

Remaining pre-experiment work: validation-only lambda pilot/budget, policy episode
budget qualification, CPU thread/full environment manifests and local/POLUS
placement. CUDA/POLUS remain unverified. Inference latency is explicitly unmeasured
in prototype checks, not inferred from compression. Current selected conda is
Python3.8.20 (package minimum>=3.10 unchanged); only Stage I was additionally
checked on transient Python3.11.13. The unwanted temporary environment was removed.
No mathematical contradiction requiring a change to R2 was found.

## Final combined verification

```sh
/usr/local/Caskroom/miniconda/base/envs/gnn_pruning/bin/python -m pytest tests/test_local_surgery_v2.py tests/test_static_local_v2.py tests/test_adaptive_local_v2.py tests/test_local_v2_integration.py --tb=short -r fEs
```

**127 passed, 2 CUDA skipped in21.08s**, exit0. No xfails or collection errors.
Both old V1 regression invocations above exited0. Additional command:

```sh
/usr/local/Caskroom/miniconda/base/envs/gnn_pruning/bin/python -m pytest tests/test_pruning_methods.py -k 'pruner_outputs_valid_plan or pruning_runs_end_to_end_on_small_model or structured_mode_reduces_parameter_count or random_structured_pruning_applies_surgery or global_magnitude_structured_pruning_applies_surgery or unstructured_mode_does_not_report_structural_compression or snip_unstructured_pruning_runs_with_gradient_saliency or grasp_structured_pruning_reuses_surgery' --tb=short -r fEs
```

`PYTHONPATH=src python -m gnn_pruning list-pruners` exited0 and lists all6 V2
names alongside unchanged V1 names. `git diff --check` passed; frozen local/POLUS
directories have no diff against d422263. Added static candidate-count fields
are diagnostics only; Stage II ranking/search criterion did not change.

Prototype stages I→II→III accepted by these bounded self-review gates and ready
for a separate independent audit. No campaign/empirical superiority is claimed.

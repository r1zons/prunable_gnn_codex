# LOCAL-STRUCTURED-V2 — Stage I evidence

Date: 2026-10-27. Implementation branch: `implementation/local-structured-v2-20261027`.
Sources verified with git fetch: spec `9076b0630a95fbbfab64422ec798189f29e589df`,
audit `5c4259664dfb5ae05e84deadacfcdc9486abb7ca`.
Audit blob matches pre-existing local file: `309cfcc5b56f6d141aa343cf5dd5a430e2a89cf2`.
V1 consolidation `d422263` preserved in merge `33984b0`; no historical evidence rewritten.
Post-audit revision / plan commit: `9d42cb6`.

## Red — measured before implementation

Invocation (canonical repository cwd):

```sh
/usr/local/Caskroom/miniconda/base/envs/gnn_pruning/bin/python -m pytest tests/test_local_surgery_v2.py --tb=line -r fEs
```

Baseline: `surgery/local_v2.py` only aliases the unchanged legacy Local Surgery.
Thus collection/import succeeds and failures test real behavior, not a missing module.

Result: **40 failed, 31 passed, 2 skipped in 5.49s**, process exit **1**.

- 16 float64 cases fail dtype preservation: legacy rebuild creates float32 parameters.
- 2 preservation tests fail (module modes; dtype/flags/PyG metadata are additional assertions).
- 10 validation cases fail: min4, duplicate/fractional/bool indices and no-op are accepted.
- 12 out-of-scope SAGE variant cases fail to reject project/normalize/max/no-root options.
- 31 baseline passes already cover CPU float32 tensors/count/logits, valid removal
  order, disk checkpoint roundtrip, some invalid inputs and warm GCN cache handling.
- 2 CUDA checks skipped: no CUDA device available. A skip is not a CUDA pass.

Environment actually used: CPython **3.8.20**, PyTorch **2.2.2**, PyG **2.6.1**,
pytest **8.3.5**, local CPU. No dependency changes. Python is below the declared
package minimum >=3.10; this is a runtime qualification blocker for a V2 campaign,
not evidence of a supported-package installation. POLUS not contacted.

## Green / review gate

Red tests and baseline adapter committed at **5a28be3** before implementation.
The tests remained unchanged for the Green run. Re-running the same invocation
against the new V2 implementation: **71 passed, 2 skipped in 5.61s**, exit **0**.
No collection errors, xfails or relaxed tolerances. The two skips are CUDA only.

The implementation deep-copies the source, retains PyG Linear types, slices only
selected rows/bias and immediate downstream columns, clears affected GCN caches,
and preserves dtype/device/requires_grad/mixed module modes. It rejects requests
outside the declared contract. The CPU RNG used by PyG deepcopy is isolated.
Manual checks also confirmed CPU RNG state unchanged and explicit minimum override
(min_channels_per_layer=2) on both architectures.

Legacy regression invocation (no training tests selected):

```sh
/usr/local/Caskroom/miniconda/base/envs/gnn_pruning/bin/python -m pytest tests/test_models.py tests/test_surgery.py tests/test_pruning.py tests/test_graphsage_config_application.py tests/test_q_learning_tabular.py -k 'not q_learning or registered or stop_action_has_stable_key or update_q_supports_terminal_stop or terminal_reward' --tb=short -r fEs
```

Result: **31 passed, 20 deselected in 7.11s**, exit **0**. Deselected tests include
policy-search/workflow paths; this is not a full-suite claim. Dense/fine-tuning
smoke tests are deliberately not run under the no-training instruction.

Self-review against the Stage I acceptance criteria:

- Independent references use dense tensor masks and activation gates; no production
  helper computes expected values. Nonzero biases and both SAGE branches are checked.
- Exact surviving tensors and analytic parameter count include bias, with/without
  bias, float32/float64, boundary4/nonuniform widths and both removal orders.
- Repeated 1→0→1 removal explicitly maps gapped original IDs to current positions.
- Strict serialized checkpoint reload preserves nonuniform architecture and logits.
  Original IDs are saved by the caller in the checkpoint; surgery itself accepts
  current positions and does not infer original identities.
- Input/source tensors and graph remain unchanged; output parameters own storage.
- Full V1 source, old tests/configs/scripts and pyproject have no edits versus
  `9076b06`. The only source/test additions are local_v2.py/test_local_surgery_v2.py.
- Frozen local/POLUS evidence directories have no diff versus `d422263`.
- `git diff --check` passes. No dependency or CI changes.

**Gate outcome:** CPU Stage I functional checks passed on the observed environment.
This is a self-review, not a new independent scientific audit. Supported Python
>=3.10 verification, CUDA and POLUS remain unverified; the current Python3.8 result
does not certify campaign readiness. No training, fine-tuning, large experiment,
real dataset download, latency benchmark or GitHub Actions run was performed.

## Subsequent stages

Stage II: blocked on SPEC §15 C01/M01/M04/m01, including comparison domain,
scoring/bias and Random rules. Stage III: blocked on Stage II and C02 return/STOP.
The referenced chat returns assistant response placeholders, so missing approved
scientific decisions have not been invented. Request for those decisions is pending.

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

Pending implementation and execution. No training, fine-tuning, large experiment,
real dataset download, latency benchmark or GitHub Actions run was performed.

## Subsequent stages

Stage II: blocked on SPEC §15 C01/M01/M04/m01, including comparison domain,
scoring/bias and Random rules. Stage III: blocked on Stage II and C02 return/STOP.
The referenced chat returns assistant response placeholders, so missing approved
scientific decisions have not been invented. Request for those decisions is pending.

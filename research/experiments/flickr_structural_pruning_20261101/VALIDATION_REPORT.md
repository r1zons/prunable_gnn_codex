# Validation gates — preparation stage

Local Python3.12.9/Torch2.2.2/PyG2.6.1, isolated verification environment outside
Git.121 focused tests PASS /0 FAIL /0 SKIPPED; full751 PASS /0 FAIL /6 SKIPPED
(131.88s). Explicit skips:2 CUDA device tests without CUDA,4 opt-in archived-Cora
Lookahead gates. Skips do not count as passes. Raw logs remain external at
`${GNN_ARTIFACT_ROOT}/flickr-structural-pruning-20261101/`.

The focused suite covers independent geometry/continuations, depths2/3/4 and both
models, all seven methods with real synthetic forwards, physical masked-reference
equivalence and reload, deterministic repeated results, independent depth2 paths
and replanning, new depth1/Greedy equality, exact ties, terminal locking/single
penalty, Q target conditioning and retained-state identity, honest callback
accounting, output uniqueness,56 independent jobs and ambiguous acceptance safety.

Native checkpoints/real Flickr technical outputs, source/input hashes, feasibility
and capacity results must be added after execution. This file is not a native
PASS certificate. No research condition may be submitted before that gate.

## Exact-inference reuse revision2

146 focused PASS/0 FAIL/0 SKIPPED; full776 PASS/0 FAIL/6 explicit skips,
119.28seconds. Twenty-four cached/uncached controller cases at all depths and
both models match decisions, rewards, all Q updates/tables, masks and tensors
exactly. Lookup hits and physical forwards are accounted separately; different
tensors at the same width miss. Original code and budget/reward policy unchanged.

First native stage1611889–1611892:4 jobs DONE,28 technical output inventories
passed; no scientific jobs submitted. Its resource gate failed Q4/256 at212.62min
against the unchanged170min forecast threshold/180min queue cap. Raw archive
`preflight-uncached-evidence.tar.gz` SHA256:
`8ea910d5c6114779f67e6e373bbbb1c1d165eb4a4934931e39f689911b831cec`.
[NATIVE_CACHE_CONTRACT.json](NATIVE_CACHE_CONTRACT.json) independently reads736
fresh second-step native forward records:381 distinct masks,355 repeats, zero
accuracy discrepancies. It performs no new forward or search.

Revised native checks remain a mandatory gate. They use three technical Q
episodes to measure repeated-state reuse; the research budget stays30 everywhere.

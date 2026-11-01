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

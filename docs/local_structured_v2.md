# LOCAL-STRUCTURED-V2 prototype (R2)

Source of scientific decisions: [canonical SPEC](../research/methodology/LOCAL_STRUCTURED_V2_SPEC.md),
§15 R2. V1 methods and historical campaigns remain separate. The implementation
uses frozen dense checkpoints; it never trains/fine-tunes a neural network.

## Entry point and configuration

The existing `prune` command recognizes six distinct identifiers:

- `random_local_v2`, `magnitude_local_v2`, `snip_pt_local_v2`;
- conditional `grasp_pt_local_v2` (mathematical CPU gates passed; recheck other platforms);
- `adaptive_greedy_local_v2`, `q_learning_local_v2`.

Start from [example.yaml](../configs/experiments/local_structured_v2/example.yaml).
It deliberately contains required provenance placeholders and cannot launch
against arbitrary/unpinned files. Freeze a dense checkpoint and its original
`splits.yaml`, record SHA256 of both files, dataset/model/seed and the dense
checkpoint's `run_signature.split_hash`. Relative split paths are resolved against
the configuration file. Missing/mismatching hashes/identity/indices fail; a new
split is never created. Optional `v2.graph_sha256` pins canonical graph tensor identity.

After a separately authorized experimental launch, the command is:

```sh
PYTHONPATH=src conda run -n gnn_pruning python -m gnn_pruning prune \
  --checkpoint /absolute/path/to/frozen_dense.pt \
  --config /absolute/path/to/v2_config.yaml
```

This is documentation, not a record of a campaign launch. Current user-selected
conda `gnn_pruning` is Python3.8.20; package metadata still requires>=3.10. The
unchanged Stage I suite also passed on temporary Python3.11.13 before that unwanted
environment was removed. No conda environment or package metadata was upgraded.

For adaptive methods, `v2.penalty_lambda` is mandatory, finite and nonnegative.
Its scientific value is selected later by validation-only pilot; unit tests use
explicit toy values. No default lambda is supplied. Defaults: max20 successful
removals, min4, tolerance0.01, gamma0.9, 30 episodes, alpha0.1, epsilon0.2.
STOP becomes legal after one successful removal, does not consume the removal
horizon and pays only the once-only terminal penalty. Gamma is fixed at0.9.
Q target crossing and accuracy drop do not terminate search. Greedy retains
its own target/max/no-actions stopping. The action set is exactly integer
(layer,k) for k=1,2,4,8; no ratio actions/clipping/legacy bonuses.

## Outputs and limits

Each invocation creates a new directory below
`run.output_dir/local_structured_v2/<method>/target_<fraction>/<unique-id>`;
it preserves prior invocations and V1 artifacts. Outputs:

- `pruned.pt`: actual final physical model/config, original kept channel IDs and provenance;
- `metrics.json`: achieved compression/gap/violation, parameter/byte counts,
  static candidate counts, search/scoring/surgery costs, validation-call counts,
  dense/final validation/test accuracy and macro-F1;
- `config_snapshot.json`: full raw and resolved settings (pruning/RL settings are hashed);
- `trace.json`: actual selected deployment transitions and Q episode resets/transitions;
- `q_table.json`: sparse state/action rows (empty for static/Greedy).

Test labels are erased from the graph given to search, and used for metrics only
after termination. Nonfinite scores/validation values are errors. Zero saliency
is allowed and reported. Local Surgery supports the established GCN and mean
GraphSAGE variants; other SAGE variants fail explicitly.

Q state contains exact widths, remaining removal horizon, validation-accuracy
bucket and terminal flag. Signed target deviation is recorded from architecture.
Without retained-channel signatures, this remains an aggregated-state heuristic,
not a Markov/convergence guarantee. Report unique states, entries, legal slots and
Q-table sparsity; deployment resets to dense and never substitutes a past best model.

Static band-search and adaptive soft target preferences can yield different
compression. Report their actual outcomes; do not claim matched-compression
superiority or causal RL effect. Inference latency is intentionally **not measured**
in these correctness checks (`inference_benchmark=null`); campaign timing/CPU threads,
full environment manifests and local/POLUS placement remain pre-experiment work.
No scientific quality or speedup claim follows from passing unit tests.

Evidence: [I](../research/implementation/LOCAL_STRUCTURED_V2_STAGE_I.md),
[II](../research/implementation/LOCAL_STRUCTURED_V2_STAGE_II.md),
[III](../research/implementation/LOCAL_STRUCTURED_V2_STAGE_III.md).

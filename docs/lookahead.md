# Lookahead with native Greedy threshold stopping

This audit supersedes the exact-reachability Lookahead at `70fe25d`. The existing
registry name `lookahead_local_v4` is retained for configuration compatibility;
it no longer means that Lookahead uses V4's exact-band controller. Historical
V3/V4 algorithms, archived results and the earlier verification receipt remain
unchanged. See the [independent audit](../research/audits/2026-10-31_lookahead_greedy_threshold.md)
and its [test evidence](../research/audits/lookahead_threshold_20261031/verification.json).

## Policy and termination

`run_lookahead(..., depth=2)` now constructs **native `LocalV2Env(greedy=True)`**,
the same environment used by `adaptive_greedy_local_v2` / `run_greedy`.
It enumerates every legal first action and every legal successor of nonterminal
first actions, scores `r1 + 0.9*r2`, executes only the selected **first** action,
and replans. A first-action terminal path scores only `r1`. Depth 1 is immediate
Greedy, including exact ties in legal enumeration order (layer 0, layer 1;
deletion sizes ascending). There is no beam, learned value or two-action commit.

Magnitude ranking, physical channel removal, sizes `{1,2,4,8}`, minimum width 4,
maximum 20 successful actions, tolerance 0.01 and lambda 2 are preserved. Shared
settings can be explicitly configured, but must match the chosen Greedy control.
Invalid settings still fail; a discretely unattainable target band does **not**.

Stopping is checked **after an action**, in the native priority order:

1. `max_steps` when the successful-action horizon is exhausted;
2. `no_valid_actions` when neither hidden layer can be reduced;
3. `target_reached` when physical parameter compression `s >= target-tolerance`.

There is no exact-reachability or upper-band action filter, and no STOP action.
Overshooting `target+tolerance` is permitted. Horizon/channel exhaustion may end
below the threshold. Target 0 (or a target already within tolerance at dense)
still makes one deletion if any is available: this native Greedy behavior is
preserved for exact depth-1 equivalence. Initially nonprunable models return the
native zero-action terminal trace and single penalty, with one dense evaluation.
A reason of `max_steps` does not imply failure to reach the threshold on that step.

Results expose `achieved_sparsity`, absolute `target_gap`, signed target deviation,
`overshoot=max(0,s-target)`, `overshoot_beyond_tolerance`, `threshold_reached`,
`target_violation` and the actual terminal reason. `target_violation` retains its
meaning of **outside the symmetric band**, even when threshold stopping succeeds;
it is a reported outcome, not an instruction to reject/correct an overshoot.

## Reward and scientific comparability

The unchanged reward is
`0.6*(s_after-s_before) - 0.4*max(0,A_dense-A_after) + terminal_penalty`.
The environment adds `-lambda*max(0,abs(s_after-target)-tolerance)` once at actual
termination. Overshoot can therefore produce a nonzero penalty, unlike feasible
exact-band paths. A second-step terminal reward, including its penalty, is
multiplied by 0.9 once. Reused terminal previews receive no additional penalty.
A nonterminal depth cutoff has zero bootstrap, no penalty, and no invented STOP.
The repeated dense-to-current drop penalty is not an incremental-drop reward;
path length and intermediate losses affect its return. Depth 2 is a truncated
objective, not the full remaining-horizon objective of Q-learning or a global
optimality guarantee.

**Threshold Lookahead and native Greedy have identical stopping/action rules.
Constrained V3/V4 Q-learning does not.** Its exact-band reachability filter can
remove actions allowed here, force other widths/masks and alter where terminal
penalties apply. The corrected Cora depth-1 check reaches about 31.08% compression
at nominal 30%, outside the former 29–31% band. Equal requested compression does
not make these physically matched models or isolate a planning/learning effect.

A full multi-method campaign must predeclare either separate controller groups
with achieved-compression/width/mask comparisons, or separately authorized new
harmonized variants. Do not relabel the old Q runs as threshold controls. No
accuracy/cost improvement is inferred from the implementation checks.

## Reuse, logging and access

First-action prefixes are evaluated once per tree. The selected preview is
adopted without evaluation; its children can be reused on the next plan only for
the identical concrete parent with matching model identity, mask, exact accuracy,
reward inputs and horizon/state. Equal aggregate keys do not enable reuse.
The planner rejects constrained subclasses even if they set `greedy=True`.
Weights, graph inputs and the validation callback must remain frozen and
deterministic; arbitrary in-place weight mutation or stochastic evaluation is
outside this cache contract. There is no global transposition cache.

`candidate_evaluations` counts new preview forwards; `validation_evaluations`
adds the one dense forward. `reused_evaluations` counts cache hits;
`explored_candidate_transitions` includes both. Scored leaf paths share prefixes.
Adopting a preview adds no forward. Scoring/surgery costs include all new branches;
search wall time is not inference latency. The frozen CLI separately reports four
post-search dense/final validation/test reporting forwards. Decisions contain
ordered legal actions, all paths/rewards/returns, proposed and selected actions,
node states, retained IDs, widths, validation accuracy, physical parameter counts,
compression, terminal reasons/penalties and reuse flags. Execution traces are
unchanged native environment records, allowing exact depth-1 comparison.

Planning receives only validation accuracy. The CLI masks test labels during
search; tests verify invariance under repinned nonvalidation-label changes.
The archived Cora gate masks **all** nonvalidation labels and reports validation
accuracy/F1 only. No test metric participates in decisions or audit case selection.

## Interface and output identity

Use the inert [configuration template](../configs/experiments/lookahead/example.yaml)
with verified frozen checkpoint/split/graph pins and a supported three-layer GCN
or mean/root-weight GraphSAGE. The existing `prune` interface supports Cora,
CiteSeer and PubMed; name-routing integration is tested with synthetic graphs.
No dataset download or multi-dataset experiment was performed for this audit.
`lookahead.depth: 1` selects the native Greedy reference; depth 2 is default.
Gamma must be 0.9; unknown Lookahead options are rejected.

The new provenance identity is `LOCAL-STRUCTURED-LOOKAHEAD-GREEDY-THRESHOLD-V1`,
with `termination_policy: greedy_threshold`. Fresh UUID outputs live under
`local_structured_lookahead_threshold`, separating them from the former exact-band
Lookahead. The registry name alone must not be used to merge those protocols.
Artifacts include the physical checkpoint, metrics, deployment trace, candidate
decisions and config snapshot. Test metrics, if reported by the CLI, are computed
only after search. Legacy split-regenerating `evaluate-pruned` remains blocked.

## Reproducible verification

```bash
python -m pytest tests/test_lookahead.py tests/test_lookahead_integration.py -r a
GNN_LOOKAHEAD_CORA_MANIFEST=/absolute/path/to/verified/inputs/manifest.json \
  python -m pytest -r a
```

The archived test reads preexisting tensor/checkpoint files through the unchanged
hash-verifying loader, never `load_dataset`. Without explicit opt-in, its four
cases skip and do not establish a PASS; an invalid explicitly supplied pin fails.
`GNN_LOOKAHEAD_AUDIT_OUTPUT` optionally points to a fresh external directory for
immutable per-case evidence. The gate runs only depth-1/native-Greedy equality at
Cora 30%/50%, both architectures; there is no depth-2 Cora campaign.

Current evidence: **53 focused PASS / 0 FAIL / 0 SKIPPED; full suite 506 PASS /
0 FAIL / 2 SKIPPED**, including all four archived Cora checks. The skips are the
existing CUDA device-preservation tests on a CPU-only host. Real-forward depth-2
oracles independently calculate magnitude ranks, legal paths, rewards, stopping
and returns; only established Local Surgery is shared. The earlier 43/492 counts
in `lookahead_verification_20261031.json` belong solely to the old implementation.

# Deterministic receding-horizon structural Lookahead

Implementation only; no new campaign or benchmark results. The dedicated branch
starts at `f4a615eda552f760558cdd7f0f98b26b710ff44f`. V3/V4 algorithms, results,
archives and the SITITO manuscript remain unchanged.

## Policy and objective

`gnn_pruning.pruning.lookahead.run_lookahead(model, evaluate, target, depth=2)`
uses the existing `experimental_v4.V4Env` with standard settings. It exhaustively
evaluates every legal first action, and every legal successor of a nonterminal
first action. It chooses the path maximizing `r1 + 0.9*r2`, **adopts only its first
successor**, and replans. Depth 1 chooses the largest immediate reward exactly as
the matching constrained Greedy. No beam, learned value, rollout policy, fixed
two-action execution or extra stopping action is used.

Exact ties follow existing enumeration order: layer 0 before layer 1, sizes ascending;
within a first-action branch, the first equal-scoring continuation wins. Equality
uses the same exact float comparison as Greedy, without an invented tie tolerance.

Defaults match V4: sizes {1,2,4,8}, magnitude ranking and Local Surgery, minimum
hidden width 4, maximum 20 successful removals, tolerance 0.01 and penalty lambda 2.
Targets use physical parameter reduction, not channel count or zero masking.
The unchanged exact reachability oracle includes remaining action horizon; an
infeasible initial target raises before validation access. No band widening occurs.
GCN and mean/root-weight GraphSAGE with two hidden layers are supported. Feature
and class counts come from the model; no Cora-specific geometry is hardcoded.

### Terminal reward semantics and scientific comparability

The inherited reward is `r = 0.6*delta_s - 0.4*max(0,A_dense-A_after) + P`.
`P=-lambda*max(0,abs(s-target)-tolerance)` is already included by the environment
once at an actual terminal transition. The planner never adds it again.
If the first action terminates, the path score is exactly `r1`; no second action
is considered and no fictitious future 0 is available as an alternative action.
A terminal second reward is discounted once, as `r1+0.9*r2`. A **nonterminal**
depth cutoff has zero bootstrap and no artificial terminal penalty or STOP. The
environment still has to reach the true target band within its remaining horizon.
The original terminal priority (`max_steps`, then no-removals, then band arrival)
is preserved. In feasible in-band paths, terminal penalties are zero (up to the
unchanged floating arithmetic at exact boundaries).

Thus depth 1 uses precisely the same immediate objective and tie rule as Greedy;
depth 2 uses the same rewards/discount/terminal convention as constrained Q's
two-transition return, but **not its full remaining-horizon learned objective**.
Q can bootstrap beyond two steps; Lookahead truncates that tail to 0. Replanning
does not make a depth 2 policy globally optimal. The repeated accuracy-drop penalty
charges the dense-to-current deficit at every step, not its increment: path length
and intermediate deficits matter, even for identical endpoint masks. Early terminal
paths have fewer reward terms. These are inherited semantics, not a new speed
bonus, terminal-quality objective, or incremental-drop reward. No terminal penalty
is imposed merely because a planning leaf remains off-band. Gamma is fixed at 0.9.

Use **V4-compatible constrained** controls: V3 `greedy_reachable_v3`,
`random_reachable_v3` and `q_reachable_v3` on common frozen inputs, or the V4
standard reference with identical settings. The legacy registered
`adaptive_greedy_local_v2` has different stopping semantics and is not an exact
control. Depth 1 equivalence is tested against native V3 constrained Greedy for
noninitial reachable targets; initial in-band behavior is inherited from V4.
Cross-method seed variation, validation reuse and search overhead must still be
reported. No claim of improved quality/cost follows from implementation tests.

## Reuse and honest resource accounting

Every first-action preview is shared across its continuations, and the chosen
preview becomes the next real state without another evaluation. Its already
evaluated second-level children become the next planning frontier. Reuse requires
**the identical unchanged concrete parent object**; its exact state, identities,
model object and reward inputs must also match. An equal aggregate key, widths or
accuracy bucket cannot enable it. This avoids retained-mask aliasing and reuses the exact
physical tensors, validation observation, incoming reward and history. Different
paths are otherwise evaluated and charged; there is no global transposition cache.
Only the previous selected frontier is retained, rather than whole search trees of
model objects. Serializable logs do retain every explored path and transition.

The evaluator contract is a deterministic, read-only validation callback on a
frozen model. Stateful/stochastic callbacks invalidate reuse; the integrated
workflow puts models in eval mode and uses validation labels only. No test metric
is passed to planning. Shared graph features/edges remain transductive, as before.

`candidate_evaluations` counts actual new preview callbacks; `validation_evaluations`
adds the one dense callback. `reused_evaluations` counts already computed
transitions revisited while replanning. `explored_candidate_transitions` includes
both, and `candidate_paths` counts scored leaves, which may share prefixes. Adoption
has no extra callback. Dense/final reporting evaluations are separately counted by
the frozen workflow (four). Scoring/surgery times sum work over **all newly explored
branches**, not just the chosen path. Wall search time includes validation, tree
construction, copying, logging and reuse lookup; it is not inference latency.

`LookaheadResult.trace` is the unchanged environment trace of executed actions.
`decisions` records every legal first action, selected first action, best proposed
path (whose second action is not committed), candidate paths/rewards/returns, and
all explored transitions with state, widths, accuracy, compression, retained IDs,
terminal reason/penalty and reuse flags. The candidate transitions disambiguate
cutoff leaves from real terminal states.

## Existing experiment interface

The new registry name is `lookahead_local_v4`; the existing `prune` command routes
it through the pinned-checkpoint workflow. Start from
`configs/experiments/lookahead/example.yaml`, supply verified checkpoint/split/graph
hashes and the matching three-layer architecture. Dataset selection can be Cora,
CiteSeer or PubMed; the tests use seven-node synthetic substitutes for all three
names, never their real datasets. The template contains placeholders and is not a
campaign launcher. Dataset support does not guarantee feasibility of every target
band at a particular width; infeasible configurations fail explicitly. Depth 1 is
selected with `lookahead.depth: 1`; depth 2 is default.
Unknown Lookahead options or gamma other than 0.9 are rejected. Shared tolerance,
minimum, horizon and lambda remain explicit in the existing `v2` settings block.

When a future run is separately authorized:

```bash
python -m gnn_pruning.cli prune --checkpoint /path/to/frozen.pt --config /path/to/pinned.yaml
```

Outputs use a fresh UUID directory under `local_structured_lookahead`, with a new
`LOCAL-STRUCTURED-LOOKAHEAD-V4` provenance identity. They include `pruned.pt`,
`metrics.json`, `config_snapshot.json`, `trace.json`, empty `q_table.json` and
`lookahead_decisions.json`. `PruningArtifacts.adaptive_trace_path` locates the
planning log; `rl_trace_path` locates executed deployment actions. Full validation
and test metrics are reported **after** search, as in the frozen workflow; candidate
decisions use validation accuracy only. Legacy `evaluate-pruned` split generation
is blocked for these pinned checkpoints, so use their frozen-split metrics.
No dense-training or finetuning pipeline is introduced by this integration.

## Verification

Focused pytest covers both architectures: an independent integer-width BFS and
algebraic depth 2 oracle at every replanned state; exact depth 1 action/trace/ID/tensor
and accuracy equivalence with constrained Greedy; stable first-action/continuation
ties; a specified three-level reward tree proving replanning can reject its old
second action; terminal and cutoff handling; remaining-one-action and infeasible
band/horizon; parent nonmutation; RNG preservation and determinism; safe concrete
frontier reuse and callback counts. Synthetic frozen-checkpoint tests verify strict
reload, original dense tensor projections, counts/compression, saved tree/trace and
provenance, and invariance to repinned nonvalidation-label changes.

```bash
python -m pytest tests/test_lookahead.py tests/test_lookahead_integration.py -r a
python -m pytest -r a
```

Final checks (2026-10-31): **43 focused tests passed; full suite: 492 passed,
2 skipped**. Both skips are existing CUDA device-preservation tests; CUDA is
unavailable on this CPU host. The 39 warnings are existing Torch JIT deprecations
under Python 3.12. [Verification receipt](lookahead_verification_20261031.json)
records environment versions, source-preservation hashes and exact test-log hashes.
Raw pytest logs and isolated plotting dependency supplements remain under
`${GNN_ARTIFACT_ROOT}/lookahead-implementation-20261031/`; installed environments
were not modified. Existing training smoke tests use synthetic fixtures only;
no dataset-scale dense training or pruning experiment was run.

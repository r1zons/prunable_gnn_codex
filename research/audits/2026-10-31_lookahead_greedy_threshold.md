# Lookahead threshold correction: independent audit

**Technical verdict: GO on the verified local CPU implementation. Full four-strategy
campaign: NO-GO until its comparison protocol is declared.** Threshold Lookahead/
native Greedy and historical exact-reachability Q have different legal/stopping
controllers. A campaign can study these method bundles with explicit controller
groups and achieved-compression controls, but cannot claim an isolated planning/
learning comparison under identical stopping. Random's controller also needs an
explicit campaign identity. No campaign, POLUS submission, push or merge occurred.

Work uses the canonical checkout on `audit/lookahead-greedy-threshold-20261031`,
starting at `70fe25d6e91d432646168bc263e193f15f45fc8f`. Both supplied files were
read as a proposal, not as authority overriding the user's scope. The script's
pinned-SHA/clean-tree dry run passed before application; guards were not bypassed.
Its resulting diff was reviewed, exercised, then corrected further. The user's
request for a commit SHA authorizes the final local commit; the attachment's own
"no commit authorized by this package" does not override that request. No push
permission is inferred from older turns.

## Findings and corrections

1. **Stopping changed as requested.** The source directly uses
   `LocalV2Env(greedy=True)`, not a rewritten approximation or V4 environment.
   The native `max_steps -> no_valid_actions -> target_reached` priority, physical
   surgery, incident-magnitude ranks, sizes, minima, reward coefficients and lambda
   remain unchanged. Threshold checking occurs after an action; native target-zero
   and initially nonprunable behavior are tested and preserved. Unattainable bands
   are attempted rather than rejected; horizon/channel exhaustion reports the gap.
2. **The supplied patch was not test-clean.** Its focused run produced five stale
   integration failures expecting `target_band`; they are corrected to actual
   native terminal reasons, not hidden by a fallback or relaxed reward check.
   The script's optional Cora skip was not counted as successful verification.
3. **Protocol separation was missing from the proposal.** Registry name
   `lookahead_local_v4` is retained, but new runs use provenance
   `LOCAL-STRUCTURED-LOOKAHEAD-GREEDY-THRESHOLD-V1`, explicit `greedy_threshold`
   metadata, and a separate `local_structured_lookahead_threshold` artifact root.
   This prevents silent pooling with the prior exact-band Lookahead.
4. **Overshoot is explicit.** Stats and saved reports include signed deviation,
   overshoot magnitude, beyond-tolerance flag, threshold-reached flag, actual
   compression and symmetric band violation. Native terminal penalties can now
   be nonzero. Independent tests verify exactly one penalty and correct discount;
   no extra penalty/bootstrap is added at a nonterminal depth cutoff.
5. **The proposed real-forward oracle was too dependent on environment logic.**
   The strengthened oracle never calls `env.preview`, environment legality/reward,
   or production saliency/rank helpers. It independently enumerates action sizes,
   sums incident absolute weights/biases, runs real GCN/GraphSAGE validation
   forwards, calculates physical compression, stopping, penalty and path return,
   then checks every logged node/path and selected first action at every replan.
   Only established Local Surgery is shared; separate existing tests validate it.
6. **The proposed Cora test could load/download the wrong graph order.** Its
   dataset/config loading was replaced with the unchanged archived-input verifier,
   pinned to the prior provenance manifest. Four checks use saved tensors and
   checkpoints, mask all nonvalidation labels, and compare depth 1 with native
   Greedy on the same runtime. No dataset download, dense training, depth-2 Cora
   search or empirical campaign was performed.
7. **Access, reuse and accounting were reviewed.** Synthetic integration tests
   preserve decisions/tensors under independently repinned nonvalidation-label
   changes. Callback counts equal dense+new previews; adopted/cached previews add
   no forwards. Logging distinguishes evaluated nodes, reused nodes and scored
   leaf paths, with physical counts. Concrete parent/state/mask/reward-input/horizon
   matching is retained. The planner now rejects constrained subclasses even with
   `greedy=True`; instrumented reward-tree tests substitute their toy type explicitly.
   Parent tensors/state remain unchanged; deterministic ties and replanning tests
   are retained. Arbitrary mutation of frozen weights or a stochastic evaluator
   remains outside the safe reuse contract.

## Actual test evidence

| Gate | PASS | FAIL | SKIPPED | Evidence |
|---|---:|---:|---:|---|
| Supplied patch as applied, before independent corrections | 40 | 5 | 1 | `proposed-patch-focused.log` |
| Corrected focused unit/integration suite | 53 | 0 | 0 | `focused-initial-corrected.log` |
| Corrected complete pytest suite | 506 | 0 | 2 | `full-initial-corrected.log` and JUnit XML |
| Archived Cora subset of complete suite | 4 | 0 | 0 | Four assertion-backed case records |

The archived subset is already included in 506, not an additional independent
sample. The original Cora skip is replaced by actual successful archived checks;
the final two skips are **only existing CUDA device-preservation tests** on this
CPU-only host. There are 39 existing Torch JIT Python 3.12 AST deprecation warnings.
Earlier 43/492 results refer to the prior exact-band implementation, not this fix.

Runtime: existing Python 3.12.9, Torch 2.2.2, PyG 2.6.1, NumPy 1.26.4, pytest 8.3.5;
two CPU threads. Existing task-local Matplotlib 3.9.4 supplements were reused for
legacy plot tests without changing installed environments. Existing training smoke
tests use synthetic fixtures only. Raw logs and XML are external under
`${GNN_ARTIFACT_ROOT}/lookahead-threshold-audit-20261031/`;
[verification.json](lookahead_threshold_20261031/verification.json) records exact
hashes, commands, per-test statuses, changed-source hashes and protected-source pins.

## Frozen Cora compatibility result

Both methods have **bitwise-identical final tensors and identical traces, retained
IDs, validation accuracy/F1 and callback counts** in every cell. The unchanged
source checkpoint, full graph and split hashes are checked before evaluation.
The [four compact case records](lookahead_threshold_20261031/frozen_cora.json)
retain actions, channel IDs, metrics and hashes; these are bounded correctness
checks, not a method ranking or new benchmark matrix.

| Model | Requested % | Actual parameter compression % | Widths | Outside symmetric band? | Search validation calls per policy |
|---|---:|---:|---|---|---:|
| GCN | 30 | 31.075782 | 22/32 | yes | 17 |
| GraphSAGE | 30 | 31.088578 | 22/32 | yes | 17 |
| GCN | 50 | 49.721251 | 16/32 | no | 17 |
| GraphSAGE | 50 | 49.741724 | 16/32 | no | 17 |

All end with `target_reached`. The 30% overshoots are respectively1.075782 and
1.088578 percentage points above nominal (tolerance 1pp); no band was silently
widened. The 50% endpoints are slightly below nominal but above the lower threshold.
The two extra forwards per case used for comparing final validation metrics are
reported separately from search calls. No test-label metric is computed in this
archived gate. These checks exercise the policy API with verified archived inputs;
the full frozen-prune CLI path is covered by synthetic integration tests.

## Remaining failures, limitations and campaign decision

**Remaining test failures: none.** GPU behavior remains unverified here because
CUDA is unavailable; no POLUS runtime/capacity preflight was attempted. GCN and
GraphSAGE checks use one frozen Cora checkpoint per model, training seed 101; no
independent-training or dataset-generalization claim follows.

Depth 2 still truncates future return at its planning boundary, whereas Q can
bootstrap further. Repeated dense-gap penalties and variable terminal path lengths
remain part of the existing objective. Removal of reachability changes candidate
sets and architectures, not just runtime. Equal target labels, matching reward
formulas and successful threshold termination are insufficient to establish matched
compression or an isolated strategy effect. Actual compression, widths, masks and
search cost must be reported; validation reuse is adaptive selection, not an
unbiased final generalization assessment.

**NO-GO for the full campaign as an identical-controller comparison.** Before
launch, preregister separate threshold/exact controller groups and their actual-
compression analysis, or separately authorize and verify new harmonized variants.
Do not change or relabel the historical Q evidence. This scientific protocol
boundary is the remaining campaign blocker, not a failing Lookahead test. The
present implementation/audit task is complete and does not authorize that launch.

## Changed files

- Policy/integration: `src/gnn_pruning/pruning/lookahead.py`,
  `registered_lookahead.py`, `workflow_v2.py`.
- Tests: `tests/test_lookahead.py`, `test_lookahead_integration.py`,
  `test_lookahead_frozen_cora.py`, and independent helper `lookahead_forward_oracle.py`.
- Usage: `configs/experiments/lookahead/example.yaml`, `docs/lookahead.md`.
- Audit evidence: this report and
  `research/audits/lookahead_threshold_20261031/{verification,frozen_cora}.json`.

Historical controllers, old verification receipts, experiment results and the
SITITO manuscript are unchanged. Supplied files remain unchanged outside the repo;
only their reviewed effect and exact hashes are recorded here.

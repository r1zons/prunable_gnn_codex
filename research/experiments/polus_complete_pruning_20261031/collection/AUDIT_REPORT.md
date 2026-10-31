# Independent audit — verified POLUS research dataset

**GO for descriptive scientific analysis**, subject to the explicit limits below.
Expected/collected/core-verified pruning conditions:**252/252/252**; dense models:
**18/18**. Per-run audit statuses:**92 PASS,160 WARNING,0 FAIL,0 NOT VERIFIED**.
Every WARNING retains verified identity, file integrity, physical tensors, final
metrics and applicable recorded-trace checks. It is not discarded from summaries.
This is not a claim of complete counterfactual/policy-optimality verification.

## Coverage and unit of replication

| Dataset | Model | Independent dense seeds | Expected runs | Core verified |
|---|---|---:|---:|---:|
| cora | gcn |3|42|42|
| cora | graphsage |3|42|42|
| citeseer | gcn |3|42|42|
| citeseer | graphsage |3|42|42|
| pubmed | gcn |3|42|42|
| pubmed | graphsage |3|42|42|

| Method | Expected | Verified | Physically pruned | Outside symmetric band |
|---|---:|---:|---:|---:|
| `static_random` |36|36|36|6|
| `static_magnitude` |36|36|36|6|
| `static_snip` |36|36|36|6|
| `sequential_random` |36|36|30|6|
| `adaptive_greedy` |36|36|36|18|
| `lookahead_depth2` |36|36|36|18|
| `tabular_q` |36|36|36|36|

The statistical unit is the independently trained dense checkpoint,**n=3 per
dataset/model/method/target**. Seed101 is one reused native training checkpoint,
not an additional platform replicate. Policy seeds101/202/303 are paired with
the dense seed for stochastic methods and are not separately varied. The1,080
Q episodes are not1,080 independent trained GNNs. No V1–V4, local CPU, Questions,
Flickr, OGB or unrelated course results enter the primary tables.

## Independent checks

The audit reads the **explicit submitted matrix**, verifies all55 acceptance IDs,
responses, launcher hashes/dependencies/resources against the pre-collection
committed receipts, and checks the deployment protocol/source archive hashes.
All717 pinned source files match; the five local model-code files used for
inference are byte-identical to their submitted versions.

All18 dense checkpoints: strict reconstruction, finite tensors, architecture and
seed, original saved split, graph/checkpoint/tensor hashes, disjoint60/20/20 node
coverage, recorded metrics. Twelve new training histories also pass epoch order,
validation-loss best-epoch selection, patience50 and time-accounting checks.
All six origins retain their separately identified20261024 source and input proof.
All six dataset/model groups have three distinct saved split hashes and model
tensor identities, consistent with the three independently trained conditions.

All252 final checkpoints: strict reconstruction and finite forward output with
correct node/class dimensions. Physical parameter counts and memory/serialized
sizes recompute. Direct tensor indexing, independent of production Local Surgery,
projects the dense retained rows/columns/biases onto every final state_dict;
all tensors match exactly, including output-layer parameters. Thus246 models are
physically smaller and six explicitly infeasible outcomes equal their dense
models. Retained original channel identities, not merely final widths, are checked.

Final validation/test accuracy and macro-F1 are recomputed using **original frozen
graph tensors and splits read from checkpoints**; no split generator or dataset
loader is called. Existing local CPU Torch2.2.2/PyG2.6.1,Python3.12.9,two threads
verifies native POWER8 Torch2.0.1/PyG2.4.0,four-thread outputs. Final metric absolute
tolerance is1e-12; maximum observed difference is2.220446049250313e-16. Integer
parameter counts and tensor projections are exact; compression tolerance1e-12.
Recorded intermediate accuracies are float32 means, so reward/drop arithmetic
allows1.2e-7 absolute error, independently chosen for float32 quantization. Q
updates and recorded Lookahead returns use stored rewards with1e-12 tolerance.

No training, saliency search, pruning rollout, policy evaluation experiment,
new timing measurement or scheduler job was executed. Only the270 final frozen
models were independently inferred per complete audit pass. Three separate audit
passes are preserved; the second adds submission anchors, resource and training
checks; the third also verifies static criterion identities and stop-reason metadata. No observation was repaired or replaced.

## Method-specific verification and limits

- **Static methods:** retained IDs, minimum widths, physical counts, target/fallback
  semantics, frozen projection and final metrics checked. SNIP means post-training
  train-node abs(weight×gradient). Full saliency/width optimization is not rerun;
  per-parameter SNIP scores were not saved. A static mask with the same width but
  different original IDs fails projection verification.
- **Sequential Random:** each stored action is legal under an independent exact
  width/horizon reachability calculation; seeded uniform legal-action choices,
  magnitude channel deletion, rewards and terminal reasons match. Six
  CiteSeer/30% conditions are genuinely infeasible in the exact band and retain
  dense widths32/32. They are valid recorded failures to prune at that target,
  not successful30% compression and not missing/corrupted experiments.
- **Greedy:** legal actions, magnitude IDs, chosen rewards, counts and native
  threshold/max-step/no-action stopping match. **Unselected candidate observations
  were not saved for36 runs**, so the Greedy argmax cannot be independently
  re-established without a forbidden rerun. Source and preflight evidence support
  the executed implementation; the subcheck remains NOT VERIFIED, not fabricated.
- **Lookahead:** all36 recorded depth-2 trees,4,175 candidate leaf paths, first-action
  execution/replanning, immediate+.9×next reward, terminal handling, stable ties,
  retained IDs, reuse and actual evaluation counts pass. This checks recorded
  counterfactual scores; their intermediate accuracies are not newly inferred.
- **Q:**30 learning episodes per run reset dense; deployment resets again and the
  table is frozen. The17,930 stored training transitions independently reproduce
  seeded exploration, legal actions, rewards, Bellman updates, all13,474 final Q
  entries, coverage and deployment argmax/termination. No episode or rollout is
  newly generated. Aggregate states omit retained masks and do not establish an
  MDP guarantee. These are **native V2 soft-target Q**, not constrained V4 Q.

Total recorded validation-oracle calls:**24,698**, independently reconciled.
Candidate enumeration, oracle calls and full scoring/training forward/backward
work are different quantities: static SNIP has zero validation-oracle calls but
nonzero scoring/search cost; Q candidate enumeration is not applicable/missing,
while its chosen-transition oracle calls are measured.

## Test-label isolation and comparability

Pinned executed source masks test labels before scoring/search; SNIP receives
train indices, and adaptive methods receive only the validation callback. Test
metrics follow completed decisions. Frozen labels also serve graph identity and
known class-count metadata; that is distinct from test-performance feedback.
The audit is based on source/path/launcher hashes and preserved native preflight
isolation assertions, not a retroactive runtime taint trace. No test metric changes
the protocol, method choice, report inclusion or plot selection.

Controllers differ by design. Greedy/audited Lookahead use lower-threshold
stopping with overshoot; static methods use the symmetric band/nearest fallback;
Sequential Random enforces exact reachability; native V2 Q permits STOP and does
not stop merely on crossing a target. **96/252 final models fall outside the
symmetric±.01 band**:18 static fallbacks,6 infeasible Random,18 Greedy,18 Lookahead,
36 Q. These warnings are preserved. Requested30/50% does not imply equal achieved
compression, widths or masks. This dataset supports comparisons of existing
method/controller packages, not an isolated causal effect of planning/learning.

## Missing measurements and unresolved issues

Critical failures/blockers:0. All per-run warning codes and counts are in
[audit_runs.csv](audit_runs.csv); severities and limitations are in
[unresolved_issues.json](unresolved_issues.json).

| Severity | Evidence limitation | Scope / consequence |
|---|---|---|
| WARNING | Greedy unselected candidate values absent |36 trajectories: argmax not independently verified |
| WARNING | Original fit wall absent for reused dense models |6 models/84 linked rows: missing, not0 |
| WARNING | Standardized dense inference latency absent |18 models: no dense-to-pruned measured speedup |
| WARNING |20261024 origin graph digest not contemporaneous |6 origins: preserved raw/processed lineage reconstructed; current frozen hashes complete |
| INFO | Raw inference latency samples absent |252 mean/std/pass-count records: consistency checked, no new timing |
| INFO | Worker RSS is cumulative high-water |252 rows across18 workers: not incremental per-method memory |
| INFO | Static saliency/optimization and intermediate forwards not rerun |Clearly separated from verified masks/recorded arithmetic/final outputs |

All target deviations and infeasible/fallback conditions remain in canonical
results. `summary.csv` includes n_expected,n_collected,n_verified,n_fail,n_warning,
physical/target-band counts and stop reasons for all84 dataset/model/method/target
cells; means/sample SD use the three independent dense seeds. Missing quantities
remain empty, never imputed. Dense training time repeated in per-run provenance
must not be summed fourteen times; unique dense records live in `dense_audit.csv`.

Search/pruning wall is separate from measured inference mean/std (5 warmup,20
timed passes). Timing verification checks recorded units, nonnegative values,
pass counts and scheduler/resource consistency; it does not reproduce latency.
Actual exclusive20-slot allocation and active/waiting/CPU time are separated in
[the collection report](COLLECTION_REPORT.md) and [resources.json](resources.json).

## Tables, figures, reproducibility and tests

- [Canonical per-run dataset](verified_results.csv):252 rows, including all warnings,
  dense/pruned validation/test accuracy/F1 and differences, exact compression,
  dimensions/original IDs, parameter/checkpoint sizes, measured costs, audit and
  provenance. Architecture is GCN/GraphSAGE; all dense models are3layers,width32,
  dropout.5, as pinned in the protocol. Six infeasible rows remain explicit.
- [Summary](summary.csv):84 cells; [dense verification](dense_audit.csv):18;
  [job accounting](job_status.csv):55; [machine audit](audit.json):full compact scopes.
- Figures:[accuracy/compression](figures/accuracy_compression.png),
  [macro-F1/compression](figures/f1_compression.png),
  [quality/search cost](figures/quality_search_cost.png),
  [coverage/physical pruning](figures/coverage.png). All use validation metrics;
  no scientific winner, test-based selection or tuning is proposed. Overlapping
  plotted points do not prove identical masks; the CSV keeps original IDs.

Focused pytest:**38 PASS/0 FAIL/0 SKIPPED**. Full pytest:**585 PASS/0 FAIL/6 SKIPPED**.
Skips:2 CUDA device checks (no CUDA),4 opt-in archived-Cora Lookahead checks not
enabled in this invocation. They are not reported as passes. Adversarial tests
cover missing/corrupt/duplicated/inconsistent outputs, altered metrics/rewards/Q
values, terminal tree extensions, false counts, equal-width different masks,
source/seed/leakage metadata, missing essential evidence and accounting status.
[Test log and audit-code hashes](tests.json) identify the exact evidence.

Rerun from this canonical checkout, using the existing ML environment and fresh
output directory (never overwrite raw evidence or earlier audits):

```bash
export PYTHONPATH="$PWD/src"
export COLLECTION="$GNN_ARTIFACT_ROOT/polus-complete-collection-20261031T110939Z"
python -m gnn_pruning.experiments.complete_audit \
  --root "$COLLECTION/raw" --source "$COLLECTION/source" \
  --inventory "$COLLECTION/remote-file-inventory.json" \
  --accounting "$COLLECTION/private/bhist.txt" \
  --output "$COLLECTION/audit-rerun-NEW"
python -m pytest tests/test_complete_campaign_audit.py -ra
```

The preserved committed pre-collection submission reference is checked by the
script. It never calls a split generator, dataset downloader, training/pruning
runner or bsub. Further report/plot generation only reads audited JSON/CSV.

**GO:** analyze this verified native dataset descriptively, with achieved
compression, paired dense seeds and the recorded missingness/controllers.
**NO-GO:** infer a universally superior strategy, treat requested targets as
matched physical models, claim independently proved Greedy argmax, or compute
unmeasured dense inference speedups. No additional experiment is authorized here.

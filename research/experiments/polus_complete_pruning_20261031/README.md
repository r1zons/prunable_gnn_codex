# Complete POLUS pruning campaign — 2026-10-31

Status: **PREFLIGHT, NOT SUBMITTED**. This is a new campaign; historical V1–V4
source and results are not overwritten. Preparation branch:
`campaign/complete-pruning-20261031`, starting at
`376610cdba94e62c98d658f60c12a4091702bf42` (Lookahead threshold audit and Questions
integration included). Questions is explicitly excluded from this campaign.

## Authorized matrix

Cora, CiteSeer, PubMed × GCN, GraphSAGE × dense seeds 101, 202, 303:
18 independent dense conditions. Three layers, hidden width 32, dropout 0.5,
Adam learning rate 0.01, weight decay 0.0005; maximum 500 epochs, patience 50,
selection by validation loss. Existing deterministic generated 60/20/20 splits
use the dense seed. Targets 0.30 and 0.50; seven methods give 252 pruning runs.
No pruning-time fine-tuning; test metrics are reporting only after decisions.

Intended method families: Static Random, Static Magnitude, post-training Static
SNIP, Sequential Random, Adaptive Greedy, depth-2 Lookahead and tabular Q-learning.
Before an executable manifest is approved, each family must map explicitly to an
existing implementation. The existing Sequential Random implementation is
`random_reachable_v3`: it samples legal **actions** and chooses channels by
magnitude; it is not random-mask Static Random. Corrected Lookahead and native
Greedy stop at the lower compression threshold and permit overshoot; constrained
V3/V4 Random/Q variants use exact reachability. These are different controllers.
No harmonized replacement is authorized by this campaign.

## Required gates and submission boundary

1. Verified SSH access and current native runtime/activation, queue/node/user limits.
2. Explicit implementation/configuration map and exact checkpoint-reuse provenance.
3. Unit/regression checks; immutable executable source and input hashes.
4. Resource decision with explicit per-job and aggregate bounds.
5. Training → strict checkpoint validation → pruning dependencies (`done`, not
   `ended`); a bounded Cora seed-101 technical pilot gates remaining work.
6. INTENT before each `bsub`, immutable acceptance response and job ID afterward;
   ambiguous attempts are never silently retried.

No bulk submission is allowed while any required gate is unresolved. Once all
jobs are accepted, stop without monitoring completion, collecting results or
scientific analysis. Preparation may be committed locally; push/merge forbidden.

Raw, private connectivity/preflight evidence is outside Git at
`${GNN_ARTIFACT_ROOT}/polus-complete-pruning-20261031/`. Access values and account
names must not enter tracked reports or manifests. Submission IDs, if any, will
be recorded separately from completed scientific results.

## Explicit implementations and scientific differences

The executable protocol maps the six public V2/Lookahead methods directly and
adds an orchestration adapter calling the unchanged V3 sequential Random control.
Tabular Q is **`q_learning_local_v2`, its existing 30-episode default** (alpha .1,
epsilon .2, gamma .9), not a silently relabelled constrained V4 policy. Lambda 2,
minimum width 4, maximum 20 successful actions, tolerance .01 and magnitude-based
adaptive channel selection match the checked Lookahead/Greedy configuration.
The action menu is {1,2,4,8}. Policy seed equals its dense-condition seed; there
are no additional policy-seed repetitions in the 252-run matrix.

- Static Random samples an eligible width pair and channel subsets; Magnitude
  uses fixed dense saliency; SNIP is **post-training**, train-label abs(w*gradient).
  Static methods search the symmetric band and use their existing nearest-width
  fallback when the discrete band is empty.
- Sequential Random samples reachability-legal actions uniformly and uses
  magnitude channel selection. Exact-band infeasibility is a recorded zero-action
  `initial_target_infeasible` outcome, not a silently widened band or new policy.
  In particular, CiteSeer hidden32 at nominal 30% has no exact-band width pair.
- Native Greedy and corrected Lookahead use threshold stopping, permit overshoot,
  and pay the unchanged once-only terminal deviation penalty. Lookahead plans
  depth 2, executes one first action and replans, discounting the second reward .9.
- Native V2 Q permits post-removal STOP and uses a soft target: reaching/crossing
  the target alone does not terminate it. It retains the repeated dense-to-current
  accuracy-drop penalty and sparse-Q deterministic ties.

Therefore this is a comparison of **existing method/controller packages**.
Requested targets do not make achieved compression, masks, action spaces or
terminal reward placement identical. It cannot isolate learning or planning as
one causal factor across all seven methods. No parameter is selected from pilot
accuracy and no method is retuned to meet the requested target.

## Execution layout and bounds

55 nodes: 18 train-or-verified-reuse, 18 validation, 18 pruning batches (14 tasks
per dense condition), and one technical pilot gate. The two Cora seed101 batches
are the 28-run pilot and are part of the 252, not extra experiments. The pilot gate
releases the other 16 train/reuse nodes, which release their own validation and
pruning nodes. Dependencies use `done(id)`, with additional runtime certificates.

Proposed limits, subject to current native preflight: four slots/threads, normal,
exclusive polus-c4, span one host, memory limit 8192 MB and rusage[mem=8192].
The site confirms `LSF_UNIT_FOR_LIMITS=MB`, `ABS_RUNLIMIT=Y`; C4 has 1 TB RAM.
Train/reuse 10 min, validation 5 min, gate 5 min, pruning Cora/CiteSeer/PubMed
20/30/45 min per batch. **845 minutes maximum aggregate requested active wall**
(14 h 5 min), 56.33 allocated slot-hours; exclusive-node occupancy additionally
reserves idle cores. Queue waiting is separate. Every job is below normal's
180-minute maximum. The prior V4-only five-hour budget is not reused as a limit
for this separately authorized complete campaign.

The six native frozen-model forward checks measured mean .0092–.0489 s.
A conservative per-condition search bound is 4548 validation forwards (two
targets: Q ≤651, Lookahead ≤1441, Greedy ≤161, Random ≤21 each), excluding
saliency/autograd, Python/surgery, I/O and reporting. Final inference adds 350
passes per batch (14 × [5 warmup +20 timed]); validation/test reporting is
separate. These are operation bounds, not guaranteed wall forecasts. The proposed
20/30/45-minute reservations include substantial overhead allowance relative to
native checks and prior 1:33–4:57 whole-condition dense+14-prune V1 jobs. Expected
active runtime is approximately 1–3 hours, with substantial uncertainty for new
Lookahead searches and 12 new dense fits. The hard reservation, not this forecast,
bounds scheduled consumption.

Raw checkpoints and full traces are external. Each run saves final validation/test
accuracy and macro-F1, actual counts/compression/widths/original IDs, overshoot,
trace/training/Q-table/candidate decisions, observed validation-forward counts,
strict reload and original-channel tensor projection, search wall time, inference
mean/std, peak RSS, graph/split/checkpoint/source identity and output hashes.

## Check status and collect later (separate user-authorized session)

With the user's existing research-account SSH configuration:

```bash
ssh -l "$POLUS_USER" polus.hpc.cs.msu.ru
source "$HOME/gnn-polus-20261023/activate-polus.sh"
cd "$HOME/gnn-polus-20261023/results/polus-complete-pruning-20261031"
python -c 'import json; d=json.load(open("submission-result.json")); print(" ".join(str(j["job_id"]) for j in d["jobs"] if j["accepted"]))'
bjobs -a -u "$USER" -w
bjobs -p -u "$USER" -l
# For one accepted ID, use: bpeek JOB_ID; bhist -l JOB_ID
```

Do not resubmit this campaign on restart: every intent/receipt is a claim, including
ambiguous attempts. A later collection may archive the entire new root, hash it
on POLUS, transfer it and compare the SHA before extraction into a fresh local
artifact directory. Then use `verify_completed` on every expected run and require
exactly the manifest's 252 run IDs; do not count submission receipts as results.
A portable collection example, **not executed in the launch-only session**:

```bash
cd "$HOME/gnn-polus-20261023/results"
tar czf polus-complete-pruning-20261031-collected.tar.gz polus-complete-pruning-20261031
sha256sum polus-complete-pruning-20261031-collected.tar.gz
# On the Mac, with fresh destination and existing SSH account variables:
scp "$POLUS_USER@polus.hpc.cs.msu.ru:gnn-polus-20261023/results/polus-complete-pruning-20261031-collected.tar.gz" "$FRESH_COLLECTION/"
shasum -a 256 "$FRESH_COLLECTION/polus-complete-pruning-20261031-collected.tar.gz"
```

Raw artifact verification uses the pinned executable snapshot's
`gnn_pruning.experiments.complete_campaign.verify_completed(Path(run_directory))`.
Failures, missing completion markers and infeasible/target-miss scientific outcomes
must remain distinct. No automatic repair, algorithm change or replacement run is
part of collection.

# POLUS V4 controlled tabular-Q diagnostic — 2026-10-29

Researcher-approved implementation, technical preflight and real LSF submission.
Scientific settings and extension rules are fixed before looking at V4 accuracy.
This is a launch protocol; acceptance is distinct from completed experiments.
V1/V2/V3 source/evidence and the SITITO manuscript remain unchanged.

## Isolation and frozen inputs

Branch `research/v4-tabular-diagnostic-20261029`, separate adjacent worktree
`prunable_gnn_v4_20261029`, based on `70854ce`. New source files only:
`pruning/experimental_v4.py`, `experiments/v4_campaign.py`; new tests and scripts.
Existing Local Surgery, magnitude ranking, V2 state, frozen-input verification,
validation metrics and atomic JSON conventions are reused without edits.

The V3 Cora inputs are copied byte-for-byte into a fresh external V4 root. Both
three-layer dense models have widths32/32, training seed101, F1433/C7. Graph,
ordered splits, checkpoint bytes/tensors and original recorded lineage are checked.
No dataset download, training, fine-tuning, packages, environments, network routes
or credentials are changed. SSH/taskroot/environment/LSF are inspected anew.

The graph tensor SHA256 is
`2b2bc996be89812f80ce7d09599dab983f804f87ddd37a09b3ce0b5689ec9a0d`;
split file `59a0209a8e2a73a5e70a720d327f8c024d36d184a6d00be820ced0e1a3eb08bc`;
GCN checkpoint `276b2ff0f7790a84afc2755e89d88f1a170865a4397e4355daf7552c7075c129`;
GraphSAGE checkpoint `09b289dcbaddc1afb1c3660f7fa5d60fb1109c3c3059bdae14d8c4de03034774`.
Historical processed-graph identity remains reconstructed from recorded raw
lineage; no contemporaneous training digest is invented.

## Predeclared scientific matrix

All conditions: Cora GCN/GraphSAGE, targets0.30/0.50, exact tolerance0.01,
minimum width4, max20 successful pruning actions, lambda2, alpha0.1, gamma0.9,
epsilon0.2, 100 episodes unless explicitly changed below. Unseen legal Q=0.
State is exactly the V2 tuple: version, two widths, remaining horizon,
floor(validation accuracy/0.01), terminal. No retained-mask state extension.

| ID | Single changed factor | Comparator |
|---|---|---|
| `q_v4_menu12` | Menu {1,2} | V3 constrained Q {1,2,4,8} |
| `q_v4_percent` | Width-dependent ceil(5%w)/ceil(10%w) | Same baseline |
| `q_v4_300` | 300 episodes | 100-episode baseline |
| `q_v4_decay300` | Linear epsilon0.4→0.05 over300 episodes | `q_v4_300` |
| `q_v4_random_ties` | Seeded uniform choice among equal maxima | Deterministic baseline ties |
| `q_v4_incremental_drop` | Reward0.6Δs−0.4(D_t−D_prev) | Repeated-drop baseline reward |
| `q_v4_gamma0` | gamma0 | gamma0.9 baseline |

Primary: seven Q variants ×two models ×two targets ×policy seeds101/5001/5002
=84 Q tasks. Eight deterministic Greedy controls: both alternative menus ×four
model/target cells, seed101 once per cell (seed does not affect Greedy).

Percentage actions use integer ceiling `(p*w+99)//100`, p=5 or10, on the CURRENT
width of the selected layer. Deduplicate, sort ascending, and reject deletions
below minimum width; do not clip an invalid k into a different action. Examples:
w32→{2,4}, w30→{2,3}, w10→{1}. Layer0 actions precede layer1. For every menu the
width graph is a directed acyclic graph. Shortest legal paths are computed with
the menu recomputed after each deletion, not fixed coin-change arithmetic.
Enumerate all exact integer-parameter goal pairs, then add independent per-layer
distances and compare against remaining successful-action horizon. The filter
uses neither validation nor labels. Off-band STOP is excluded; arrival in the band
terminates. Original max_steps/no_valid_actions priority is preserved; the terminal
penalty is applied once. Infeasible initial targets raise an explicit error.

C uses `epsilon(e)=0.4+(0.05−0.4)*e/299`, e=0..299; final deployment epsilon0.
All other learning conditions keep epsilon0.2. E uses the same frozen dense
accuracy to define D=max(0,A_dense−A); D_prev is from the preceding transition,
initially0. Recovery can yield a positive reward component. Terminal penalty and
gamma0.9 remain unchanged. F changes no setting beyond gamma.

Snapshots at25/50/100/200/300 completed episodes, when within the run budget,
save a copy of Q and the Python RNG state. They perform no policy rollout, metric
evaluation or RNG draw. They therefore cannot perturb subsequent training.

## Reuse and baseline provenance

`q_v4_reference` is an unscheduled regression entry point, reproducing V3
`q_reachable_v3`. Synthetic and bounded two-episode real-Cora tests compare exact
training/deployment traces, Q values and physical tensors. Existing full100-episode
V3 baseline results are referenced and hash-verified, never rerun to recreate data.

V3 has12 local baseline outcomes (three policy seeds) and four POLUS outcomes
(seed101 only). Platform labels are retained. The existing POLUS seed101 is the
matched remote baseline; local5001/5002 are developmental references and must not
be represented as completed POLUS baselines or pooled across hardware. No extra
baseline conditions are silently added to the approved84 new Q tasks. Optional
5003/5004 baseline outcomes do not exist and are not invented.

## Predeclared optional extension and budget

Optional policy seeds5003 AND5004 are added to ALL seven new Q variants in all
four model/target cells, or neither seed is added anywhere:56 additional Q tasks,
total140 Q+8 Greedy=148. Greedy is not replicated. The decision uses costs only,
never achieved compression, validation accuracy or ranking of methods.

One normal/exclusive/four-slot C4 capacity job has a15-minute LSF limit. It checks
the V3 reference and runs every new Q condition at seed101 for the first30 of its
planned episodes, plus the eight Greedy controls. C retains its300-episode decay
schedule during truncation. These labelled technical gates are not full-matrix
scientific outcomes and are not substituted for the main tasks.

Per-cell full cost = measured search×episode ratio + measured non-search worker
overhead (at least15s). Double this estimate for planning. Every group must fit
inside80% of its Python execution budget. First qualify the complete92-task primary
matrix; extend uniformly only if the complete148-task matrix also fits. Otherwise
omit the extension or block an unsafe primary matrix, rather than submit a partial
scientific selection. All36 capacity cells must be present and completed.

Four main jobs, one per model/target, each have70-minute LSF limits and one worker
with four Torch threads. Batch software stops120s before its LSF limit; each task
has a maximum1800s timeout, further reduced by remaining job budget. Including
the15-minute preflight, aggregate requested active LSF wall time is295 minutes,
strictly below300. An INTENT ledger reserves even ambiguous submissions; repeated
preflights would consume budget and require a new capacity calculation. Queue
waiting is separate, not counted as active job time. Exclusive C4 reserves the
whole host as in V3 even though the scientific process requests four CPU tasks.

The requested4–5h is a capacity-planning target, not permission to pad idle time,
repeat identical runs or add methods. Measured expected time and conservative
planning allowance are recorded separately; actual active time may be lower.

## Verification, execution and outputs

Unit tests use independent exhaustive width BFS, explicit Bellman equations,
telescoping reward checks, controlled-factor comparisons, RNG/snapshot invariance,
V3 equality and immutable submission-claim tests. Relevant V1/V2/V3 regressions
and local real-data smoke precede remote scientific submission. Existing frozen
loader masks EVERY non-validation label to−1. No test metric is computed.

Both source and launcher pins must be clean. All input/config hashes and initial
feasibility are checked before bsub. The current discovered activation file is
hash-pinned and checked by the LSF script. Separate remote worktree; historical
repo and V3 worktree are left unchanged. Normal queue/C4 status and current user
jobs are rechecked. INTENT is written before bsub; accepted response/ID is saved
in an immutable per-job receipt and incremental ledger snapshots. Ambiguous
responses are UNRESOLVED and never automatically retried.

Each task saves physical checkpoint, exact widths/count/parameter bytes/compression,
retained IDs, dense/final validation accuracy+macro-F1, search/total time,
evaluation count, full training/deployment action+reward traces, per-decision
legal Q values/unseen choices/Bellman before+after, Q coverage, stopping reasons,
snapshots, source SHA, training/policy seed, platform/runtime and frozen input hashes.
Strict reload must reproduce logits; tensors must equal a physical projection
from the original dense retained channels. Completed markers hash every output.

Raw artifacts are outside Git at `${GNN_ARTIFACT_ROOT}/polus-v4-diagnostic-20261029/`
and `${POLUS_TASK_ROOT}/results/polus-v4-diagnostic-20261029/`. Only implementation,
tests, config/gate indexes and acceptance receipts are versioned. After the main
submissions, report accepted versus blocked conditions and exit without claiming
scientific findings or waiting for the campaign to finish.

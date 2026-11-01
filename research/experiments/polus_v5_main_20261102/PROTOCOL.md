# POLUS V5 main campaign — fixed protocol

Authorized scope: Cora/CiteSeer/PubMed × GCN/GraphSAGE × Dense training/split
seeds101/202/303 × requested physical compression30/50% × exactly seven V5
methods =252 conditions,18 frozen original Dense checkpoints. Three layers,
hidden32,dropout.5; original500epoch/patience50/lr.01/weight decay.0005 Dense
provenance, **no new training or fine-tuning**. Policy seed equals the Dense/split
seed for the three stochastic methods; it is an identity, not an extra replicate.

Scientific authority is unchanged `pruning/budget_v5.py`: minimum4, sizes1/2/4/8,
bias-inclusive integer parameter geometry, exact reachable continuations, unit-
deletion horizon56, symmetric target band±.01; if empty, global closest physically
smaller widths, explicitly marked fallback/target miss. Static methods choose a
complete fixed-rank plan; adaptive methods share V5 legal transitions. Random
static plans differ from sequential reachable random with magnitude channels.
Greedy maximizes immediate reward; Lookahead exhaustively maximizes r1+.9r2,
executes one action and replans, stable first ties and safe preview reuse.

Q trains30 reset-to-Dense episodes (alpha.1,gamma.9,epsilon.2), then resets again
for a separate epsilon-zero deployment. State contains exact retained original
IDs, widths, remaining parameters/horizon, target, tolerance, accuracy bin and
terminal flag. Reward is .6 times incremental compression minus .4 times the
current nonnegative Dense-accuracy drop, plus one terminal band-deviation penalty
with lambda2. Original Bellman update and deterministic first tied maximum remain
unchanged. An unseen action has value0, which can beat learned negative values.
A nonempty table or target attainment is not evidence of policy learning.

These methods are new V5 conditions, **not relabeled historical V2–V4 outcomes**.
Historical threshold Greedy/Lookahead results used other stopping rules; no
cross-version paired equivalence claim is made.

Input preparation reads the archived graph/checkpoint only for identity
verification and copies the original checkpoint bytes and original saved split.
Working graphs have every test label replaced by−1 before any current inference.
No split regeneration, graph preprocessing, test reporting, dataset download,
architecture tuning or accuracy-based gate. SNIP uses training labels only; all
adaptive decisions use validation accuracy only. Test-only reporting requires a
separately frozen future protocol.

Technical preflight: two normal/exclusive C4 jobs,1 and4 threads, sequential fresh
processes, same seed101 inputs for all three datasets/both models/both targets/all
seven methods:84 profiles per thread,168 total. Q is **technical3 training episodes
and2-action deployment prefix**, other methods finish; technical outputs are
never counted among252. All18 Dense identities/validation metrics are checked
natively before each technical job. Choose a common thread count using measured
search/runtime costs only, after checking cross-thread scientific decisions.

Production:18 safe job groups,14 independently isolated condition subprocesses
per frozen Dense input; no concurrency inside a job. Explicit wall/memory bounds
and fresh252-row resource gate required before any production bsub. INTENT is
created before each single bsub attempt; ambiguous responses stop submission.
No automatic retry of accepted, ambiguous or completed jobs/condition directories.

Resource forecast (declared before measurement): per condition, maximum over six
seed101 dataset/model strata at the same method/target; Q search scaled31/3 from
3 complete episodes plus a2-action deployment prefix (conservative deployment
allowance), all searches ×2; max observed non-search worker time ×2 plus10s.
Memory3×maximum native peak RSS +2GiB, rounded upward. Both target costs measured
separately. All seeds use the cross-stratum envelope; no seed-dependent quality
selection. Each group sum+120s must be≤170min and RAM≤8GiB; requested wall is
ceil(forecast/minute)+5min (minimum10),≤180. All production jobs on the measured C4
host; no unsupported cross-host speed assumption. Normal queue's EXCLUSIVE policy
means actual20 allocated slots despite4 requested/1 or4 active threads. Queue
waiting is separate from active compute; scheduling throughput is not promised.

Reporting: search wall/CPU, true callback requests/forwards/cache hits, candidate
transition requests (V5 counter, may reuse exact forward cache), full validation
forwards used for reporting separately, SNIP gradient-forward count separately;
complete traces/Q/candidate paths, final physical checkpoint and original IDs,
termination/fallback/target miss. Inference is5 warmup+20 uncached full-graph CPU
forwards, eval mode, within the recorded host/thread allocation. Static partial
layer-deletion traces have no intermediate validation measurement; none is invented.

Analysis pairs within Dense checkpoint/dataset/model/seed/target. Three Dense
seeds per dataset/model; episodes and targets are not independent training units.
Use accuracy/F1 alongside achieved compression and actual search cost. Report
all failures/target misses without removing unfavorable outcomes. Q coverage,
unseen alternatives, deterministic ties, exploration and observed Bellman values
support mechanism descriptions; no unrecorded counterfactual validation scores
or significance claims.

The approved asynchronous continuation uses an additional normal/exclusive C4
release job (90min/8GiB/one active thread), depending on **ended** for both native
preflight jobs. It independently requires bhist **Done successfully** for both;
failed/ambiguous/partial evidence creates a blocked record and submits no research.
A separately pinned reporting/controller file invokes the immutable scientific
snapshot. Source/protocol/activation/controller/analysis hashes are checked before
execution. Current queue/users/host state is queried again at release. Only a
complete PASS gate produces the252-row measured resource manifest and18 research
bsub requests. Every acceptance is recorded exactly once. Partial/ambiguous
submission stops; it never silently retries.

After18 successful acceptance responses, a separate90min/8GiB C4 verification job
is submitted depending on ended for all18; it requires successful LSF history,
all252 final artifacts, exact hashes/projections/reward/decision/Q arithmetic and
validation reproduction before writing a VERIFIED marker or running article-table/
figure generation. A failed verification preserves raw evidence and no scientific
completion/analysis success is inferred. This reporting work performs no training,
policy search, test evaluation or new scientific condition. Local collection only
proceeds after verification-job success; original remote files are read only.

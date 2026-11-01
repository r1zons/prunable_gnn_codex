# Controlled Flickr CPU thread scaling — predeclared technical protocol

Scientific base: campaign/flickr-structural-pruning-20261101, checked out from
db3fa65; budget-V5/cache/surgery scientific code must match executed1b8b8b7 byte
for byte. New resource-only adapter and separate artifact root. No CUDA, dense
fitting, fine-tuning, multiprocessing/parallel Q, continuation or full56 launch.

## Allocation and controls

Three bounded normal-queue/exclusive/single-host jobs: C2 and C3 topology only
(1 slot,3min,8192MB each), one C4 benchmark (20 slots,175min,32768MB). C4 must
actually expose at least20 physical cores and20 allocated LSF slots; otherwise
fail before numerical work. Topology records physical core/socket IDs, all visible
SMT siblings, NUMA, scheduler variables, actual affinity and allocation receipt.
C4 is occupied at the initial scheduler check; queue waiting is not compute time.
No cancellation, bypass or assumed concurrency.

Choose exactly one permitted logical sibling per physical core. All configurations
share the same20-core CPU affinity; OpenMP per-thread binding is disabled and
memory policy stays default. This placement differs from the historical broad
160-logical-CPU visibility and is recorded, not silently equated with it. New4
threads is the matched timing reference; the old4-thread scientific trace is an
additional independent gate. No hardware-frequency ranking is assumed.

Eight **serial fresh Python processes** in declared order: threads4/8/12/20,
then targets30/50% within each. Each gets the same frozen graph/checkpoint;
Torch intra-op, OpenMP, BLAS and NumExpr counts set to N, Torch inter-op1,
OMP_DYNAMIC=FALSE and OMP_MAX_ACTIVE_LEVELS=1. No concurrently running worker.
Record library thread pools after warmup/search and every live thread's actual
CPU mask. An unexpected numeric pool above N fails the condition.

Each child has1200s maximum including setup/measurement/reporting. Timeout or
failure is an explicit incomplete benchmark, never a shortened completed episode.
Eight child ceilings total160min; orchestration/metadata reserve fits175min,
below absolute180. No automatic rerun or ambiguous submission retry.

## Frozen workload

GraphSAGE4layers/256hidden, Dense/split/policy seed101. Exact original Dense
checkpoint SHA dcbf0722a61842ec2db4f24491e90ffaddabea0749ce411898110d3025fb5566.
Split tensor/protocol pins remain those in the original resolved protocol. Test
labels are -1 in the only graph loaded; original test labels are never transferred
or evaluated. Strict source/input/checkpoint hashes, nonmutation and retained-ID
projection required.

Uncached dense and fixed representative physically pruned forwards:2 warmups,
7 timed full-batch repetitions per model/process. Pruned artifact is the prior
static magnitude50% checkpoint: widths[111,230,219],266433parameters,
SHA b4a75142b2167ee67997ed1794fc88286704caa35aca13f88941a74a3ab46fc1;
verified tensor identity/projection, not regenerated saliency or a new experiment.
Record median/mean/sample SD/min/max, individual timings, process CPU utilization,
actual native pools, affinity and whole-process peak RSS. No forward cache here.

Q profile calls unchanged budget_v5.run: technical3 **complete training episodes**,
then exactly the old at-most2-action deployment prefix, seed101, alpha.1,gamma.9,
epsilon.2, original state/actions/reward/target semantics and FrozenAccuracy.
Scientific protocol remains30episodes. Fresh cache per process; separate target
measurements, no cross-target/thread reuse. This profile cannot claim research
completion or final target-reaching deployment metrics.

Timing-only wrappers preserve original args/returns/RNG and measure physical
surgery, magnitude scoring/ranking, reset cloning, legality/state encoding and Q
updates. Actual forward versus cache/hash overhead counted separately. Inclusive
step timing overlaps components and is never summed with them. Residual includes
selection, Python trace bookkeeping and instrumentation. Episode intervals include
reset cloning and end after terminal Q update, just before trace append. Total
instrumented wall time (conservatively including observer overhead) drives forecast.

Full action/reward/state/Q/retained-ID/cache traces, final checkpoint and validation
accuracy/F1 are saved; strict reload/tensor equality/projection checked. Compare
all settings against fresh4 on identical target, and fresh4/50 against archived
native4/50 (410requests/401forwards/9hits/630.4851435758173s). Report exact trace,
mask, tensors, parameter/compression, accuracy/F1 equivalence; report loss drift.
Numerical Q/reward differences are compared only at identical concrete states and
actions, with first divergence, not fabricated counterfactuals. Any selected-action
change disqualifies that configuration from a resource-only remedy.

## Forecast and decision

Unchanged C4 formula: measured Q search seconds ×31/3 ×1.75 +180s. Threshold170min;
absolute LSF limit180min; memory ceiling uses3× measured peak RSS+2GiB <=32768MB.
Each target gets its own estimate. Since the old protocol used50% as proxy for
both, the thread recommendation conservatively uses **max(measured30%,50%)** in
that same formula. This does not assume equal runtimes or that50 is always harder.

Recommend the smallest measured worst-target forecast among exact-equivalent
configurations only if all8 profiles and archived4-thread identity checks pass.
Other54 conditions keep their prior4-thread policy and evidence; no scaling gain
is imputed to them. A successful Q remedy is not automatic full56 release: a
fresh complete manifest/resource gate would still be required. Do not launch it.

If any submitted job is PENDING/RUNNING at handoff, report accepted job IDs and
unmeasured fields as null/NOT_RUN, then stop. Do not wait for completion or infer
scientific findings. All raw logs/checkpoints remain external; compact report and
manifest only in Git; no push or merge.

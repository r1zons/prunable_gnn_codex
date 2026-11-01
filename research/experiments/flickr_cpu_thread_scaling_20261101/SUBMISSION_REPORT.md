# CPU thread-scaling submission and C4 verification

**Benchmark RUNNING, not a completed scaling result.** Latest user-requested host
confirmation: **2026-11-01 22:49:33 MSK**. Stop here while LSF executes the bounded
technical job; no further completion monitoring or full56 submission.

## All measurements run on C4

**Job1612046 is actually executing on `polus-c4-ib.bmc.hpc.cs.msu.ru`.** This is
confirmed both by LSF execution/allocation records and by topology/placement
written inside that job. It is not inferred only from a requested host.

C4 exposes **20 physical cores,160 logical CPUs (SMT8),2 sockets,2 NUMA nodes**.
LSF assigned20 slots/exclusive. All eight profiles share one20-core affinity,
one SMT sibling per physical core:
`0,80,8,88,16,96,24,104,32,112,40,120,48,128,56,136,64,144,72,152`.
They run serially, in fresh Python processes, inside the same C4 job. The runner
also rejects a different host, fewer20 slots or insufficient physical-core
visibility. [Native host confirmation](C4_CONFIRMATION.json).

The user clarified that measurements must be on C4. That condition was already
met by the original submission. **No duplicate benchmark or cancellation was
performed.** C2/C3 jobs contain only CPU topology/allocation metadata commands;
they do not load Flickr, run model forwards or execute Q-learning.

## Accepted jobs and actual allocation

All three `bsub` acceptance responses/IDs verified, with INTENT recorded before
the single submission attempt. Immutable private launchers/responses retained
outside Git; [compact receipts](SUBMISSION_RECEIPTS.json).

| ID | Purpose | Host | LSF status | Requested slots / exclusive reservation | Wall limit / memory |
|---|---|---|---|---|---|
|1612044|Topology metadata only|C2|DONE at22:44 snapshot|1 /20|3min /8192MB|
|1612045|Topology metadata only|C3|DONE at22:44 snapshot|1 /20|3min /8192MB|
|1612046|All eight forward/Q scaling profiles|C4|RUN at22:49 confirmation|20 /20|175min /32768MB|

C2/C3 metadata jobs completing does **not** mean any pruning experiment completed.
Their topology artifacts have not yet been collected/independently verified.
The C4 check collected only existing topology, placement and worker-status
metadata; no scientific outputs or latency measurements were collected.
[Machine-readable job manifest](JOB_MANIFEST.json), [CSV](JOB_MANIFEST.csv).

At22:49 the only started worker directory was `t4-target30`; no child completion
record existed. Remaining4/50,8/30,8/50,12/30,12/50,20/30,20/50 profiles are queued
sequentially within that allocation. This is a timestamped observation, not a
claim about subsequent progress. Native pool counts/latencies/quality still need
per-profile artifact verification.

## Scientific/resource controls

Executable source **10d778223f049935fd5d4ae554494da15377aaba**, branch
`campaign/flickr-structural-pruning-20261101`;876 snapshot files hash-verified on
POLUS. Scientific engine/cache/surgery source matches historical executed
**1b8b8b78667519f97c0682d2645837d4990960b8** byte for byte. Original campaign
artifacts remain immutable; new benchmark namespace only.

Exact frozen GraphSAGE4/256 checkpoint, original masked graph, split and seed101;
no test labels, fine-tuning, CUDA, parallel policies or continuation. Scientific
budget remains30episodes; each resource profile executes3 full training episodes
and the historical two-action deployment prefix. Timed forwards bypass the cache;
Q search retains the unchanged exact per-run cache. Representative pruned widths
[111,230,219] come from the verified historical physical checkpoint. Full design,
source/input pins, instrumentation accounting and fail-closed conditions:
[PLAN.md](PLAN.md), [protocol](BENCHMARK_PROTOCOL.json), [provenance](PROVENANCE.json).

One process at a time; intra-op/BLAS/OpenMP=N, inter-op1, no nested OpenMP teams.
Each process has1200s maximum including setup/benchmarks/output; timeout creates
an incomplete condition rather than shortening a completed scientific run. Eight
child ceilings total160min, with175min LSF walltime for orchestration/headroom,
below the180min absolute limit. Physical affinity placement is controlled across
the new4/8/12/20 comparisons and differs from historical broad SMT visibility;
new4 is the matched timing reference and historical4 is a separate scientific
identity check.

## Measurements and forecasts pending

**0/8 native Q profiles collected and verified.** Dense/pruned forward medians,
variability, CPU utilization/RSS, episode/search/evaluation/surgery/planning/Q time,
forward/cache counts, masks/trajectories/Q/accuracy/F1 comparisons and new full-run
forecasts are **not yet available**. Null fields are explicit in
[BENCHMARK_RESULTS.json](BENCHMARK_RESULTS.json) and [CSV](BENCHMARK_RESULTS.csv).
These files are a submission-stage status table, not fabricated measured results.

The old4-thread50% resource reference remains630.4851435758173s,
410requests/401forwards/9hits; its193.021min conservative forecast still fails.
The original170min acceptance and180min absolute limit are unchanged.

After all native artifacts are available, use each measured target separately
with `seconds ×31/3 ×1.75 +180`; the thread recommendation uses the maximum of
measured30/50 costs and requires exact matched scientific equivalence on both,
plus archived4-thread identity checks. No assumed identical target runtimes.
Other54 conditions retain their old4-thread allocation/evidence; do not infer
scaling for unmeasured architectures/methods.

**Recommended thread count: UNDETERMINED. Full56: NO-GO pending completed native
verification and a fresh complete campaign resource manifest. Full56 jobs0.**
No claim of speedup, acceptance or scientific quality improvement is made.

## Local tests and later checking

Regression158 PASS/0 FAIL/0 SKIPPED; final focused13 PASS/0 FAIL/0 SKIPPED
(overlapping selections). Real tiny GCN/SAGE tests establish timing-wrapper
transparency for actions/rewards/Q/retained masks/tensors/RNG and honest cache
accounting. [Test details](TEST_REPORT.md). No full-suite or native-result PASS
is inferred from these local tests.

Use the approved research login from the private external access file, not an
unverified default SSH account. Later scheduler checks:

```sh
ssh "$POLUS_LOGIN" 'bjobs -a -w 1612044 1612045 1612046'
ssh "$POLUS_LOGIN" 'bhist -l 1612046'
```

New remote artifacts:
`${POLUS_TASK_ROOT}/results/flickr-cpu-thread-scaling-20261101/`.
When DONE, independently check each worker status/completed file inventory,
strict checkpoints/source/input pins, `q_timing.json`, `forward_profile.json`,
`analysis.json` and `benchmark_summary.csv` before publishing any completed
result. Do not rerun or overwrite identities to recreate missing files.
Local raw receipts/metadata/tests:
`${GNN_ARTIFACT_ROOT}/flickr-cpu-thread-scaling-20261101/`.
No environment/routing changes, push or merge.

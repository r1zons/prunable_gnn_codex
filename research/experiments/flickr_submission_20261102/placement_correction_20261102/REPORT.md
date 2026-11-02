# Flickr V5 host-placement correction — verified

Exactly the original56 accepted scientific job IDs1612130–1612185 were inspected. No new campaign or scientific job was submitted.

**Original state:**31 DONE,1 RUN,24 PEND. Explicit C4 binding existed for26 jobs;30 jobs had no mandatory host list. Some unbound jobs had executed on C4, while others ran on C2/C3. Execution host is distinct from a placement restriction.

**21 PEND jobs modified successfully**, using only `bmod -mn JOB_ID`. Mandatory host lists cleared and resulting metadata verified. The installed LSF bmod manual documents the n-suffix as resetting an option; `-mn` is supported. No `-R` replacement, queue change, suspension, cancellation or resubmission was used.

Original IDs/states and complete before/after constraints:[JOB_PLACEMENT.csv](JOB_PLACEMENT.csv). Each original bmod response:[BMOD_RESPONSES.json](BMOD_RESPONSES.json).

| Initial state | Original job IDs |
|---|---|
| DONE (31) |1612130–1612135;1612137–1612141;1612144–1612148;1612151–1612155;1612172–1612176;1612179–1612183|
| RUN (1) |1612136|
| PEND (24) |1612142–1612143;1612149–1612150;1612156–1612171;1612177–1612178;1612184–1612185|

Modified IDs:1612142,1612149,1612150,1612156,1612157,1612158,1612159,1612160,1612161,1612162,1612163,1612165,1612166,1612167,1612168,1612169,1612170,1612177,1612178,1612184,1612185.

**33 jobs already RUN/DONE were not targeted:**32 in the initial snapshot plus1612143, which became RUN before its guarded operation. The two C4-only jobs1612164/1612171 were never targeted. Initial22 otherwise eligible PEND jobs therefore resulted in21 modifications and one state-transition skip.

**Final snapshot:**33 DONE,3 RUN,20 PEND. All18 nonspecial PEND jobs are unbound. The only C4-bound PEND jobs are1612164 and1612171. Historical C4 bindings on1612135/1612136 (DONE) and1612143 (RUN) are preserved according to the leave-RUN/DONE-untouched instruction. No unresolved pending placement restriction.

For every modified job:normal queue,4 requested tasks/slots,4 computational threads,32GiB memory limit,180min walltime,exclusive execution,dependency condition and launch command preserved. The full requested resource expression `rusage[mem=32768] span[hosts=1]` is unchanged. LSF automatically changes its default combined type selection from `type == any` to `type == local` when the explicit host list is reset; no hostname selection/exclusion or affinity clause was introduced. Queue normal permits C1/C2/C3/C4; eligible C2/C3/C4 are not individually excluded. C4 remains eligible for all unbound jobs.

Both protected Q4/256 jobs retain C4,20 requested slots,12 computational/BLAS threads,interop1,exclusive allocation,32GiB/180min/normal and the verified physical CPU mask. Their source execution-resource JSON has the same SHA256 as the original accepted receipts. No runtime resource file or pinned source was changed.

Verification handled asynchronous LSF metadata propagation:1612142 initially still displayed its old host immediately after accepted bmod; a later snapshot confirmed the cleared host list and execution on C2. The modification was not repeated. Two jobs2178/2185 were initially held by a conservative text parser because LSF wrapped `4 Task(s)`; after the actual4-task field was read, each received its first and only bmod. All initial operation records are preserved.

**Future protection:**local submit.py now checks the two-only C4 policy both in manifest validation and immediately before command construction. All other mandatory host values are rejected; no explicit C4 exclusion is generated. [FUTURE_CONFIGURATION.json](FUTURE_CONFIGURATION.json) contains the54-unbound/two-C4 template and requires repinning source before any future use. It was not submitted. Original accepted jobs.json,manifests,receipts and their hashes remain immutable historical evidence.

Raw scheduler snapshots,installed CLI/manual evidence and73 original operation metadata records remain outside Git under `${GNN_ARTIFACT_ROOT}/flickr-placement-correction-20261102/`. Remote operation records:`${POLUS_TASK_ROOT}/results/flickr-structural-v5-submit-20261102/placement-correction-20261102/`. Only scheduler metadata was read; no scientific results collected.

No pytest,PyTorch,model/graph loading,inference,profiling or scientific audit workload. No scientific algorithms/checkpoints/seeds/targets/methods or citation campaign changes. No duplicate jobs,bsub,cancel,requeue,push or merge. Stop after scheduler correction verification.

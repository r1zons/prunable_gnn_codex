# Flickr structural comparison — STOPPED at resource gate

**Research: SUBMITTED0/56; ACCEPTED0; RUNNING0; COMPLETED0; BLOCKED56;
REJECTED0; UNRESOLVED0.** No partial matrix was released. The56-condition
[JOB_MANIFEST.csv](JOB_MANIFEST.csv) distinguishes two direct forecast failures
from54 conditions held by the complete-matrix gate. No scientific result,
method ranking or superiority claim is made.

**Technical:8/8 accepted jobs DONE,56/56 technical outputs independently verified.**
These are two separate28-check preflight versions, not56 completed research
conditions. Exact IDs: uncached **1611889–1611892**; cached revision2
**1611894–1611897**. Job1611893 is not part of this campaign. Original INTENT,
response, launcher and acceptance hashes plus successful scheduler histories
are indexed in [submission.json](submission.json).

## Blocking evidence and decision

The unchanged research Q budget is30episodes. Native GraphSAGE4/256 profiling
with exact validation reuse took **630.485seconds** for three complete training
episodes and a two-action deployment prefix:410 validation requests,401 real
forwards,9 exact-cache hits. The predeclared forecast is
`630.485 × (31/3) ×1.75 +180 =11,581.273seconds =193.021minutes`.
This exceeds both the170minute acceptance threshold and the **180minute absolute
normal-queue limit**. The same harder50% cost proxy was predeclared for both
targets;30% itself was not separately timed to193minutes. All other54 forecasts
are at most95.757minutes. [Complete capacity proof](CAPACITY.json).

Read-only scheduler inspection established that this account can use **short
(30min)** and **normal (180min)**. Longer extended/preempt/fullcluster/gpulong
queues and admin have explicit user ACLs that do not include this account.
No restricted queue was probed with a submission, no access identity or policy
was changed, and no unsafe walltime request was sent to normal. The raw scheduler
outputs remain external, hashed in [PROVENANCE.json](PROVENANCE.json).

**NO-GO for the complete independent-job matrix under the available limits.**
There is no permission request in this report: the task explicitly requires
stopping safely when a mandatory gate cannot pass. Finishing would require a
permitted longer queue (a240minute allocation would accommodate this forecast),
or a separately implemented/tested deterministic episode-boundary continuation
protocol preserving the Q table, RNG, counters and exact validation cache across
jobs. Neither is silently substituted here. Methods, architectures, targets,
30episodes and four threads were not reduced; no scientific accuracy tuning.

## Implementation and verification

Canonical local branch: `campaign/flickr-structural-pruning-20261101`.
Executed source: **1b8b8b78667519f97c0682d2645837d4990960b8**,848 source files.
Source archive SHA256:
`4a4977a3cac50c838087c046d1e09c58033af359a24676caf9fb1a18e70c71d2`.
Resolved protocol file SHA256:
`b451c60bec314cf019d11be00d20417c5461b1abe8718015b81126599d401670`.
The earlier immutable source is `0dad7f29dd833ff9cb8e328341941e5efe9ae7dc`.
The policy module is byte-identical between these revisions; only exact-model
validation execution reuse/accounting and technical profiling changed.
[Execution proof](EXECUTION_REUSE_PROOF.json), [method changes](METHOD_CHANGES.md).

| Verification | PASS | FAIL | SKIPPED |
|---|---:|---:|---:|
| Final focused pytest |146|0|0|
| Full pytest |776|0|6|
| Uncached native artifact/reload/projection audit |28|0|0|
| Cached native artifact/reload/projection/cache accounting audit |28|0|0|
| Paired native execution equivalence contracts |28|0|0|
| Final per-condition resource forecasts |54|2|0|

Full pytest:119.28s. Six explicit skips:2 CUDA device tests,4 opt-in archived-Cora
Lookahead gates. They are not passes. The native equivalence check pairs all
non-Q technical deployment traces/tensors/planning paths and the shared complete
Q training episode0; technical profiles differ1 versus3 episodes, so their final
Q deployments are not falsely compared as identical settings. It executes no new
forwards. Detailed [validation](VALIDATION_REPORT.md), [native audit](NATIVE_PREFLIGHT.json),
[paired proof](NATIVE_EXECUTION_EQUIVALENCE.json), [resource evidence](RESOURCE_PREFLIGHT.md).

Four exact archived Dense checkpoint identities/architectures/metrics and frozen
graph/split pins are in [FROZEN_INPUTS.json](FROZEN_INPUTS.json). All eight target
bands are structurally feasible: [FEASIBILITY.json](FEASIBILITY.json). This is
therefore a runtime-capacity blocker, not target infeasibility or checkpoint
corruption. Physical surgery, scoring/ranking and reward are preserved; new
namespaced budget/Q controllers support depths2/3/4. Historical V2–V4 controllers,
results, Dense checkpoints and the manuscript remain unchanged.

## Evidence and resumption

Raw local root: `${GNN_ARTIFACT_ROOT}/flickr-structural-pruning-20261101/`.
Current remote technical root:
`${POLUS_TASK_ROOT}/results/flickr-structural-pruning-20261101-cached-r2/`.
Earlier root:`${POLUS_TASK_ROOT}/results/flickr-structural-pruning-20261101/`.
Current snapshot:
`${POLUS_TASK_ROOT}/source-snapshots/flickr-pruning-v5-cached-1b8b8b78667519f97c0682d2645837d4990960b8/`.

Current native archive SHA256:
`34b6e4cd9fe688849b3e892d46ff21e727c1385757ca385a9369c6ec357c2dd6`.
Earlier archive SHA256:
`8ea910d5c6114779f67e6e373bbbb1c1d165eb4a4934931e39f689911b831cec`.
Both archives and all completed-marker output hashes were checked after download;
raw checkpoints/logs/traces remain outside Git. Sources/inputs/Dense copies on
POLUS are read-only. Access values remain in private external overrides.

[Commands and later collection](COMMANDS.md) and the prepared analysis module
fail closed without56 acceptances/completions. Do not rerun the already accepted
technical identities or edit a FAIL certificate into PASS. A changed capacity or
execution protocol needs a fresh, pinned gate. There is no research job to monitor
or collect now. No further experiment, push or merge was started after this gate.

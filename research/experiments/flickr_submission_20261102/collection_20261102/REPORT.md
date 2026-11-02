# Flickr V5 collection — 2026-11-02

**COMPLETED 56/56; scheduler DONE 56; RUNNING 0; PENDING 0; failed 0.**
Original LSF IDs **1612130–1612185**, matched individually to original ACCEPTED
receipts. No duplicate jobs, resubmission, cancellation or scientific rerun.

Branch: `submission/flickr-structural-v5-20261102`.
Executed immutable source: `686cf3f1913c2085260ccfef40c384de8580a042`.
Collection archive created **2026-11-02 12:52:42 MSK** (09:52:42 UTC).
The operative `bjobs` retention window no longer contains these IDs. Completion
is instead established independently for each ID by `bhist -l` (`Done
successfully.`), original job stdout (`Successfully completed.`), and the run's
completed marker. Original private scheduler history and logs are preserved.

## Collected evidence and checks

The new server snapshot and local external archive contain **1,951 files,
765,751,620 uncompressed bytes**; compressed archive **80,995,991 bytes**.
SHA256: `27dc5d62bbc02861224b796ea908d176006d1df6a8d4eaf5cb582eaa78c3c7b1`.
All archive, inventory and individual file hashes match. All **504 completed-marker
output hashes** match; all **922 executable snapshot files** match their pinned
SHA256 manifest and Git blobs at the executed commit. Original receipts and
INTENT records match their acceptance-time hashes. Historical evidence is intact.

For all 56 runs the metadata-only verifier checks: exact condition/seed identity;
canonical protocol digest; original graph/split files and Dense checkpoint hashes;
final checkpoint byte hash; finite recorded validation metrics; analytical physical
parameter counts for the actual GCN/SAGE layer widths; retained-ID uniqueness/range;
minimum width; deployment endpoint; compression; complete status; evaluation cache
request/forward counts; and required execution profiles. All eight Q runs contain
**30 complete training episodes**, followed by separate deployment; Q-entry counts
agree with saved tables. All 56 stop at `target_band`, with **zero outside-band**
outcomes. `runs.csv` retains all 56 observations, exact widths/compression, validation
accuracy/macro-F1, search cost, cache counts, Q coverage, unseen choices, masks'
identity digests, host/resources and source/input pins.

**Collection verification: PASS 56 / FAIL 0 / incomplete 0.** This is a file and
recorded-metadata integrity result, not independent scientific tensor validation.
Strict checkpoint reload, original-mask tensor projection and validation forwards
were performed by the original workers and are recorded PASS in their reports;
they were **not independently re-executed during collection**. No pytest, PyTorch,
model/graph deserialization, inference, profiling, training or pruning was executed.
Pytest PASS/FAIL counts therefore are not claimed; pytest was NOT RUN.

## Recorded completion and resources

| Architecture | Target | Completed | Achieved compression across 7 methods | Sum search minutes |
|---|---:|---:|---:|---:|
| graphsage-l3-h128 | 30% | 7/7 | 29.336–30.880% | 10.18 |
| graphsage-l3-h128 | 50% | 7/7 | 49.082–50.952% | 18.77 |
| graphsage-l4-h128 | 30% | 7/7 | 29.026–30.828% | 34.04 |
| graphsage-l4-h128 | 50% | 7/7 | 49.006–50.972% | 63.01 |
| graphsage-l4-h256 | 30% | 7/7 | 29.004–30.922% | 68.95 |
| graphsage-l4-h256 | 50% | 7/7 | 49.008–50.964% | 132.93 |
| gcn-l2-h256 | 30% | 7/7 | 29.295–30.077% | 3.06 |
| gcn-l2-h256 | 50% | 7/7 | 49.216–50.388% | 4.58 |

These ranges summarize recorded outputs, not architecture/method selection or
statistical evidence of superiority. One frozen training checkpoint per architecture
(seed101) provides no independent multi-seed estimate.

Actual execution hosts: **C2 19 / C3 14 / C4 23**. All54 ordinary runs report
4 computational threads; only the two specified Q4/256 runs report12 threads,
interop1, C4 and the exact verified 20-physical-CPU affinity mask. All allocations
remain normal/exclusive/32GiB/180min; LSF can reserve20 node slots even when4
computational slots were requested. Placement-correction history is preserved.

| Special job | Target | Search minutes | LSF RUN minutes | Q training episodes |
|---|---:|---:|---:|---:|
|1612164|30%|35.60|36.03|30|
|1612171|50%|66.90|67.48|30|

Aggregate recorded search **5.592 job-hours**, LSF RUN **5.919 job-hours**;
queue waiting is saved separately per job. These are sums of job times, not elapsed
campaign time. `runs.csv` includes LSF CPU seconds, peak memory and worker peak RSS.
Do not compare architecture walltimes without accounting for host/thread differences.

## Limits and storage

This collection does not select winners or add test evaluation. Test labels were
masked in the prepared scientific inputs; reports state no test-label access.
The campaign saves validation metrics only. There is **no standardized standalone
inference-latency benchmark or test metric** to manufacture from these outputs.
Recorded search forwards and original `validation_forward_seconds` remain in raw
reports, but are not a substitute for comparable dense/pruned inference timing.
Action/reward/candidate traces, full Q tables, environment records, input graph,
splits, Dense/final checkpoints and all logs remain outside Git.

Local raw provenance group:
`${GNN_ARTIFACT_ROOT}/flickr-v5-collection-20261102T094740Z/`.
Remote snapshot is a separate sibling of the original campaign namespace with the
same collection name; the private access metadata records its absolute path.
Original results/source/submission/placement evidence were read, never overwritten.
No citation-campaign files or jobs, algorithms, checkpoint contents or manuscript
were changed. No new experiment or LSF job was launched.

Reproduce the metadata checks from the canonical repository (standard library only):

```sh
python3 research/experiments/flickr_submission_20261102/collection_20261102/collect.py \
  --raw ../prunable_gnn_artifacts/flickr-v5-collection-20261102T094740Z \
  --output /tmp/flickr-v5-collection-recheck
```

Compact files: `runs.csv`, `VERIFICATION.json`, this report, `collect.py` and
`HASHES.json`. Raw archives and private account/access values are excluded from Git.

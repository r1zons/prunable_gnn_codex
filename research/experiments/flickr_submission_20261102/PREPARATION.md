# Flickr structural V5 — submission-only release

Authorization:2November2026, explicitly submit56 scientific conditions using
existing verified evidence, without new tests, model/graph loading, inference,
profiling, preflight, audit jobs or scientific verification. No auxiliary job,
controller, postprocessor, dependency on citation jobs, environment installation,
network change, push or merge is part of this submission.

Branch starts from latest canonical0251954 (including current citation V5
preparation), not an old Flickr checkout. Existing citation source and untracked
citation preparation files remain untouched. Historical source/evidence is not
rewritten. The fixed seven methods all delegate to unchanged structural_budget_v5.

Four original frozen Flickr Dense checkpoints, training/split/policy seed101,
original masked graph/split, targets.30/.50, tolerance.01/minimum4, existing exact
reachability/reward/menu/stopping/ties. Q:30 complete episodes, alpha.1/gamma.9/
epsilon.2, independent epsilon-zero deployment. Lookahead:depth2, executeone/replan.
No Dense fitting, fine-tuning, validation-based tuning or test-label decisions.

The old campaign's CAPACITY.json remains FAIL under the old four-thread Q4/256
allocation. The completed C4 benchmark and its collection remain historical
PASS8/FAIL0/SKIPPED0; their timestamps are not refreshed. The user's present
explicit authorization permits reliance on these existing records and replaces
only the two Q4/256 execution-resource profiles. submission-basis.json records
that authority and exact referenced evidence hashes; it is **not** a new computed
PASS record. The old freshness-based submitter is preserved; this separate
standard-library submitter implements this newly authorized submission basis.

54 jobs retain original four computational threads, four requested LSF slots,
32GiB/180min/normal/exclusive/single-host and their original C4 or unbound host
assignments. Their inter-op default and affinity behavior remain unchanged.
Two GraphSAGE4/256 tabular_q jobs use C4 only,20 requested LSF slots/exclusive,
12 computational/BLAS threads,interop1,32GiB/180min. Exact tested CPU mask:
0,80,8,88,16,96,24,104,32,112,40,120,48,128,56,136,64,144,72,152.
Same benchmark OpenMP environment (no nested teams/no per-thread binding/default
NUMA memory policy); remove GOMP_CPU_AFFINITY, apply mask before Torch import.
These controls are per job, not persistent environment changes.

Minimal infrastructure change: flickr_pruning.execute accepts optional execution
resources instead of unconditionally forcing4; its default remains4. Resource
bootstrap sets the declared environment/affinity before importing Torch; the
existing frozen scientific runner performs the original single scientific run.
Only resource metadata is added to outputs. No algorithm or citation module
changes. **No new tests, PyTorch invocation or computational verification executed**
for this patch, per explicit instruction. Source changes are reviewed as text.
Runtime checks already integral to each scientific run remain intact.

Resource evidence: prior verified assessment retains54 four-thread forecasts.
For Q4/256, measured12-thread projections are85.287min at30% and133.051min at50%;
both below the unchanged170min acceptance threshold/180min LSF limit. Estimates
from3 technical episodes are not completed30-episode science or a hard runtime
guarantee. No citation one-thread setting is reused.

Metadata-only preparation expands exactly56 independent jobs and preserves exact
checkpoint/tensor/input/split/source references. Existing Flickr submission
INTENT/receipt metadata is inspected before any bsub. Duplicate/ambiguous prior
claims block dispatch; no broad scheduler status polling is performed. Each bsub
has a prior immutable INTENT, exact stdout/stderr/exit response and acceptance
receipt with unique ID. Unproven acceptance stops immediately with no retry.
Unattempted conditions stay NOT SUBMITTED. No dependency flag is used.

After the batch, only submission metadata/receipts are retrieved, documented and
committed locally. No execution status monitoring, result collection or analysis.

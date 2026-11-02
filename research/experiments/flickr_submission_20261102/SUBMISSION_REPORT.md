# Flickr structural V5 — accepted submission

**56/56 conditions ACCEPTED by LSF. REJECTED0 / UNRESOLVED0 / NOT SUBMITTED0.**
Acceptance is established from each original successful bsub response, not execution monitoring. RUNNING/COMPLETED states are deliberately not checked.

Submission intent window (Moscow): 2026-11-02T03:35:15.709193+03:00 to 2026-11-02T03:35:16.570011+03:00.

Branch:`submission/flickr-structural-v5-20261102`.
Immutable executable source:`686cf3f1913c2085260ccfef40c384de8580a042`; parent canonical source0251954, with current citation preparation preserved.

Four frozen Flickr Dense seed101 checkpoints ×30/50% ×seven unchanged structural_budget_v5 methods. Q30 full episodes/.1 alpha/.9 gamma/.2 training epsilon/separate epsilon-zero deployment; Lookahead depth2/replan. No Dense retraining or fine-tuning.

| Architecture | Accepted IDs | Requested resources |
|---|---|---|
| flickr-graphsage-l3-h128-s101 | 1612130, 1612131, 1612132, 1612133, 1612134, 1612135, 1612136, 1612137, 1612138, 1612139, 1612140, 1612141, 1612142, 1612143 | 4threads/4slots; original host assignment |
| flickr-graphsage-l4-h128-s101 | 1612144, 1612145, 1612146, 1612147, 1612148, 1612149, 1612150, 1612151, 1612152, 1612153, 1612154, 1612155, 1612156, 1612157 | 4threads/4slots; original host assignment |
| flickr-graphsage-l4-h256-s101 | 1612158, 1612159, 1612160, 1612161, 1612162, 1612163, 1612164, 1612165, 1612166, 1612167, 1612168, 1612169, 1612170, 1612171 | Q:12threads/20slots/C4; other12:4threads/4slots/C4 |
| flickr-gcn-l2-h256-s101 | 1612172, 1612173, 1612174, 1612175, 1612176, 1612177, 1612178, 1612179, 1612180, 1612181, 1612182, 1612183, 1612184, 1612185 | 4threads/4slots; original host assignment |

All56:normal queue,180min,32768MiB RAM,exclusive/single host; no dependency on citation jobs or other science conditions.

**1612164 (30%) and1612171 (50%)**:C4,12 computational/BLAS threads,interop1,20 requested LSF slots; physical mask0,80,8,88,16,96,24,104,32,112,40,120,48,128,56,136,64,144,72,152 applied before Torch import. Forecasts85.2869/133.0511min, below170min acceptance. These are historical measured-workload projections, not full30-episode completion or runtime guarantees.

Other54 retain4 computational threads/4 requested slots and prior host assignments. Overall26 C4-bound/30 unbound. No citation one-thread settings are applied.

**No new tests or computational verification executed**, as explicitly instructed. Minimal execution-resource plumbing reviewed as source text; scientific dependencies match the verified1b8b8b7 source bytes. Existing reports/benchmarks/old FAIL records and their timestamps are preserved verbatim. The submission authority is recorded in submission-basis.json, not a fabricated fresh PASS.

Duplicate prevention read92 INTENT/receipt metadata files across4 existing Flickr namespaces; zero matching scientific claims. No scheduler status queries were needed. Each job has prior INTENT, original response and a unique ACCEPTED receipt. No auxiliary controller, auditor, preflight, profiler or postprocessor was submitted.

Citation source files (19 protected paths) and jobs1612111–1612129 untouched. Current scientific algorithms, historical data/evidence and manuscript untouched. No environment installation, network/credential change, push or merge.

Manifest:[JOB_MANIFEST.csv](JOB_MANIFEST.csv). Accepted receipts and original stdout:[SUBMISSION_RECEIPTS.json](SUBMISSION_RECEIPTS.json). Resolved source/input references:[protocol.json](protocol.json); resource profiles:[execution-resources.json](execution-resources.json); historical evidence mapping:[REUSED_EVIDENCE.json](REUSED_EVIDENCE.json).

Private external submission root:`${GNN_ARTIFACT_ROOT}/flickr-submission-20261102/`; originals of all227 submission metadata files in `submission-receipts/`, plus receipt-bundle-private.json, preparation/transfer metadata and private execution access. No experiment results collected.

POLUS namespace:`${POLUS_TASK_ROOT}/results/flickr-structural-v5-submit-20261102/`. Immutable source under `source/`, copied original frozen inputs under `inputs/`/`dense/`,56 INTENT/response/receipt/launcher directories under `submissions/`. Existing science-run logs/artifacts will be written by LSF jobs; they were not inspected.

Submission stop condition reached. No execution waiting, monitoring, result collection or scientific analysis performed.

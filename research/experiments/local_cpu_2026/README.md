# Local CPU canonical campaign — 2026-10-25

Status: complete and frozen. Research code: `72293149d8fff186dd268789e7b8fea5848c9360`.

- Seeds: 101, 202, 303.
- Conditions: Cora/CiteSeer/PubMed × GCN/GraphSAGE L3/H32.
- Results: 18 dense conditions and 252 method-target rows.
- Full audit: [`audit/AUDIT_REPORT.md`](audit/AUDIT_REPORT.md).
- Raw results/checkpoints: `${GNN_ARTIFACT_ROOT}/local-cpu-20261025/raw/results/`.
- Original backup retained at `/Users/perfect/Documents/ChatGPT/Диплом/local_pilot_20261025/`.

The CSVs under `audit/` are compact tracked copies. Artifact paths use `${GNN_ARTIFACT_ROOT}` after consolidation; original absolute provenance remains in the external frozen copy. `artifact_manifest.csv` records the hashes of that frozen raw tree, not the sanitized Git copies. `configs/` and `status/` are sanitized snapshots. Scripts are preserved for provenance and audit, not as permission to rerun the completed campaign.

Do not combine these rows with POLUS or historical debug rows. For paper summaries, use `aggregate_by_condition_method_target.csv`; the cross-condition aggregate is descriptive only.

# Consolidation record — 2026-10-25

## Initial state

- Canonical repository: clean `main`, `72293149d8fff186dd268789e7b8fea5848c9360`, remote `https://github.com/r1zons/prunable_gnn_codex.git`.
- Consolidation branch created in that checkout: `consolidation/gnn-research-20261025`.
- Additional folder `/Users/perfect/Documents/ChatGPT/Диплом` was not a Git repository. It contained:
  - `local_pilot_20261025`: 2,679 files, about 161 MB;
  - `polus-research`: 125 files, about 4.8 MB;
  - `polus-results`: four files, about 27 MB;
  - `polus-wheel`: four files, about 53 MB.
- The separate HPC course was identified as wave-equation work and excluded.

## Material consolidated into Git

- `research/STATUS.md` and `research/experiments/INDEX.md` now define the entry point and campaign registry.
- Local CPU: audit report, nine compact CSV/JSON audit outputs, 18 resolved config snapshots, 19 completion/status records and four audit/execution helper scripts.
- POLUS: four operational documents, six prepared configs plus provenance records, campaign status, 84-row validation summary, environment/package evidence, VSX patches and 23 setup/LSF/helper scripts.
- Account, server-home and Mac route values were replaced in tracked copies by documented variables. Original exact provenance remains in frozen external evidence.
- `AGENTS.md` now pins the canonical checkout, reading order, storage boundary and no-rerun/no-clone rules.

The scientific pipeline under `src/`, the reviewed Kaggle protocol and all model/pruning parameters were left unchanged.

## Material consolidated outside Git

Adjacent artifact root: `/Users/perfect/Desktop/code/MSU/MSU_SQI_25_DIPLOMA/test_codex/prunable_gnn_artifacts` (3,774 files, about 301 MB after consolidation).

- Full local campaign copied byte-for-byte to `local-cpu-20261025/raw/`.
- POLUS result and evidence archives copied to `polus-pilot-20261024/archives/`; result archive also extracted under `raw/` for direct access.
- POWER8 wheel, setup evidence, full server-evidence tree and prior `polus-research.zip` copied under `polus-environment-20261023/`.
- External `README.md` and `MANIFEST.sha256` document the layout and principal hashes.

The source folder `/Users/perfect/Documents/ChatGPT/Диплом` was not deleted or modified and remains a second copy.

## Verification and conflicts

- Sorted per-file SHA256 comparison found no difference between the source and copied local campaign.
- Both POLUS tarballs pass gzip integrity checks and match their recorded SHA256 values.
- Local and POLUS seed-101 split hashes match per dataset; dense checkpoints/results do not. The two platforms remain separate provenance groups.
- Adaptive Layerwise is not a matched 0.30/0.50 comparator on either platform; this limitation is retained, not repaired or hidden.
- The Kaggle plan remains a protocol-only record because no Kaggle campaign was executed. Local/POLUS execution does not rewrite that historical plan.
- The older top-level `RESEARCH_STATE.md` is retained as historical context and now points to the current status.

## Deliberately not added to Git

- Raw checkpoints, logs, the 4.1 MB PyTorch build log, wheels and archives.
- Private keys, passwords, VPN profiles, actual SSH account, server-home path and Mac route addresses.
- Historical container-image registry dumps `1.6.2-pytorch-ubuntu18.04-py37-ppc64le.json` and `1.7.0-pytorch-ubuntu18.04-py37-ppc64le.json`; they remain in the source folder and `polus-research.zip`, but are not needed to reproduce the successful CPU environment.
- Any content from the wave-equation HPC course.

## Not performed

- No training, pruning, smoke experiment, LSF submission, server upload, package update or environment reinstall.
- No SSH connection was needed: the local archives and evidence were complete enough for integrity and provenance verification. Live server presence of the remote directory was therefore not rechecked during consolidation.
- No push or merge to `main`.

## Remaining external checks

- Confirm the official 2026 submission route, template and deadline; 2026-11-02 is currently a researcher-designated target.
- Before any future POLUS work, perform read-only access/status checks using the existing local SSH/network configuration and verify available quota; do not resubmit the completed campaign.
- Before publication, review figures/tables against the condition-level three-seed local aggregate and the claim/evidence registry.

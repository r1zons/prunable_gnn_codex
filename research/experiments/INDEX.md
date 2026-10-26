# Experiment campaign registry

`GNN_ARTIFACT_ROOT` defaults locally to the adjacent `../prunable_gnn_artifacts`. Raw artifacts are excluded from Git. Compact reports, summaries, configs and evidence live under `research/experiments/`.

| ID | Status | Platform / seeds | Scope | Canonical compact evidence | Raw evidence |
|---|---|---|---|---|---|
| `LOCAL-CPU-20261025` | complete, frozen | Mac CPU / 101, 202, 303 | 18 dense; 252 method-target | [`local_cpu_2026`](local_cpu_2026/README.md) | `${GNN_ARTIFACT_ROOT}/local-cpu-20261025/raw/` |
| `POLUS-CPU-101-C4-V2` | complete, frozen | polus-c4 CPU / 101 | 6 dense; 84 method-target; jobs 1598695–1598700 | [`polus_cpu_2026`](polus_cpu_2026/README.md) | `${GNN_ARTIFACT_ROOT}/polus-pilot-20261024/` |
| `KAGGLE-PILOT-2026` | protocol only; not executed on Kaggle | planned / 101 then 202,303 | reviewed protocol source | [`kaggle_pilot_2026/PLAN.md`](kaggle_pilot_2026/PLAN.md) | none |
| `HISTORICAL-COMPACT` | historical/debug only | local / 42,43; partial 44 | older 30-epoch outputs; do not pool | [`../audits/2026-10-22_local-codex.md`](../audits/2026-10-22_local-codex.md) | existing ignored `runs/` tree |
| `HPC-WAVE-COURSE` | excluded | POLUS / unrelated | 3-D wave equation, not GNN pruning | exclusion recorded in local audit | `/Users/perfect/Documents/ChatGPT/Математические дисциплины/cmc_msu_hpc_course_2026/` |

## Artifact integrity

| Artifact | Bytes | SHA256 |
|---|---:|---|
| `polus-pilot-20261024/archives/pilot-cpu-101-c4-v2-results.tar.gz` | 28,354,373 | `2b97d6923efa6cd8836830fdb7e13872c336da1700b514ab538118d47763fa62` |
| `polus-pilot-20261024/archives/pilot-cpu-101-c4-v2-evidence.tar.gz` | 32,196 | `66a905939510edb50e1a64ae814069c91a052fb362e61c6d7fe3cd88ea83aa3f` |
| `local-cpu-20261025/raw/audit/artifact_manifest.csv` | manifest | `f80545f83282e7fe1909f794f525a9baa43dbeb116f21a7ea9d5c9a00990376a` |
| `polus-environment-20261023/wheel/torch-2.0.1+polus.power8.1-cp310-cp310-linux_ppc64le.whl` | 55,591,834 | `30660e31f8c88159461ea647b2f175bdd23e501cad19b3a464fb451ddec8b830` |
| `polus-environment-20261023/wheel/setup-evidence.tar.gz` | 239,351 | `d334511a0950678081431a1d7482ed9ef34a948b9c8413dd8a3e9cba68687251` |
| `polus-environment-20261023/polus-research.zip` | 408,964 | `b5d59fb06f5124f8f74123fd687c40206eb560ef22c29df91d19f50b951439c8` |

The full local raw copy was compared file-for-file with `/Users/perfect/Documents/ChatGPT/Диплом/local_pilot_20261025`; no difference was found. A deterministic digest of the sorted per-file SHA256 listing is `66d43135c2eeaf5ac1bff216c1d775bc2093521d57c99c179f6b167bc2f2c27b`.

## Interpretation boundary

The two completed campaigns used the same research commit and seed-101 split hashes, but separate CPU stacks produced distinct dense checkpoints and non-identical post-prune outcomes. POLUS rows are therefore replication/platform evidence, not an additional independent seed. See [`polus_cpu_2026/PLATFORM_COMPARISON.md`](polus_cpu_2026/PLATFORM_COMPARISON.md).

Never overwrite a registered campaign. A corrected protocol or implementation creates a new ID, directory, artifact root and provenance record.

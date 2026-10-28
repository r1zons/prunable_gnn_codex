# POLUS CPU pilot — 2026-10-24

Status: complete and frozen. Campaign: `pilot-cpu-101-c4-v2`. Research code: `72293149d8fff186dd268789e7b8fea5848c9360`.

- Six jobs 1598695–1598700, all `DONE` on polus-c4.
- Six dense conditions and 84 method-target rows, seed 101 only.
- Maximum full LSF wall time: 4:57; CPU-only, four threads per job.
- Compact result table: [`evidence/pilot_validation_summary.csv`](evidence/pilot_validation_summary.csv).
- Campaign manifest: [`evidence/campaign-status.json`](evidence/campaign-status.json).
- Full execution report: [`docs/RUN-20261024.md`](docs/RUN-20261024.md).
- Environment and access notes: [`docs/SERVER-STATUS.md`](docs/SERVER-STATUS.md).
- Preserved launch/build helpers: [`scripts/`](scripts/). Existing completed jobs must not be resubmitted.

Raw results and two immutable archives are under `${GNN_ARTIFACT_ROOT}/polus-pilot-20261024/`. The POWER8 CPU wheel, environment evidence and setup archive are under `${GNN_ARTIFACT_ROOT}/polus-environment-20261023/`.

Tracked copies replace the account name, home path and Mac route values with `POLUS_SSH_TARGET`, `POLUS_TASK_ROOT`, `POLUS_IPV4` and `LOCAL_GATEWAY_IPV4`. Real access material remains in the user's existing SSH/network configuration or ignored local override. No secret, private key, VPN profile or password is tracked.

This campaign is platform replication/operational evidence. Seed 101 is the same seed used locally and is not a fourth independent seed.

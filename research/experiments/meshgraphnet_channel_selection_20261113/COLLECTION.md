# CylinderFlow · collection from POLUS

**LSF 1618600 — DONE, collected 2026-11-14.** The saved LSF report records a successful run on one Tesla P100 on C3 in queue `normal` with 820 s reported runtime and 1,070 MB peak host memory. The job requested four CPU slots, 8 GiB RAM, and 180 minutes. The stderr file is empty.

The executed source is `096b7fe00d7e275b76f3daf017931397a3313ff3`. The [protocol](PROTOCOL.md) pins the frozen Adam update 47,000 checkpoint, the validation input hash and the disjoint subsets. The saved calibration records exact replay of a five-step selection candidate and a 2,347 s runtime forecast.

## Verified at collection

- SHA-256 of all 323 downloaded files matched the remote manifest. The source manifest matched the local staged payload, and the submission pins verified.
- All 45 decisions and 12 scientific physical checkpoints are present. The archive also contains one calibration checkpoint.
- All 224 expected diagnostic windows are present. Each has 100 finite, nonnegative MSE values. No windows are missing or failed.

This verifies file integrity and completeness. The tensors and error curves were recorded by the worker. Scientific comparisons and independent model evaluation are pending.

## Archive

Local directory `${GNN_ARTIFACT_ROOT}/meshgraphnet-channel-selection-20261113/collected/`. Original POLUS directory `$HOME/gnn-p100-20261110/meshgraphnet-channel-selection-20261113/`.

- Archive `run.tar.gz`, SHA-256 `0e526ca44cd764b89d46e9b67987e84731432952835ffebed4a314f0d17722e9`.
- Transfer evidence `remote-files.sha256`, `transfer-check.log`, and `completeness-check.log`.
- Scheduler evidence `submission/INTENT`, `submission/ACCEPTED`, `submission/bsub.out`, and `lsf.1618600.out`.

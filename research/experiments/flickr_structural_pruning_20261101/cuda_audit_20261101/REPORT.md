# Flickr CUDA audit — 2026-11-01

**NO-GO for native CUDA execution and the full campaign in the discovered environment.**
Audit source `521f20aacabb0a531858247f8a927f1db11323da`, branch
`campaign/flickr-structural-pruning-20261101`; initially clean canonical checkout.
No scientific implementation, historical output, CPU environment or manuscript changed.
No new LSF job, installation, driver update, route change, push or merge.

The user explicitly requires stopping if no compatible CUDA environment exists.
That gate failed: two discovered PyTorch installations are CPU builds; the other
runnable prefixes contain no PyTorch. Therefore phases 3–4 were **NOT RUN**, rather
than represented by synthetic comparisons or a guessed GPU speedup.

## Repository compatibility

The complete component inventory and exact source anchors are in
[CODE_AUDIT.csv](CODE_AUDIT.csv). “Portable source” below means the inspected
operations support device tensors in principle, **not a verified P100 pass**.

| Area | Evidence and classification |
|---|---|
| Device configuration | Generic config supports `auto`/CUDA; default is CPU. The dedicated Flickr V5 runner does not consume this device option. |
| GCN / GraphSAGE | Standard PyG convolutions, activations and dropout; no forced CPU in forward. Matching model/graph/index placement is required. Native CUDA reductions and backward operations remain untested. |
| Graph / evaluation | Generic training and V2 checkpoint workflow transfer data/model/indices, and convert metrics through CPU. Frozen Flickr loader instead maps graph/checkpoint to CPU and builds a CPU model. Dataset loading alone does not establish graph residency. |
| Physical surgery | Deepcopy and index selection preserve tensor dtype/device; index tensors use the parameter device; GCN cached graph state is cleared. No replacement by mask-only pruning. Existing GPU tests cover tiny three-layer models, not the current Flickr depth-4 run. |
| Static Random / Magnitude / SNIP | Device-aware parameter masks/scores and physical surgery. SNIP transfers its cloned graph/train indices to the model device and uses training labels. Ranking and V5 fixed-budget planning deliberately use CPU doubles/NumPy; these transfers still cost time on GPU. |
| Sequential Random / Greedy / Lookahead | V5 environment/planner clones models and calls an injected evaluator; host policy/reachability/reward logic can drive CUDA models. No native seven-method V5 GPU qualification exists. |
| Q-learning | Q table, legal action generation and updates remain host code; evaluator and surgery determine device execution. Its existing scientific state/reward/episode/action/budget rules were not changed. |
| Checkpoints / identity | Generic loading accepts `map_location`; canonical tensor hash explicitly converts detached contiguous tensors to CPU bytes. This is suitable for device-independent identity, but each cache key can entail GPU-to-CPU transfer. Serialized file hash and tensor hash are distinct checks. |
| Timing / memory | Generic benchmark helper warms up and synchronizes CUDA around forward timings, and records peak allocated memory. Current Flickr runner uses CPU timing/RSS instrumentation; no explicit GPU boundaries, reserved-memory record or utilization sampling. |

**Confirmed portability gaps, not failures of the existing CPU contract:**

1. `experiments/flickr_pruning.py:frozen/execute` and final reconstruction are CPU
   specializations with no GPU transfer. The runner is not CUDA-enabled merely
   because model classes accept `.to(device)`.
2. Flickr `per_class` and `flickr_dense.metrics` call `.numpy()` directly on label
   and prediction tensors. Moving these tensors to CUDA would make reporting
   fail. A future adapter must explicitly transfer reporting arrays to CPU.
   This follows the pinned [PyTorch tensor-to-NumPy implementation](https://raw.githubusercontent.com/pytorch/pytorch/v2.0.1/torch/csrc/utils/tensor_numpy.cpp).
3. `FrozenAccuracy` keys contain input identity, architecture and tensor bytes,
   but not device/backend/precision. Its existing single frozen CPU callback is
   valid. Reusing one cache across CPU and GPU could return a CPU scalar without
   a GPU forward, concealing numerical differences. Native comparison must use
   **fresh separate caches per device and fixed callback**, and establish
   repeatability before relying on memoization. Do not change canonical tensor
   hashes to solve execution-context identity.

CUDA seeding exists; deterministic CUDA kernels are not enforced. `fork_rng`
with `devices=[]` preserves CPU RNG only. This is an untested CUDA/default-device
boundary, not evidence of an observed CUDA RNG bug. Tiny floating-point reduction
differences can alter magnitude ordering, validation buckets, tied decisions and
Q updates; exact masks/actions/Q equality must be measured, never inferred.
Historical generic workflows/Colab use do not qualify current budget-V5 or justify
substituting old controllers with different stopping rules.

## Verified POLUS inventory

Read-only approved research-account SSH, scheduler metadata, environment imports,
`nvidia-smi`, `nvcc --version` and NVML queries; no direct compute-node SSH or
GPU kernels on the frontend. Timestamped raw snapshot starts 2026-11-01 20:27 MSK.
Private raw output remains external; [ENVIRONMENT.json](ENVIRONMENT.json) is sanitized.

| Host | CPU / slots / host memory | LSF GPU inventory |
|---|---|---|
| `polus-c2-ib.bmc.hpc.cs.msu.ru` | POWER8 / 20 / 256G | 2 P100, 16,280 MiB each, driver 440.33.01 |
| `polus-c3-ib.bmc.hpc.cs.msu.ru` | POWER8 / 20 / 256G | 2 P100, 16,280 MiB each, driver 440.33.01 |
| `polus-c4-ib.bmc.hpc.cs.msu.ru` | POWER8 / 20 / 1T | 2 P100, 16,280 MiB each, driver 440.33.01 |

C1 was unavailable. The frontend separately exposes two
**Tesla P100-SXM2-16GB**, driver **440.33.01**, compute capability **6.0** verified
by NVML, consistent with [NVIDIA's P100 specification table](https://developer.nvidia.com/cuda/gpus/legacy).
`nvidia-smi` reports CUDA maximum **10.2**, 16,280 MiB total and 10 MiB used per GPU,
0% utilization, no processes in that snapshot. This is hardware/idle metadata,
not a memory or performance benchmark. System `nvcc` is **10.2.89**.
Compute-node GPU inventory is scheduler-reported; visibility inside a reserved
one-GPU job was **not tested**.

`normal` is Open:Active, USERS all, 180-minute limit, exclusive allocation allowed,
NO_INTERACTIVE, hosts C1–C4. Dedicated `gpulong` is Open:Active with 720 minutes
and preemption, but its ACL excludes the approved research account. No unfinished
user jobs at the initial query. No jobs were cancelled.

LSF reports `LSB_GPU_NEW_SYNTAX=Y`, MB limit units and CPU/memory enforcement;
`bsub -h -gpu` documents `num` and `j_exclusive` requests. The extended `bhosts
-gpu` query is unavailable under this setting; this does **not** prove `bsub
-gpu` is unsupported. One-GPU reservation acceptance and job-level device
isolation remain unverified. Normal is the candidate route, not a claimed GPU
allocation. No ACL bypass or gpulong submission attempted.

## Environment gate

Visible research/shared inventory: **15 prefixes, 12 runnable Python interpreters**.
Search boundary is explicit in ENVIRONMENT.json; unknown private environments
and inaccessible images are not ruled out.

| Environment | Verified imports | CUDA outcome |
|---|---|---|
| Existing research CPU | Python 3.10.22, PyTorch 2.0.1, PyG 2.4.0, ppc64le | `USE_CUDA=0`, `torch.version.cuda=null`, availability false, arch list empty |
| Shared `ska23-ml` | Python 3.10.13, PyTorch 1.13.1 CPU, no PyG | `USE_CUDA=0`, availability false; also below project Torch>=2.0 requirement |
| Shared `test-cuda` | Python 3.10.12, cudatoolkit 11.8 runtime | No Torch or PyG; runtime presence does not provide a working stack |
| Shared `test-cuda-9.1` | Python 3.10.12, actually cudatoolkit 10.2.89 | No Torch or PyG; directory name is not version evidence |
| Remaining prefixes | Ten runnable interpreters total without Torch; three non-runnable prefixes | No candidate CUDA PyTorch found; some Jupyter prefixes also have CUDA 12 runtime packages only |

Docker/Singularity executables exist, but Docker daemon is inaccessible; checked
shared image directories were absent. No image downloaded or executed.

Pinned PyTorch 2.0.1 [requires CUDA >=11.0 at build time](https://raw.githubusercontent.com/pytorch/pytorch/v2.0.1/cmake/public/cuda.cmake).
The current ordinary R440/CUDA10.2 stack does not satisfy this; NVIDIA's
[CUDA 11.0 release requirements](https://docs.nvidia.com/cuda/archive/11.0/cuda-toolkit-release-notes/index.html)
move to the R450 driver family. Merely installing a runtime package or setting
`CUDA_VISIBLE_DEVICES` cannot enable an existing CPU build. POWER8/ppc64le,
host OS and P100 sm_60 compatibility need qualification together; an x86 CUDA
wheel recipe is not a reproducible solution here. No major installation attempted.

PyG imports alone do not test CUDA operations. GCN/SAGE edge-index paths require
native scatter/index/aggregation kernels, and SNIP additionally needs backward.
Optional compiled packages depend on the selected PyG path; do not demand or
claim all extensions merely from their names. See [pinned PyG 2.4 installation guidance](https://pytorch-geometric.readthedocs.io/en/2.4.0/install/installation.html).

## Actual test and benchmark outcomes

| Verification in this audit | PASS | FAIL | SKIPPED / NOT RUN |
|---|---:|---:|---|
| Existing CUDA-only tests, local CPU environment | 0 | 0 | 2 SKIPPED, 71 deselected; 4.96s |
| Focused config, surgery, benchmark, cache pytest | 112 | 0 | 2 SKIPPED; 15.53s |
| Native one-GPU LSF tests, checkpoint reload, seven methods / short Q trajectory | — | — | NOT RUN: environment gate failed |
| CPU/GPU validation accuracy/F1, logits, counts/compression, retained IDs/actions/Q | — | — | NOT RUN; no equivalence claim |
| P100 dense/pruned latency, Q runtime, cache calls, allocated/reserved VRAM, utilization, synchronization overhead | — | — | NOT RUN; all measurements null |

The same two CUDA cases occur in both local selections; only **two unique skips**.
Tests use local Python3.12.9/Torch2.2.2/PyG2.6.1, not the POLUS runtime. GPU cases
are GCN/GraphSAGE tiny-graph float64 surgery/logit checks. Configuration assertions
mentioning CUDA do not execute CUDA. The full suite was not rerun for this
source-read-only audit; earlier CPU campaign suite results are not new GPU evidence.
Exact commands and counts: [TEST_RESULTS.json](TEST_RESULTS.json).

## Runtime conclusion and next gate

**No defensible numerical P100 forecast or 180-minute GO is available.**
The previous CPU preflight, not repeated here, measured Q 4/256 three-episode
technical cost 630.485s with 410 requests, 401 real forwards and 9 cache hits;
its safety forecast is 193.021min. This explains the CPU blocker, but cannot
supply an unmeasured GPU speedup or guarantee a 16GB memory fit. Host surgery,
Python planning, hashing and transfers may remain significant.
[Machine-readable benchmark status](BENCHMARK_STATUS.json) records nulls, not zeros.

[SETUP_PROPOSAL.md](SETUP_PROPOSAL.md) specifies the minimal qualification and
isolated environment proposal, followed by the single bounded technical-job gate.
It is **not executed or a ready-tested installation recipe**. After that gate,
only measured synchronized forward/search/memory costs and explicit numerical
comparisons can support a conservative unchanged-30-episode forecast.

Raw evidence: `${GNN_ARTIFACT_ROOT}/flickr-cuda-audit-20261101/`; hashed inventory
and audited source hashes in [PROVENANCE.json](PROVENANCE.json). Original campaign
[HASHES.json](../HASHES.json), inputs and algorithms are preserved. Audit files
are a new provenance group; no full56 launch, training, test-label access or
scientific accuracy selection occurred.

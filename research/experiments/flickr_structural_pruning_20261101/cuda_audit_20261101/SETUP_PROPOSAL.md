# Proposed minimal reproducible CUDA qualification — NOT EXECUTED

The present audit stops at its failed environment gate. No permission for a driver
upgrade, PyTorch rebuild, package installation or CPU-environment replacement is
inferred. This proposal requires a separately authorized infrastructure task.

1. **Locate an existing compatible prefix first.** The audit found no CUDA Torch
   in the visible research/shared inventory. An administrator-supplied existing
   prefix or pinned image may avoid installation; verify its hashes and imports,
   not its directory name. No message to administrators was sent.
2. **Qualify the complete platform.** Record native compute-host OS/glibc,
   POWER8 ISA/ppc64le, compiler, P100 sm_60, driver and runtime. PyTorch2.0.1 needs
   CUDA>=11.0; ordinary R440/CUDA10.2 is insufficient. Choose an administrator-
   supported combination, not a blind driver upgrade. NVIDIA's [CUDA11.7 platform
   table](https://docs.nvidia.com/cuda/archive/11.7.0/cuda-installation-guide-linux/index.html)
   lists POWER9/RHEL8 for its native POWER qualification; this does not establish
   support on this POWER8 host. It also does not prove every possible CUDA11
   POWER8 combination impossible. If the existing platform cannot be qualified,
   obtain an approved compatible GPU execution host before proceeding. Containers
   do not replace host driver/ISA requirements. Avoid an unqualified latest CUDA
   toolkit: [NVIDIA's architecture matrix](https://docs.nvidia.com/datacenter/tesla/drivers/cuda-toolkit-driver-and-architecture-matrix.html)
   limits Pascal's toolkit line to CUDA12.x.
3. **Use an independent user prefix.** Preserve the working CPU prefix and
   activation script. Proposed scientific pins: Python3.10.22, PyTorch2.0.1 CUDA
   build, PyG2.4.0, NumPy1.24.4, SciPy1.10.1, scikit-learn1.2.2; derive remaining
   dependencies from the archived CPU lock/receipt. Obtain compatible
   ppc64le/POWER8 binaries with source/build receipts, SHA256 and an explicit
   `sm_60` kernel image. If unavailable, a separately authorized build would need
   `USE_CUDA=1`, `TORCH_CUDA_ARCH_LIST=6.0`, a qualified CUDA/host compiler pair
   and preservation of the existing POWER8 build-fix provenance. These flags
   are requirements, not a tested complete build recipe. Do not silently
   downgrade Torch or execute an x86 `pip cu117` recipe on POWER8.
4. **Prepare only a device-aware technical adapter**, leaving scientific engines
   unchanged. Use the four pinned frozen checkpoints and original validation
   split from `../FROZEN_INPUTS.json`; verify original file/tensor/graph/split
   hashes before transfer. Keep test labels masked. Transfer independent CPU/GPU
   model/graph/index clones once; CPU-convert metric arrays. Strictly reconstruct
   nonuniform checkpoints then move to the requested device; check retained-ID
   projection and device-independent tensor hashes. Do not replace budget-V5
   with legacy controllers or change rewards/targets/horizons/actions/episodes.
5. **One bounded one-GPU LSF technical job only after the environment works.**
   Candidate queue `normal`, four CPU threads, single host/exclusive convention,
   one GPU requested with documented `-gpu "num=1:j_exclusive=yes"`, and a
   predeclared short technical time/memory budget under 180min. This request
   syntax is documented locally, but scheduler acceptance/isolation remain to be
   tested; no executable submission file or acceptance is claimed. Save
   INTENT/ACCEPTED receipt, environment and actual host, and require exactly one
   allocated/visible GPU. Fail if reservation cannot be verified. Do not use the
   inaccessible gpulong queue, MPS or multi-GPU parallel Q-learning.
6. **Correctness before performance.** Run the existing two native CUDA tests
   first; then GCN/SAGE forward, graph transfer, physical surgery, strict reload,
   representative steps of all seven V5 methods and a bounded deterministic Q
   prefix. Use fresh separate per-device caches and test repeatability. Declare
   logit tolerances in advance (starting float32 `atol=1e-5, rtol=1e-4`, report
   actual max/mean error rather than loosening based on results). Report exact
   counts/compression/masks and all selected-action/Q differences separately
   from metric/logit tolerances. Accuracy/F1 equality is an observation to check,
   not a premise. A deliberately stopped prefix is not a completed 30-episode
   scientific run and must retain its stopping/accounting flags.
7. **Synchronized P100 measurements.** GraphSAGE4/256 dense and representative
   pruned model, same inputs/seed/scientific settings. Keep graph resident;
   warm up; synchronize before and after each wall-time boundary; record actual
   forwards/cache requests/hits, host scoring/hash/copy time, allocated and
   reserved peak memory, utilization, device identity, threads and clocks.
   Compare uncached CPU/GPU correctness before using caches. Bound the technical
   prefix without claiming changed episode settings as the full method.
   Extrapolate search cost conservatively using measured state/action evaluation
   counts and a safety factor, with host overhead and full30 deployment included;
   report queue waiting separately. No full30 run until a measured <=180min
   resource gate and separate campaign authorization are satisfied.

Until these gates pass: native CUDA correctness **UNVERIFIED**, performance and
30-episode fit **UNKNOWN**, full campaign **NO-GO**.

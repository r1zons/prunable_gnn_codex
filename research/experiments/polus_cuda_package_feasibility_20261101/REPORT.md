# POLUS prebuilt CUDA PyTorch and driver feasibility

2026-11-01. Source `db3fa65b683dba20c5487293b6654bb4cf8ede97`; investigation branch
`audit/polus-cuda-package-feasibility-20261101`. No installation, candidate import,
CUDA kernel, driver change, sudo, LSF submission, campaign change, push or merge.

**NO-GO for a production driver upgrade or campaign launch on the evidence available.**
There is a **promising prebuilt PyTorch 2.0.1/CUDA11.8 payload**, with actual
POWER8 compiler flags and `sm_60` in the inspected binary. A separate Python3.10
technical dependency plan resolves. This is stronger evidence than a package
listing, but **not verified execution of the whole stack on POWER8/P100**.
The stock solver-selected RocketCE Python is a confirmed POWER9-targeted build;
current R440 and the unqualified POWER8/RHEL7.5 driver path are separate blockers.

## 1. Verified package candidates

Read-only snapshots: IBM's supplied archive (1,136 records), OSU OpenCE1.0.0,
1.2.2,1.4.1, explicitly named1.5.1-power8 and1.9.1, RocketCE package API/main and
1.9.1-label repodata, official PyTorch ppc64le channel, and PyG/PyPI metadata.
The queried official PyTorch ppc64le channel has no packages. This is a bounded
repository search, not a claim that no privately built package exists anywhere.
Exact filenames, direct URLs, hashes, sizes and original dependency constraints:
[PACKAGE_CANDIDATES.json](PACKAGE_CANDIDATES.json) / [CSV](PACKAGE_CANDIDATES.csv).

| Repository / actual payload | Python / CUDA | Assessment for this project |
|---|---|---|
| IBM `pytorch-base-1.3.1-gpu_py37_21941.g1a3a219.tar.bz2` | 3.7 /10.2, cuDNN7.6.5, NCCL2.5.6 | GPU dependency metadata confirmed; whole payload/POWER8/sm_60 runtime not verified. Project/API incompatible. |
| OSU OpenCE1.0.0 `pytorch-base-1.6.0-cuda10.2_py38_4.tar.bz2` | 3.8 /10.2 | GPU metadata confirmed; project incompatible; CPU ISA/runtime not established. |
| OSU OpenCE1.2.2 `pytorch-base-1.7.1-h1234567_cuda10.2_py38_pb3.11.2_18.tar.bz2` | 3.8 /10.2, cuDNN7.6.5, NCCL2.7 | Later CUDA10.2 payload found, but still below project/PyG requirements; not a valid campaign replacement. |
| OSU OpenCE1.5.1-power8 `pytorch-base-1.10.1-h1234567_cuda11.2_py39_pb3.14_1.conda` | 3.9 /11.2 | A POWER8-specific channel exists, but its Torch/Python are too old and CUDA needs a newer driver. Channel name alone is not runtime verification. |
| RocketCE1.9.1 `pytorch-base-2.0.1-cuda11.8_py310_pb4.21.12_1.conda` | 3.10 /11.8 | Real CUDA payload independently downloaded/hash-verified and statically inspected; promising, unqualified whole stack. |
| RocketCE1.9.1 `pytorch-2.0.1-cuda11.8_py310_1.conda` | 3.10 /11.8 | Only a 4,541-byte metapackage, not the Torch binary. Adds torchtext/vision/data dependencies not needed for this audit. |
| RocketCE `python-3.10.11-h8b3979d_4_cpython.tar.bz2` | 3.10 | Independently hash-verified; recipe and installed sysconfig contain `-mcpu=power9`. Unsuitable as a supported POWER8 interpreter. |

The IBM payload URL is under the [supplied IBM archive](https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda/linux-ppc64le/);
OSU payloads are from [OpenCE1.2.2](https://ftp.osuosl.org/pub/open-ce/1.2.2/linux-ppc64le/)
and the separate [POWER8 channel](https://ftp.osuosl.org/pub/open-ce/1.5.1-power8/linux-ppc64le/).
None was inferred from an x86 wheel. Legacy package compatibility is not inferred
from successful Conda indexing or from the installed system's old PowerAI RPMs.

## 2. CUDA10.2 assessment

**No validated CUDA10.2 solution for the unchanged current campaign was found.**
The project's metadata requires Python>=3.10 and Torch>=2.0. Pinned upstream
[PyTorch2.0.1 CMake requires CUDA>=11.0](https://raw.githubusercontent.com/pytorch/pytorch/v2.0.1/cmake/public/cuda.cmake).
PyG2.4's basic path [requires Torch>=1.11](https://pytorch-geometric.readthedocs.io/en/2.4.0/install/installation.html).
The recovered CUDA10.2 binaries predate these requirements. Current code also uses
modern strict frozen-checkpoint loading including `weights_only=True`; replacing
Torch/Python with legacy packages would require a separate compatibility port and
scientific revalidation. No such substitution or downgrade was made. The IBM legacy payload also pins
PowerAI release/tools, DDL and Spectrum MPI; its entire vendor dependency/licensing
closure and native CPU/GPU compatibility were not validated. Public archive
availability does not establish a usable complete legacy environment.

R440 is consistent with the installed CUDA10.2 stack, but cannot execute ordinary
CUDA11.8 libraries. Runtime packages, environment variables or a successful
solver result do not upgrade the kernel driver. No forward-compatibility package
was installed, and none is claimed to provide a supported R440/POWER8 escape.

## 3. What the RocketCE binary actually contains

[Exact CUDA payload](https://conda.anaconda.org/rocketce/linux-ppc64le/pytorch-base-2.0.1-cuda11.8_py310_pb4.21.12_1.conda):
355,342,042 bytes, SHA256
`a2f1cc16ebb107b0b53b8412e93e9f3267a896821b68db8a125aa7a2a964775a`.
The downloaded archive matches published metadata. Only archive members/ELF data
were read; nothing from it was imported or executed.

- `torch/version.py` records2.0.1, CUDA11.8, upstream git identity
  `4a0c48d857f70b2fe09d63a3bbadfcc71e2de057`.
- `_C`, `libc10_cuda`, `libtorch_cuda`, `libtorch_cpu` and `libtorch_python` are
  actual payload members. Inspected ELF headers identify 64-bit little-endian
  PowerPC (`e_machine=21`), not x86_64.
- The compiled CUDA architecture string in `libtorch_python.so` includes
  `sm_60 sm_70 sm_75 sm_80 sm_86 sm_90`; installed CUDAConfig header records the
  `compute_60`/`sm_60` code-generation pair, and recipe enables `USE_CUDA=1`.
  Thus P100 kernels are a declared compiled target. No P100 kernel was launched.
- The CPU compile-settings string contains `-mcpu=power8 -mtune=power8`; the
  inspected binaries reference GLIBC_2.17. These are encouraging binary facts,
  not proof that every bundled routine/dependency works on POWER8/RHEL7.5.
- Recipe includes `cpu_opt_arch: power8`, `cpu_opt_tune: power9`, VSX-related
  patches, and a `ppc_arch: p9` build variant. The tuning setting alone does not
  imply a mandatory POWER9 ISA, but it does not certify the whole dependency set.

[Binary evidence and compact extracts](BINARY_INSPECTION.json). Rocket's
[release announcement](https://community.rocketsoftware.com/rocket-ce-for-power-38/releasing-rocketce-pytorch-v2-0-1-with-opence-v1-9-1-8802)
qualifies POWER9/POWER10, with GPU use on POWER9; POWER8 is outside that claim.
A [first-person Rocket forum report](https://community.rocketsoftware.com/rocket-ce-for-power-38/install-pytorch-environment-on-power8-arch-9427)
describes POWER8 Python3.9 illegal instructions even with driver520.61.05. That
report is not a test of our Python3.10 payload, but its concern is independently
corroborated here: the selected Python3.10.11 archive also has POWER9 compilation
flags in its recipe and two saved sysconfig files. We did not execute it on
POWER8 to deliberately provoke a crash.

## 4. Dependency solving and PyG

Local Conda25.7.0/libmamba cross-platform **dry-runs only**; all target prefixes
remained absent. Dedicated external caches, no changes to POLUS/CPU environment.
Virtual CUDA11.8, glibc2.17, Linux3.10 and POWER8 were supplied explicitly as
hypothetical solver inputs. They do not assert a driver upgrade or enforce all
CPU instructions used by package binaries. [Commands/outcomes](SOLVER_RESULTS.json).

| Requested plan | Actual dependency result |
|---|---|
| 1.9.1 label plus defaults/forge | FAIL: label omits required `_pytorch_select 2.0.*`. |
| Add RocketCE main, exact CUDA Torch payload, Python3.10 | PASS:89 Conda packages. Selected RocketCE Python later identified as POWER9-targeted. |
| Full technical stack, exact working research Python3.10.22 build | FAIL: Python requires libzlib>=1.3.2 whereas Torch constrains zlib<1.3. |
| Full technical stack, unpinned Python3.10 | PASS:111 packages, but selects the POWER9-targeted Python. |
| Full technical stack, defaults Python3.10.13 `h112ed23_0` | PASS:105 packages plus separately selected universal PyG wheel; whole runtime UNVERIFIED. |

The alternative Python3.10.13 build matches the interpreter previously imported
successfully on the POWER8 frontend in the preceding environment audit; it is not
the existing research3.10.22 prefix, and was not installed here. This removes a
known interpreter selection problem, **not all CPU-ISA or runtime risks**.
The native extensions, NumPy/SciPy/OpenBLAS, CUDA/cuDNN/NCCL and remaining linked
libraries still need the platform qualification and native tests below.

Key resolved versions in the alternative105-package plan:

| Dependency | Version / exact build |
|---|---|
| Python | 3.10.13 `h112ed23_0` |
| Torch payload | 2.0.1 `cuda11.8_py310_pb4.21.12_1` |
| CUDA runtime | 11.8.0 `h70ec917_2` |
| cuDNN | 8.8.1_11.8 `h9ceb136_2` |
| NCCL | 2.17.1 `cuda11.8_2` |
| NumPy | 1.23.5 `py310hba26dad_0` |
| SciPy | 1.10.1 `py310h6a19e1b_1` |
| scikit-learn | 1.2.2 `py310h4a02239_1` |
| OpenBLAS | 0.3.23 `openmp_h1234567_1` |
| torch_scatter | 2.1.1 `cuda11.8_py310_1` |
| torch_sparse | 0.6.17 `cuda11.8_py310_1` |
| PyG | 2.4.0 `py3-none-any` wheel |

All105 exact URLs/versions/builds/dependency constraints/hashes and the PyG URL:
[PROPOSED_TECHNICAL_LOCK.json](PROPOSED_TECHNICAL_LOCK.json). This is a solved
**technical Torch/PyG/metrics stack**, not a runtime-certified environment or all
optional project extras (e.g. MLflow). Most package files have metadata hashes
only; whole-file verification was performed for the Torch and Rocket Python
archives, not all105 downloads.

PyG2.4's wheel is pure Python, platform-independent, and `pip --dry-run --no-deps`
selects it for Python3.10/ppc64le; its core dependency names are present in the
Conda plan. The no-deps dry-run is not a second complete dependency solve.
The queried [official cu118 extension wheel index](https://data.pyg.org/whl/torch-2.0.0+cu118.html)
has **zero ppc64le entries**, so its x86 wheels cannot be reused. Instead RocketCE
has ppc64le CUDA scatter2.1.1 and sparse0.6.17 packages matching Torch2.0.* in their
metadata. Their presence does not establish ABI/kernel/POWER8 correctness.
Basic edge-index GCN/SAGE can use native Torch aggregation without optional
extensions; the technical benchmark must record the actual backend and avoid
silently introducing a different dependency path. No neighbor-sampling or
multi-GPU Q-learning is proposed.

## 5. OS, existing driver and CUDA11.8 upgrade feasibility

Fresh approved read-only frontend metadata:
**RHEL7.5 Maipo**, kernel **3.10.0-693.1.1.el7.ppc64le**, glibc**2.17**,
GCC**4.8.5-44.el7**. Matching kernel-devel/headers and `/lib/modules/.../build`
exist. Loaded module440.33.01 has matching vermagic; its module and user libraries
come from RPM/DKMS packages (`dkms2.6.1`), not a proven runfile installation.

Earlier independently collected native Dense environments on **C2/C3/C4** report
kernel **3.10.0-693.el7.ppc64le**, glibc2.17. These differ from the frontend's
kernel. Per-compute-node current OS minor version, matching headers, firmware
and driver transaction readiness were **not refreshed by a new job**; administrator
verification is mandatory. Frontend headers are not evidence for C2/C3/C4.
[Sanitized host/RPM/driver evidence](HOST_AND_DRIVERS.json).

| Candidate | Verified availability | Support / feasibility boundary |
|---|---|---|
| R450.80.02 | NVIDIA RHEL8/ppc64le driver, CUDA libraries and DKMS RPMs exist; none found in queried RHEL7/ppc64le repo. | CUDA11.x minor compatibility may allow11.8 on qualifying drivers with limitations; not a certificate for POWER8/P100/RHEL7.5. R450's own OS matrix marks RHEL7 POWER unsupported. |
| R520.61.05 | NVIDIA RHEL8/ppc64le driver/DKMS RPMs and ppc64le redistribution archive exist. | More direct CUDA11.8-era driver candidate, but packages are EL8 and native toolkit table qualifies POWER9/RHEL8, not this host. No supported in-place EL7 POWER8 transaction established. |

Exact candidate RPM URLs and requires are embedded in HOST_AND_DRIVERS.json.
Sources: [RHEL7 ppc64le repo](https://developer.download.nvidia.com/compute/cuda/repos/rhel7/ppc64le/),
[RHEL8 ppc64le repo](https://developer.download.nvidia.com/compute/cuda/repos/rhel8/ppc64le/),
[R450 support matrix](https://docs.nvidia.com/datacenter/tesla/tesla-release-notes-450-80-02/index.html),
[CUDA11.8 native platform table](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-installation-guide-linux/).

Do not infer GLIBC_2.28 as an actual driver-library requirement merely from an
EL8 filename: inspected RPM requires include GLIBC_2.17. The blockers are the
unsupported combined platform/packaging and untested kernel integration, not an
invented driver glibc error. EL8 RPMs also depend on their own driver packages and
DNF integration; mixing them into EL7 is not a supported proposed transaction.

[CUDA11.8 release notes](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-toolkit-release-notes/index.html)
list520.61.05 as the corresponding Linux driver; their450.80.02 minimum table is
labelled x86_64. The [minor compatibility guide](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
adds restricted features, explicit architecture targeting and PTX limitations.
Neither number alone establishes POWER support. R440 is below this CUDA11 path.
Availability of a `cuda-compat` RPM is not permission to substitute user libraries
for the running module or bypass support checks. P100 requires the proprietary
kernel-module flavor; [open modules require Turing or newer](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-installation-guide-linux/).
No binary patch, override, module compilation/unload or installation attempted.

## Compatibility classifications

| Claim | Status established by this investigation |
|---|---|
| Package is real ppc64le CUDA Torch2.0.1 with declared P100 target | **CONFIRMED binary/metadata facts**, not native runtime compatibility |
| Torch payload and alternative105-package plan work together on POWER8/P100 | **PLAUSIBLE, UNVERIFIED**; no imports/kernels performed |
| Stock RocketCE Python3.10.11 is an appropriate POWER8 interpreter | **UNSUPPORTED ISA target**; POWER9 flags independently inspected |
| Current R440 can execute ordinary CUDA11.8 Torch | **BLOCKED** by the driver/runtime generation mismatch |
| Exact research Python3.10.22 can be reused in this pinned Rocket stack | **BLOCKED dependency solve**: libzlib/zlib constraints |
| R450/R520 is a vendor-qualified in-place upgrade on POLUS POWER8/RHEL7.5 | **NOT ESTABLISHED / outside cited official platform qualification** |
| All dependencies and PyG CUDA operators are runtime-correct | **UNVERIFIED**; dependency solving and binary indexing are insufficient |

## 6. Recommendation and safe next step

**Do not upgrade a shared node solely because the Torch solver succeeds.**
Request administrator/vendor qualification of the exact POWER8/P100/OS/kernel
combination first, or an approved supported execution platform. No administrator
was contacted and no credentials requested/used. A driver-only update cannot
make a POWER9-targeted Python executable compatible with POWER8.

The105-package Python3.10.13 plan is a **plausible isolated qualification candidate**
after driver/platform approval. Preserve the existing CPU prefix, activation,
CUDA10.2 toolkit and campaign snapshots. Install nothing into them. Inspect all
remaining binary ISA/ABI requirements, then separately authorize installation
and a bounded native one-GPU technical job; validation must use the original
frozen checkpoints/split with masked test labels. No production campaign release
until real imports, CUDA kernels, model/surgery/checkpoint tests, CPU/GPU numerical
comparisons, backend accounting and synchronized timing/memory have passed.
The existing Flickr runner's device/reporting/cache gaps from the
[preceding audit](../flickr_structural_pruning_20261101/cuda_audit_20261101/REPORT.md)
remain; this feasibility investigation did not fix or bypass them.

[ADMINISTRATIVE_PLAN.md](ADMINISTRATIVE_PLAN.md) specifies canary isolation,
maintenance, backups, health gates and rollback. It is a conditional plan, not
instructions executed on POLUS or a currently supported upgrade recipe.

No pytest/model tests were run: scientific source is unchanged and this task
executes metadata parsing/solver/integrity checks only. Native CUDA compatibility
and performance remain **NOT TESTED**. All campaign conditions remain blocked;
no accuracy, speedup or180-minute eligibility claim follows from this work.
Raw evidence/packages stay external at
`${GNN_ARTIFACT_ROOT}/polus-cuda-package-feasibility-20261101/`;
[provenance](PROVENANCE.json) and [verification](VERIFICATION.json).

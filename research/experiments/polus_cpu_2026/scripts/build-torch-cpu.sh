#!/usr/bin/env bash
# Native POWER8 recipe for upstream PyTorch v2.0.1; no project experiments run here.
set -euo pipefail
: "${CONDA_PREFIX:?Activate the separate build environment first}"
: "${LSB_JOBID:?Build inside an LSF CPU allocation}"
: "${CXX:?Conda C++ compiler activation is missing}"
: "${CC:?Conda C compiler activation is missing}"
if [ "$(uname -m)" != ppc64le ]; then
    echo 'Expected native linux-ppc64le build host.' >&2
    exit 1
fi
source_dir="${1:?Usage: bash build-torch-cpu.sh /path/to/pytorch-v2.0.1}"
cd "$source_dir"
if [ "$(git describe --tags --exact-match HEAD)" != v2.0.1 ]; then
    echo 'Expected upstream tag v2.0.1.' >&2
    exit 1
fi
if [ -n "$(git status --porcelain --untracked-files=no --ignore-submodules=dirty)" ]; then
    recorded_diff="$source_dir/../../patches/pytorch-vsx.diff"
    if [ ! -s "$recorded_diff" ] || ! git diff --cached --quiet || \
       ! git diff --no-ext-diff --ignore-submodules=dirty | cmp -s - "$recorded_diff"; then
        echo 'Tracked changes differ from the recorded upstream VSX backport.' >&2
        exit 1
    fi
fi
# Validate root gitlinks. Nested components of disabled CUDA/Kineto/FBGEMM
# features are not required; setup.py checks the remaining required scaffolding.
submodule_status="$(git submodule status)"
while IFS= read -r status_line; do
    case "$status_line" in
        [-+U]*)
            echo 'Submodules are missing or differ from the pinned tree.' >&2
            exit 1
            ;;
    esac
done <<< "$submodule_status"
"$CC" --version
"$CXX" --version
export CFLAGS="${CFLAGS:-} -mcpu=power8 -mtune=power8"
export CXXFLAGS="${CXXFLAGS:-} -mcpu=power8 -mtune=power8"
export CMAKE_PREFIX_PATH="$CONDA_PREFIX"
export OpenBLAS_HOME="$CONDA_PREFIX"
export BLAS=OpenBLAS
# GNU BFD ld 2.39 fails REL24 relocations for this large POWER8 DSO.
# Use the gold linker supplied by the same isolated Conda toolchain.
export CMAKE_SHARED_LINKER_FLAGS="${LDFLAGS:-} -fuse-ld=gold -Wl,--stub-group-size=1000000"
export CMAKE_MODULE_LINKER_FLAGS="$CMAKE_SHARED_LINKER_FLAGS"
export CMAKE_EXE_LINKER_FLAGS="$CMAKE_SHARED_LINKER_FLAGS"
export USE_CUDA=0 USE_CUDNN=0 USE_NCCL=0
export USE_DISTRIBUTED=0 USE_KINETO=0 USE_MKLDNN=0
export USE_FBGEMM=0 USE_NNPACK=0 USE_QNNPACK=0 USE_PYTORCH_QNNPACK=0
export USE_XNNPACK=0 USE_ITT=0 BUILD_TEST=0 BUILD_CAFFE2=0
export PYTORCH_BUILD_VERSION=2.0.1+polus.power8.1 PYTORCH_BUILD_NUMBER=1
export MAX_JOBS="${POLUS_BUILD_JOBS:-${LSB_DJOB_NUMPROC:-4}}"
python setup.py --cmake bdist_wheel
echo 'Wheel created under dist/. Install it into the same Conda environment separately.'

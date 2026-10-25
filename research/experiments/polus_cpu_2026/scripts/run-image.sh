#!/usr/bin/env bash
set -euo pipefail
if [ -z "${LSB_JOBID:-}" ]; then
    echo 'Start an LSF GPU allocation first; see README.md.' >&2
    exit 1
fi
if [ "$(uname -m)" != ppc64le ]; then
    echo 'Expected a ppc64le compute node.' >&2
    exit 1
fi
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo 'CUDA_VISIBLE_DEVICES is missing. Verify how LSF exposes assigned GPUs before running.' >&2
    exit 1
fi
if command -v singularity >/dev/null 2>&1; then
    runtime=singularity
elif command -v apptainer >/dev/null 2>&1; then
    runtime=apptainer
else
    echo 'Singularity/Apptainer is not loaded.' >&2
    exit 1
fi
image="${POLUS_IMAGE:-$PWD/polus-wmlce-1.6.2.sif}"
test -r "$image" || { echo "Missing image: $image" >&2; exit 1; }
if [ "$#" -eq 0 ]; then
    echo 'Usage: bash run-image.sh python smoke_train.py' >&2
    exit 1
fi
# --cleanenv avoids host Python/CUDA paths; preserve the scheduler GPU mask.
export SINGULARITYENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
export APPTAINERENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
export SINGULARITYENV_OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export APPTAINERENV_OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export SINGULARITYENV_POLUS_CHECKPOINT="${POLUS_CHECKPOINT:-smoke-checkpoint.pt}"
export APPTAINERENV_POLUS_CHECKPOINT="${POLUS_CHECKPOINT:-smoke-checkpoint.pt}"
exec "$runtime" exec --cleanenv --nv --bind "$PWD:/work" --pwd /work "$image" \
    /bin/bash -c '
        set -e
        source /opt/anaconda/etc/profile.d/conda.sh
        conda activate wmlce
        exec "$@"
    ' polus-command "$@"

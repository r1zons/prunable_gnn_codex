#!/usr/bin/env bash
# Read-only inventory. Does not install packages or start training.
set -u
show() {
    printf '\n--- %s ---\n' "$*"
    "$@" 2>&1 || true
}
show date -u
show hostname
show uname -a
show lscpu
show cat /etc/os-release
show getconf GNU_LIBC_VERSION
show free -h
show df -h .
if [ -r /proc/device-tree/model ]; then
    show tr '\0' '\n' < /proc/device-tree/model
fi
show nvidia-smi
show nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
show nvcc --version
show gcc --version
show singularity --version
show apptainer --version
show python3 --version
show conda --version
if type module >/dev/null 2>&1; then
    show module list
    show module avail
fi
show bqueues
show bhosts
printf '\nLSF job ID: %s\nCUDA_VISIBLE_DEVICES: %s\n' "${LSB_JOBID-unset}" "${CUDA_VISIBLE_DEVICES-unset}"

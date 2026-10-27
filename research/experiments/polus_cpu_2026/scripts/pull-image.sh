#!/usr/bin/env bash
set -euo pipefail
if [ "$(uname -m)" != ppc64le ]; then
    echo 'Run on the POWER ppc64le host. These images cannot execute on x86 or Apple Silicon.' >&2
    exit 1
fi
if command -v singularity >/dev/null 2>&1; then
    runtime=singularity
elif command -v apptainer >/dev/null 2>&1; then
    runtime=apptainer
else
    echo 'Load the installed Singularity module, then retry.' >&2
    exit 1
fi
# Immutable manifest verified through Docker Registry on 2026-10-23.
ref='docker://ibmcom/powerai@sha256:3e9dd10881f230303e44eee369faf74d680ebaa50efc2bc7d31fca0c8a7f2654'
image="${POLUS_IMAGE:-$PWD/polus-wmlce-1.6.2.sif}"
if [ -e "$image" ]; then
    echo "Already exists: $image. Refusing to overwrite." >&2
    exit 1
fi
"$runtime" pull "$image" "$ref"
sha256sum "$image" > "$image.sha256"
"$runtime" inspect "$image"

#!/bin/bash
# Run from the staged POLUS campaign directory, once.
set -euo pipefail
export LC_ALL=C LANG=C
umask 077
sha256sum --quiet -c source.sha256
test ! -e submission
mkdir submission
sha256sum source.sha256 source-commit.txt preflight.json > submission/pins.sha256
source_sha=$(cat source-commit.txt)
cat > submission/INTENT <<RECEIPT
status=INTENT
created=$(date -u +%Y-%m-%dT%H:%M:%SZ)
source=$source_sha
name=mgn-block-distillation-20261121
queue=normal
hosts=polus-c2-ib.bmc.hpc.cs.msu.ru polus-c3-ib.bmc.hpc.cs.msu.ru polus-c4-ib.bmc.hpc.cs.msu.ru
slots=4
threads=4
interop_threads=1
exclusive=yes
memory_mb=8192
wall_minutes=45
gpus=1
gpu_request=num=1:mode=shared:j_exclusive=yes
RECEIPT
if bsub -J mgn-block-distillation-20261121 -q normal -W 0:45 -n 4 -x \
    -m "polus-c2-ib.bmc.hpc.cs.msu.ru polus-c3-ib.bmc.hpc.cs.msu.ru polus-c4-ib.bmc.hpc.cs.msu.ru" \
    -R "span[hosts=1] rusage[mem=8192]" -M 8192 \
    -gpu "num=1:mode=shared:j_exclusive=yes" -cwd "$PWD" \
    -o "$PWD/lsf.%J.out" -e "$PWD/lsf.%J.err" \
    < scripts/run_cylinderflow_block_distillation.lsf \
    > submission/bsub.out 2> submission/bsub.err; then
    rc=0
else
    rc=$?
fi
printf '%s\n' "$rc" > submission/exit-code
cat submission/bsub.out submission/bsub.err
pattern='^Job <([0-9]+)> is submitted to queue <normal>\.$'
reply=$(cat submission/bsub.out)
if [[ $rc -eq 0 && "$reply" =~ $pattern ]]; then
    printf 'status=ACCEPTED\njob_id=%s\ncreated=%s\n' "${BASH_REMATCH[1]}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        > submission/ACCEPTED
    cat submission/ACCEPTED
else
    printf 'No exact acceptance receipt. Inspect scheduler and stored response before any future action.\n' >&2
    exit 1
fi

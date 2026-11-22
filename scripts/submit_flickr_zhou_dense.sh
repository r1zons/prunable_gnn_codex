#!/bin/bash
# Run from the immutable POLUS campaign root.
set -euo pipefail
export LC_ALL=C LANG=C
umask 077
stage=${1:?Use parity or train}
case "$stage" in
  parity) wall=0:10 ;;
  train) test -f parity/gate.json; wall=0:45 ;;
  *) exit 2 ;;
esac
sha256sum --quiet -c source.sha256
mkdir -p submission
receipt="submission/$stage"
mkdir "$receipt"
{
  printf 'status=INTENT\ncreated=%s\nsource=%s\n' "$(date -u +%FT%TZ)" "$(cat source-commit.txt)"
  printf 'stage=%s\nqueue=normal\nwall=%s\nslots=4\nexclusive=yes\nmemory_mb=8192\ngpus=1\n' "$stage" "$wall"
  sha256sum source.sha256 upstream.sha256
} > "$receipt/INTENT"
export ZHOU_STAGE="$stage"
if bsub -J "flickr-zhou-$stage-20261122" -q normal -W "$wall" -n 4 -x \
   -m "polus-c2-ib.bmc.hpc.cs.msu.ru polus-c3-ib.bmc.hpc.cs.msu.ru polus-c4-ib.bmc.hpc.cs.msu.ru" \
   -R "span[hosts=1] rusage[mem=8192]" -M 8192 -gpu "num=1:mode=shared:j_exclusive=yes" \
   -cwd "$PWD" -o "$PWD/lsf.%J.out" -e "$PWD/lsf.%J.err" \
   < scripts/run_flickr_zhou_dense.lsf > "$receipt/bsub.out" 2> "$receipt/bsub.err"; then
    rc=0
else
    rc=$?
fi
printf '%s\n' "$rc" > "$receipt/exit-code"
cat "$receipt"/bsub.{out,err}
reply=$(cat "$receipt/bsub.out")
pattern='^Job <([0-9]+)> is submitted to queue <normal>\.$'
if [[ $rc -eq 0 && "$reply" =~ $pattern ]]; then
    printf 'status=ACCEPTED\njob_id=%s\n' "${BASH_REMATCH[1]}" > "$receipt/ACCEPTED"
    cat "$receipt/ACCEPTED"
else
    exit 1
fi

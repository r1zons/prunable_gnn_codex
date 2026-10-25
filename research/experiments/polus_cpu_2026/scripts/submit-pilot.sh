#!/usr/bin/env bash
# Submit independent seed-101 conditions to normal, restricted to polus-c4.
set -euo pipefail
usage() {
    printf 'Usage: bash %s [--dry-run|--prepare] [--all|CONDITION]\n' "$(basename "$0")"
}
campaign="${POLUS_CAMPAIGN:-pilot-cpu-101-c4-v2}"
[[ "$campaign" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]] || exit 2
export POLUS_CAMPAIGN="$campaign"
mode=submit
all=false
condition="${POLUS_CONDITION:-cora_gcn_l3_h32}"
condition_given=false
while (($#)); do
    case "$1" in
        --dry-run) mode=dry-run ;;
        --prepare) mode=prepare ;;
        --all) all=true ;;
        --help|-h) usage; exit 0 ;;
        --*) usage >&2; exit 2 ;;
        *)
            if "$condition_given"; then usage >&2; exit 2; fi
            condition="$1"; condition_given=true
            ;;
    esac
    shift
done
if "$all" && "$condition_given"; then usage >&2; exit 2; fi
if "$all"; then
    conditions=(cora_gcn_l3_h32 cora_graphsage_l3_h32 citeseer_gcn_l3_h32 citeseer_graphsage_l3_h32 pubmed_gcn_l3_h32 pubmed_graphsage_l3_h32)
else
    conditions=("$condition")
fi
for condition in "${conditions[@]}"; do
    case "$condition" in
        cora_gcn_l3_h32|cora_graphsage_l3_h32|citeseer_gcn_l3_h32|citeseer_graphsage_l3_h32|pubmed_gcn_l3_h32|pubmed_graphsage_l3_h32) ;;
        *) printf 'Unknown condition: %s\n' "$condition" >&2; exit 2 ;;
    esac
done
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
job_file="$script_dir/pilot-cpu.lsf"
for directive in '#BSUB -q normal' '#BSUB -n 4' '#BSUB -W 03:00' \
    '#BSUB -R "span[hosts=1]"' '#BSUB -m polus-c4-ib.bmc.hpc.cs.msu.ru' '#BSUB -x' \
    '#BSUB -o logs/lsf/pilot-cpu.%J.out' '#BSUB -e logs/lsf/pilot-cpu.%J.err'; do
    grep -Fxq -- "$directive" "$job_file" || {
        printf 'Missing LSF directive: %s\n' "$directive" >&2; exit 1
    }
done
printf 'normal; polus-c4 only; 180 minutes/job; 4 CPU slots; exclusive host; seed 101\n'
if [[ "$mode" == dry-run ]]; then
    for condition in "${conditions[@]}"; do
        printf 'Condition=%s campaign=%s output=results/%s/jobs/%s data=data-by-condition/%s/%s\n' "$condition" "$campaign" "$campaign" "$condition" "$campaign" "$condition"
        printf 'Would submit: POLUS_CAMPAIGN=%s POLUS_CONDITION=%s bsub -J %s < "%s"\n' "$campaign" "$condition" "$campaign-$condition" "$job_file"
    done
    printf 'No files changed; no jobs submitted.\n'
    exit 0
fi
if [[ "$mode" == submit ]]; then
    command -v bsub >/dev/null || { echo 'Run this on Polus with bsub available.' >&2; exit 1; }
fi
task_root="$(cd -- "$HOME/gnn-polus-20261023" && pwd -P)"
[[ "$script_dir" == "$task_root" ]] || { echo 'Install scripts in ~/gnn-polus-20261023.' >&2; exit 1; }
cd "$task_root"
source ./activate-polus.sh
python - <<'PY'
import json
from pathlib import Path
ready = json.loads(Path('logs/environment-ready.json').read_text())
if ready.get('status') != 'READY_FOR_MANUAL_PILOT' or ready.get('vsx_regression') != 'PASS':
    raise SystemExit('Environment verification is required')
PY
# Check the complete selected batch before submitting its first job.
for condition in "${conditions[@]}"; do
    [[ ! -e "results/$campaign/jobs/$condition/seed_101/$condition" ]] || {
        echo "Results already exist: $condition. Inspect them before retrying." >&2; exit 1
    }
    [[ ! -e "submissions/$campaign/$condition" ]] || {
        echo "Submission already recorded or unresolved: $condition. Do not submit twice." >&2; exit 1
    }
done
for condition in "${conditions[@]}"; do
    python prepare-condition-data.py --source "$task_root/data" \
        --destination "$task_root/data-by-condition/$campaign/$condition" --condition "$condition"
    python prepare-gnn-pilot.py --repo-root "$task_root/repo" \
        --expected-commit 72293149d8fff186dd268789e7b8fea5848c9360 \
        --output-root "$task_root/results/$campaign/jobs/$condition" \
        --data-root "$task_root/data-by-condition/$campaign/$condition" --condition "$condition"
done
if [[ "$mode" == prepare ]]; then
    printf 'All selected configs and verified raw inputs prepared. No jobs submitted.\n'
    exit 0
fi
mkdir -p logs/lsf "submissions/$campaign"
for condition in "${conditions[@]}"; do
    claim="submissions/$campaign/$condition"
    mkdir "$claim" || { echo "Submission already claimed: $condition" >&2; exit 1; }
    printf 'INTENT\n' > "$claim/status"
    cp prepare-gnn-pilot.py "$claim/prepare-gnn-pilot.py"
    cp pilot-cpu.lsf "$claim/pilot-cpu.lsf"
    (cd "$claim" && sha256sum prepare-gnn-pilot.py pilot-cpu.lsf > launcher.sha256)
    export POLUS_CONDITION="$condition"
    if bsub -J "${campaign}-${condition}" < "$claim/pilot-cpu.lsf" > "$claim/response.txt" 2>&1; then
        cat "$claim/response.txt"
        job_id="$(sed -n 's/^Job <\([0-9][0-9]*\)>.*$/\1/p' "$claim/response.txt")"
        if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then
            printf 'UNKNOWN\n' > "$claim/status"
            echo 'Unrecognized bsub response. Check LSF before any retry.' >&2; exit 1
        fi
        printf '%s\n' "$job_id" > "$claim/job_id"
        printf 'ACCEPTED\n' > "$claim/status"
        printf '%s %s\n' "$condition" "$job_id"
    else
        cat "$claim/response.txt" >&2
        printf 'UNRESOLVED\n' > "$claim/status"
        echo 'Submission stopped. Inspect response and LSF; no automatic retry.' >&2; exit 1
    fi
done

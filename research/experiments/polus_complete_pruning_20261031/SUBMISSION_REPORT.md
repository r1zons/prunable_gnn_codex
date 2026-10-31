# Submission report — complete POLUS campaign, 2026-10-31

**SUBMITTED: 55/55 LSF jobs accepted**, IDs **1611561–1611615**. They cover
18 independent dense conditions (six verified seed101 reuses,12 new fits for
seeds202/303) and **252 pruning tasks**, including the28-run Cora seed101 pilot.
**BLOCKED:0; REJECTED:0; UNRESOLVED:0.** No scientific condition was removed.
**RUNNING: not queried. COMPLETED: not queried.** Acceptance is not experiment
completion; no campaign quality, timing outcome or success claim is made.

The launch-only session ends after verifying acceptance and writing this report.
No scheduler monitoring, scientific output collection/analysis, extra experiment,
environment/package update, Git push or merge was performed. The existing network
exception was replaced by the user using the verified physical gateway; trusted
SSH access was subsequently confirmed. Credentials and VPN settings were unchanged.

## Pins and evidence

- Branch: `campaign/complete-pruning-20261031`.
- Executable source: `55d0fc242b543c5725276144bd8e970eedc48ecd`;717 source files verified after transfer.
- Source archive SHA256: `7ef42db5eeddd3403157c7a2cb534c990bffd09bafe315a19614f6586aff59b6`.
- Deployment protocol SHA256: `5e400a59c27fe98506f16eb0055098de780edd26f0b209695be3a397f1edf256`.
- Frozen graph/input manifest SHA256:
  `414fbd317890e12c3be85fbe3bc21b8e09326662a8b66b6de5305b8a9fab265f`.
- Immutable receipt archive: `submission-evidence-20261031T021106Z.tar.gz`;
  SHA256 `42171aac9656c2b4225150842068cec7e643d396c111d366e0aebc2252ecec94`.
- Every acceptance response, unique job ID, launcher hash and INTENT/ACCEPTED
  receipt was independently rechecked from that archive. No bsub was retried.
- [Machine-readable acceptance/dependency manifest](submission.json),
  [native preflight](native_preflight.json), [verified input hashes](verified_inputs.json),
  [pytest evidence](verification.json), [fixed matrix](manifest.json),
  [scientific identities and execution protocol](README.md).

Raw local evidence:
`${GNN_ARTIFACT_ROOT}/polus-complete-pruning-20261031/`.
Remote artifacts:
`${POLUS_TASK_ROOT}/results/polus-complete-pruning-20261031/`.
Pinned executable snapshot:
`${POLUS_TASK_ROOT}/source-snapshots/complete-pruning-55d0fc242b543c5725276144bd8e970eedc48ecd/`.
Access/account-specific values occur only in external private evidence.

## Tests and technical gates

| Check | PASS | FAIL | SKIPPED |
|---|---:|---:|---:|
| Focused orchestration pytest |20|0|0|
| Complete pytest suite |547|0|6|
| Native POWER8 synthetic method smoke |14|0|0|
| Archived native dense checkpoint provenance/reload/validation |6|0|0|

Six pytest skips: two CUDA tests on a CPU-only host; four opt-in archived-Cora
Lookahead checks not enabled in this invocation. Those four are not reported as
passes. Native smoke uses20 synthetic nodes, both models/all seven methods,
real forwards, test-label perturbation, exact tensor projection/reload and honest
evaluation counts; it is not the scientific Cora pilot. The Cora pilot is submitted
as jobs1611563/1611566; its automated acceptance job is1611567. Its result has not
been read. Remaining training/reuse and pruning cannot release on a failed pilot.

One preparation wrapper attempt failed **before any smoke ran** because optional
`/usr/bin/time` is absent on POLUS. Its log was preserved; the identical pinned
smoke then passed when invoked directly. No package was installed and no failed
scheduler submission was repeated. There are no unresolved technical test failures.

## Resources and guardrails

Current native environment: Python3.10.22, Torch2.0.1, PyG2.4.0; activation SHA256
`ac464bbf980c499281c31ccb018b0e9e172a00df6c1b12627d3f075c45121fce`.
Normal queue open,180-minute absolute limit; user active-job limit4 and pending
capacity300; zero existing user jobs at preflight. Exclusive C4 has20 CPU slots
and1TB RAM. Each accepted job requests4 slots/threads, one host, normal, exclusive
C4, rusage[mem=8192], memory limit8192 in the verified site MB units.

Train/reuse10min, validation5min, gate5min; pruning batches Cora20min,
CiteSeer30min, PubMed45min. Maximum aggregate active wall reservation is
**845min (14h5min),56.33 allocated slot-hours**. This is a cap, not measured runtime;
exclusive allocation also withholds other node cores. Expected1–3h is a broad
preflight estimate, not a completion promise. Queue waiting is separate and
unmeasured. Costs were set from prior POLUS evidence and six frozen-forward means
(.0092–.0489s), not pilot accuracy. No later capacity or scientific results were read.

All seven method implementations are preserved. Native V2 Q uses30 episodes and
soft-target STOP; it is not constrained V4 Q. Sequential Random is the existing
V3 reachability-filtered action sampler with magnitude channel choice. Greedy
and audited depth-2 Lookahead share threshold stopping and allow overshoot;
static methods retain nearest-width fallback. Six CiteSeer/30% Sequential Random
cells are predeclared exact-band-infeasible outcomes and remain in the matrix;
they must not be counted as successful30% pruning or silently repaired. Achieved
compression, masks, costs and terminal reasons require separate later analysis.
These method/controller packages do not isolate a single causal learning effect.

The reused six checkpoint byte hashes match the independently verified20261024
archive. Raw-file lineage, preserved native graph tensors, architecture, training
settings, split/seed identity and reloaded validation metrics were checked. The
original campaign had no contemporaneous graph tensor digest; this lineage is
reconstructed from preserved raw/processed inputs and is explicitly recorded.
New fits use the same unchanged DenseTrainer, model and split implementations;
no dense models are trained locally. No historical V1–V4 evidence or manuscript
was modified.

## Accepted graph

| Node | LSF ID | Required successful IDs | Wall min |
|---|---:|---|---:|
| `train-cora-gcn-s101` | 1611561 | — | 10 |
| `validate-cora-gcn-s101` | 1611562 | 1611561 | 5 |
| `prune-cora-gcn-s101` | 1611563 | 1611562 | 20 |
| `train-cora-graphsage-s101` | 1611564 | — | 10 |
| `validate-cora-graphsage-s101` | 1611565 | 1611564 | 5 |
| `prune-cora-graphsage-s101` | 1611566 | 1611565 | 20 |
| `pilot-gate` | 1611567 | 1611563, 1611566 | 5 |
| `train-cora-gcn-s202` | 1611568 | 1611567 | 10 |
| `validate-cora-gcn-s202` | 1611569 | 1611568 | 5 |
| `prune-cora-gcn-s202` | 1611570 | 1611569 | 20 |
| `train-cora-gcn-s303` | 1611571 | 1611567 | 10 |
| `validate-cora-gcn-s303` | 1611572 | 1611571 | 5 |
| `prune-cora-gcn-s303` | 1611573 | 1611572 | 20 |
| `train-cora-graphsage-s202` | 1611574 | 1611567 | 10 |
| `validate-cora-graphsage-s202` | 1611575 | 1611574 | 5 |
| `prune-cora-graphsage-s202` | 1611576 | 1611575 | 20 |
| `train-cora-graphsage-s303` | 1611577 | 1611567 | 10 |
| `validate-cora-graphsage-s303` | 1611578 | 1611577 | 5 |
| `prune-cora-graphsage-s303` | 1611579 | 1611578 | 20 |
| `train-citeseer-gcn-s101` | 1611580 | 1611567 | 10 |
| `validate-citeseer-gcn-s101` | 1611581 | 1611580 | 5 |
| `prune-citeseer-gcn-s101` | 1611582 | 1611581 | 30 |
| `train-citeseer-gcn-s202` | 1611583 | 1611567 | 10 |
| `validate-citeseer-gcn-s202` | 1611584 | 1611583 | 5 |
| `prune-citeseer-gcn-s202` | 1611585 | 1611584 | 30 |
| `train-citeseer-gcn-s303` | 1611586 | 1611567 | 10 |
| `validate-citeseer-gcn-s303` | 1611587 | 1611586 | 5 |
| `prune-citeseer-gcn-s303` | 1611588 | 1611587 | 30 |
| `train-citeseer-graphsage-s101` | 1611589 | 1611567 | 10 |
| `validate-citeseer-graphsage-s101` | 1611590 | 1611589 | 5 |
| `prune-citeseer-graphsage-s101` | 1611591 | 1611590 | 30 |
| `train-citeseer-graphsage-s202` | 1611592 | 1611567 | 10 |
| `validate-citeseer-graphsage-s202` | 1611593 | 1611592 | 5 |
| `prune-citeseer-graphsage-s202` | 1611594 | 1611593 | 30 |
| `train-citeseer-graphsage-s303` | 1611595 | 1611567 | 10 |
| `validate-citeseer-graphsage-s303` | 1611596 | 1611595 | 5 |
| `prune-citeseer-graphsage-s303` | 1611597 | 1611596 | 30 |
| `train-pubmed-gcn-s101` | 1611598 | 1611567 | 10 |
| `validate-pubmed-gcn-s101` | 1611599 | 1611598 | 5 |
| `prune-pubmed-gcn-s101` | 1611600 | 1611599 | 45 |
| `train-pubmed-gcn-s202` | 1611601 | 1611567 | 10 |
| `validate-pubmed-gcn-s202` | 1611602 | 1611601 | 5 |
| `prune-pubmed-gcn-s202` | 1611603 | 1611602 | 45 |
| `train-pubmed-gcn-s303` | 1611604 | 1611567 | 10 |
| `validate-pubmed-gcn-s303` | 1611605 | 1611604 | 5 |
| `prune-pubmed-gcn-s303` | 1611606 | 1611605 | 45 |
| `train-pubmed-graphsage-s101` | 1611607 | 1611567 | 10 |
| `validate-pubmed-graphsage-s101` | 1611608 | 1611607 | 5 |
| `prune-pubmed-graphsage-s101` | 1611609 | 1611608 | 45 |
| `train-pubmed-graphsage-s202` | 1611610 | 1611567 | 10 |
| `validate-pubmed-graphsage-s202` | 1611611 | 1611610 | 5 |
| `prune-pubmed-graphsage-s202` | 1611612 | 1611611 | 45 |
| `train-pubmed-graphsage-s303` | 1611613 | 1611567 | 10 |
| `validate-pubmed-graphsage-s303` | 1611614 | 1611613 | 5 |
| `prune-pubmed-graphsage-s303` | 1611615 | 1611614 | 45 |

## Later status and collection commands

These commands are supplied for a **later** session; they were not run after
submission. On POLUS, using the existing research account:

```bash
export POLUS_TASK_ROOT="$HOME/gnn-polus-20261023"
source "$POLUS_TASK_ROOT/activate-polus.sh"
cd "$POLUS_TASK_ROOT/results/polus-complete-pruning-20261031"
bjobs -a -w $(python -c 'import json; d=json.load(open("submission-result.json")); print(" ".join(str(j["job_id"]) for j in d["jobs"] if j["accepted"]))')
bjobs -p -u "$USER" -l
bpeek 1611567
bhist -l 1611567
```

All commands and the archive/transfer procedure are in [the protocol](README.md).
After an authorized collection, verify the remote/local archive SHA256 before
extraction into a fresh external directory. Verify exactly252 run directories
against the expected manifest and every `completed.json` output hash. Example
on POLUS after all work has ended, with the pinned source in PYTHONPATH:

```bash
export PYTHONPATH="$POLUS_TASK_ROOT/source-snapshots/complete-pruning-55d0fc242b543c5725276144bd8e970eedc48ecd/src"
python - <<'PYVERIFY'
from pathlib import Path
import json
from gnn_pruning.experiments.complete_campaign import matrix, verify_completed
root=Path.cwd()
protocol=json.loads((root/'protocol.json').read_text())
expected={run['id'] for c in matrix(protocol) for run in c['runs']}
actual={p.name for p in (root/'runs').iterdir() if p.is_dir()}
assert actual == expected and len(expected) == 252
for name in sorted(expected):
    verify_completed(root/'runs'/name)
print('252/252 output hash inventories verified; scientific interpretation remains separate')
PYVERIFY
```

Do not resubmit accepted or ambiguous claims on restart. Failed validations or
pilot acceptance leave dependents pending without starting their computation;
retain that evidence and diagnose in a separately authorized continuation.

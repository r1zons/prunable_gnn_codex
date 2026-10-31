# FLICKR-DENSE-SWEEP-20261031 — full32 authorization

GCN and GraphSAGE × depths2/3/4/5 × widths32/64/128/256:32 independent Dense jobs,
training seed101. One frozen generated60/20/20 split, seed101, shared by all jobs.
Historical seed42 debug evidence is context only, never a research baseline/reuse.
See [historical investigation](HISTORICAL_CONTEXT.md).

Common protocol: existing GCN/mean-GraphSAGE implementations, dropout0.5, Adam,
weight decay0.0005, **common learning rate0.001**, at most500 epochs, validation-loss
early stopping patience50, best validation-loss checkpoint restored. Raw feature
float32 conversion is preserved; no normalization, new architecture components,
pruning or fine-tuning. Each independent job starts from seed101 initialization;
no checkpoint resume/reuse is permitted.

LR0.001 is predeclared conservatively from historical unstable LR0.01 curves,
not selected for short-run accuracy. Technical comparisons of0.01/0.001 at2/64
are recorded for both models; numeric failure blocks launch rather than assigning
different research LRs or removing low-accuracy architectures.

## Frozen inputs and test isolation

Prepare a new plain-tensor graph from the existing trusted Flickr cache, without
downloading, rewriting or copying historical results. Pin raw/processed hashes,
original graph digest, training graph digest and generated split digest. Require
89,250 nodes,500 features,7 classes and899,756 unique reciprocal edge entries.
No reverse edges are appended. Official50/25/25 roles remain a separate protocol.

**Test labels are replaced by−1 before upload.** Held-out labels are quarantined
locally outside Git and are absent from the POLUS training artifact root. Training,
checkpoint verification, resource checks and later architecture analysis evaluate
train/validation only. The proposed finalist set is not a finalized selection.
Held-out evaluation requires a later explicit selection decision; it is not
automatically executed. No test metric is a tuning or selection input.

This is a **full-batch transductive** protocol: features and edges of all nodes
remain in the frozen graph, while held-out labels are absent. It is not an
inductive GraphSAINT benchmark. Training-loss curves use dropout-on train mode;
validation-loss curves use eval mode, and final train/validation metrics describe
the restored best checkpoint rather than the last dropout-on training iterate.

## Native technical gate, separate from research

Three host-specific normal-queue jobs, one each on the available C2/C3/C4, each run
six fresh child processes: both models at2/64, LR0.01/0.001,12epochs; both models
at5/256, LR0.001,6epochs. Fixed short lengths/patience are technical settings only.
Record real epoch times, process peak RSS, strict reload, finite states, actual
LSF host/slots and native CPU frequencies. Each child has independent peak-memory
accounting. No short-run accuracy threshold eliminates a configuration.

Verify current queue policies, memory units, absolute wall limits, host membership,
pending capacity and user jobs. LSF CPU factor may not encode reduced frequency:
record its limitation explicitly and corroborate LSF execution identities with
native frequency observations. Never label a slow node solely from its name.

Use measured small/maximum epoch maxima and operation scaling with1.75 margin
plus120seconds to require a full500-epoch budget within175minutes on each eligible
host; actual jobs request180minutes. Choose memory from3× native peak RSS+2GiB,
minimum8GiB. If an unchanged condition cannot fit, preserve evidence and report the
specific blocker, rather than reducing epochs or silently dropping that condition.

## Scheduling

Submit all32 independent jobs in one submission pass, with **no research-job
dependencies or wait-for-training loops**. Use normal, four CPU slots/threads,
single host and exclusive allocation. Exclusivity bounds this campaign to at most
one job per available eligible node, not a promise of three concurrent jobs.
Fairshare/other users/current availability remain LSF decisions.

The researcher subsequently clarified scheduling: **large jobs on C4; all others
without host binding**. Large means width256 at any depth, plus4/5-layer width128
(12 conditions). The remaining20 can run wherever normal/LSF permits. No frequency
classification or smart distribution is required. LSF reports identical CPU factors
and native frequency interfaces report4023MHz on all three; this does not establish
the cause of measured performance differences. The full-budget wall gate is checked
on C4 for large jobs and on all three verified hosts for unbound jobs.
All32 retain distinct identities/checkpoints/directories. Every bsub has immutable
INTENT, response and ACCEPTED/UNRESOLVED receipts; ambiguous responses are never
retried. Existing claims stop duplicate submission on restart.

## Outputs and later interpretation

Each run records complete epoch JSONL curves, training/validation accuracy, final
macro-F1/loss/class and prediction distributions, zero/one-based best epoch,
executed epochs, stopping/instability flags, parameter/checkpoint sizes, training
wall, epoch times, peak process RSS, actual host/threads/CPU/software, strict reload,
source/config/input/split/tensor hashes. Best partial checkpoint and flushed curves
survive Python failures; uncatchable LSF kills may leave only earlier partial files.
Scheduler completion/accounting must be collected separately before claiming DONE.

After completion, collector requires all32 accepted IDs and scheduler success,
checks transferred archive SHA256 and extracts safely into a new external group.
Analysis verifies32 output inventories, architecture/source/config/split/checkpoint
identities and strict reload before creating RESULTS.csv/ANALYSIS.md and figures.
Missing/invalid results produce a completeness report, no scientific ranking.

Plots use saved learning curves, validation accuracy/macro-F1 versus model size,
and quality/cost **separately by actual execution host**. No cross-host walltime
ranking of architectures or inferred pruning/inference speedup is valid. One seed
supports screening only. Recommend at most two finalists per model, ordered by
validation macro-F1, accuracy, then size, for later seeds202/303 confirmation;
never launch those additional fits automatically. Current three-layer pruning
compatibility/horizon limitations remain explicit when recommending finalists.

## Commands

Local preparation (fresh external directory):

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python \
  research/experiments/flickr_dense_sweep_20261031/scripts/prepare_inputs.py \
  --cache data/flickr --output "$GNN_ARTIFACT_ROOT/flickr-dense-sweep-20261031/inputs"
```

Later collection, after the campaign ends (supply the existing approved login):

```sh
python research/experiments/flickr_dense_sweep_20261031/scripts/collect.py \
  --login "$POLUS_LOGIN" \
  --remote-root "$POLUS_TASK_ROOT/results/flickr-dense-sweep-20261031" \
  --output "$GNN_ARTIFACT_ROOT/flickr-dense-collection-NEW-TIMESTAMP"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m gnn_pruning.experiments.flickr_analysis \
  --root "$COLLECTION/raw" --protocol "$COLLECTION/raw/protocol.json" \
  --scheduler "$COLLECTION/scheduler.json" --output "$COLLECTION/analysis" --figures
```

Do not create RESULTS.csv/ANALYSIS.md as completed results before those gates.
No manuscript edit, push, merge or further experiment is part of this campaign.

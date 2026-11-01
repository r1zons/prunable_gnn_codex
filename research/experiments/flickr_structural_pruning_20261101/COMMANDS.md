# Reproducible launch and later collection

Access values must come from private external configuration. Never use the old
default SSH account blindly. `ROOT`, `SOURCE`, `TASK_ROOT`, `LOGIN` and activation
digest below are caller-supplied verified values, not instructions to change any
network route or environment. No package installation on POLUS is part of this
campaign. Snapshot/checkpoint/input hashes are required before execution.

Preparation (fresh external output, exact Git source revision):

```sh
PYTHONPATH=src "$PYTHON" research/experiments/flickr_structural_pruning_20261101/scripts/prepare.py \
  --collection "$DENSE_COLLECTION" --output "$PREPARED" \
  --source-commit "$SOURCE_COMMIT" --c4-host "$VERIFIED_C4_HOST"
git archive --format=tar.gz "$SOURCE_COMMIT" > "$PREPARED/source-$SOURCE_COMMIT.tar.gz"
```

Copy the archive, protocol, full per-file source hashes and separately generated
`smoke-jobs.json`/`research-jobs.json` (using `flickr_pruning_lsf.jobs`). Copy exact
prior masked inputs and four original Dense run directories into a fresh remote
campaign root. Verify all source hashes from inside the extracted source root;
make source/inputs/Dense copies read-only. Do not overwrite a prior root or intent.

On POLUS under the existing verified activation, with PYTHONPATH=$SOURCE/src:

```sh
python -m gnn_pruning.experiments.flickr_pruning_lsf \
  --root "$ROOT" --source "$SOURCE" --taskroot "$TASK_ROOT" \
  --protocol "$ROOT/protocol.json" --jobs "$ROOT/smoke-jobs.json" \
  --activation-sha "$ACTIVATION_SHA256"
```

After all four technical jobs are DONE and all28 output inventories/checkpoints
verify, recheck queue/user/host limits, then calculate the mandatory gate:

```sh
python "$SOURCE/research/experiments/flickr_structural_pruning_20261101/scripts/capacity.py" --root "$ROOT"
python -m gnn_pruning.experiments.flickr_pruning_lsf \
  --root "$ROOT" --source "$SOURCE" --taskroot "$TASK_ROOT" \
  --protocol "$ROOT/protocol.json" --jobs "$ROOT/research-jobs.json" \
  --activation-sha "$ACTIVATION_SHA256"
```

The gate must PASS and be less than one hour old. Every research receipt records
the exact source, protocol and launcher hashes. One bsub attempt per immutable
identity; acceptance ambiguity stops the batch. After56 verified acceptances,
write the report and stop. These are reproducibility commands, not authorization
to rerun already claimed identities.

For a later separately requested status check, use accepted IDs from the
submission manifest with `bjobs -a -w`, `bpeek JOB_ID` and `bhist -l JOB_ID`.
Job acceptance does not mean completion. After56 successful completions:

```sh
"$PYTHON" research/experiments/flickr_structural_pruning_20261101/scripts/collect.py \
  --login "$LOGIN" --remote-root "$ROOT" --output "$NEW_COLLECTION"
MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON" \
  -m gnn_pruning.experiments.flickr_pruning_analysis \
  --root "$NEW_COLLECTION/raw" --scheduler "$NEW_COLLECTION/scheduler.json" \
  --output "$NEW_COLLECTION/analysis" --figures
```

The collector checks archived LSF success, transfer SHA256 and safe extraction;
the analyzer requires all56 completed inventories, source/input/checkpoint
identities, strict reload, mask projection and validation metrics. Missing data
produce completeness evidence, not rankings. Keep raw artifacts external; only
compact verified outputs should later be versioned. No test labels are loaded.

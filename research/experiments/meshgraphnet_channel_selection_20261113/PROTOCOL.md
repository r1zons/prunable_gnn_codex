# CylinderFlow · controlled channel selection

This development experiment compares cumulative pruning by magnitude rank, E1 and mean MSE over autoregressive steps 1–5. E100 is primary. The diagnostic subset informed protocol development. The comparisons and intervals are exploratory, conditional on one frozen checkpoint and this candidate generator.

## Inputs and decisions

- Frozen Adam update 47,000 from LSF 1618479. Checkpoint SHA256 `d4a3360760f191d4b05ee80111a32067f33aea5609466b12328b4e802b2a1386`, tensor SHA256 `ba1b9d7bc024205428be3ad6f60d23e9034e21f421d00ddb0c7c5b6a03332b08`. The original weight and normalizer tensors are used in eval mode.
- Official validation TFRecord SHA256 `79171fa282fcd4fd5ff93d13e04ade43e83c1b11a0b96eee0951e4e0ed37f3ca`. IDs for checkpoint selection are 31, 7, 40, 3, 48, 96, 67, 10, 65, 35, 66, 99, 89, 71, 76, 33. Channel selection uses 58, 12, 49, 61, 82, 90, 5, 53, raw start 200. Diagnostic evaluation uses 87, 19, 72, 15, 18, 80, 79, 88, raw starts 100 and 300. Only validation records are opened.
- At each of 15 decisions, remove two original channel IDs from the second hidden layer of each of the 30 processor MLPs. Only the incoming rows and biases and outgoing columns of that hidden layer are sliced. Each model is rebuilt from the cumulative original IDs and frozen dense tensors. Focused tests compare this with successive physical slicing of the parent.

| Decision | Removed per MLP | Width | Parameters | Reduction |
|---:|---:|---:|---:|---:|
| 2 | 4 | 124 | 2,302,090 | 1.322% |
| 4 | 8 | 120 | 2,271,250 | 2.644% |
| 8 | 16 | 112 | 2,209,570 | 5.288% |
| 15 | 30 | 98 | 2,101,630 | 9.915% |

Ranks are fixed from the dense tensors using CPU float64 and the product of the L2 norm of the incoming row including bias and the L2 norm of the outgoing column. Equal norms are ordered by original ID. Each candidate library starts with the two lowest-ranked remaining channels per MLP, then four draws from the lowest eight remaining and three from all remaining. Draws sample two channels without replacement within an MLP. Duplicate full actions are retried in their own group, up to 1,000 attempts per draw. Layer order is block 0–14, edge then node. The RNG seed is the big-endian SHA256 integer of compact JSON `[101, sorted_removed_ids_per_mlp]`. Identical parents have identical candidate libraries. Divergent parents follow the same rule on their own remaining channels.

Magnitude minimizes the sum of newly removed channels' zero-based ranks. E1 and mean MSE over 1–5 use the same eight selection windows, active nodes, both velocity components and equal trajectory weights. The first frame initializes the forecast, then predictions feed back into the next step. Minimize the score and take the lexicographically smallest action in original IDs among scores within `1e-9 + 1e-4 × max(abs(score), abs(minimum))` of the exact minimum. Magnitude uses exact integer ties. Nonfinite and runtime failures are recorded per window. A candidate must have all its windows valid to compete. A strategy terminates if none qualify.

## Evaluation and resume

The runner seals decisions and physical checkpoints before loading diagnostic windows. Dense, constant initial velocity persistence and all 12 snapshots are scheduled for 16 windows each, with 100 autoregressive steps. Record full curves and first failed steps, candidate masks/scores and forward counts, timings, source and input hashes. Save snapshot tensors and verify strict reload and activation-mask equivalence. Inference timing uses the largest diagnostic mesh, five warmups, 20 independently synchronized single-step timings and allocator peaks. Report timing and rollout memory separately. Hardware speed is measured independently of parameter reduction.

The paired bootstrap samples the eight diagnostic trajectories 10,000 times with replacement, using one shared draw matrix across variants and compression levels. Both windows stay together. Per-metric aggregates and contrasts require all corresponding observations. Missing and failed windows remain visible. The analysis uses no model or dataset files. It exports per-window metrics, summary and contrasts, mask overlap, and the E100 curve over all four compression levels.

Resume in the same remote directory with the same pinned source and software. Each decision is sealed atomically after all candidates, and its original IDs and score-based winner are checked on resume. A snapshot that was interrupted is reconstructed from the saved decision. Each diagnostic window is published independently. The runner enables deterministic PyTorch operations and checks exact replay of one GPU selection candidate, reconstructed twice. Cross-host/stack numerical identity is not asserted. A continuation requires separate authorization. The initial submission script refuses an existing INTENT or submission directory.

## Resource forecast and preflight

Queue `normal`, 180-minute hard limit, 175-minute internal limit, one P100, four requested slots with exclusive host allocation, four intra-op threads, one interop thread, 8 GiB. C2–C4 are eligible. The existing POWER8 PyTorch 1.13.1/CUDA 10.2 activation and four-core affinity are reused. The pinned validation TFRecord and activation hashes are checked inside the allocation.

The pre-submission estimate uses a new local CPU probe on the verified checkpoint and archived validation trajectory `valid:0`. It measured 0.0273 s for physical projection and 1.468 s for a five-step rollout. Mask parity max absolute difference was 2.27e-6. The probe was for timing and parity only. Prior P100 job 1618575 measured at most 35.2 s for a model's 16 diagnostic windows, including setup. Charging 240 model candidates at five steps each on eight trajectories, all 224 diagnostic windows as model windows, 5 s per checkpoint, and the timing samples yields 804 s. Doubling that subtotal and adding 120 s for startup gives **1,728 s (28.8 minutes)**, or **83.5% headroom** against the internal limit. The full design is scheduled in one job. Inside the same allocation, a bounded calibration measures candidate search, strict snapshot publication, synchronized timing and a 100-step rollout before scientific decisions. If that forecast exceeds 75% of the remaining internal allocation, the worker runs a bounded prefix with a 25% reserve and leaves resumable records. No scientific settings are reduced.

Focused local tests cover nested channel identity, exact widths/counts, parity, strict reload, frozen tensors, autoregressive feedback, split isolation, deterministic libraries/ties, failed candidates and decision replay. The test command and checks are recorded with submission.

## Collection after completion

Use the approved POLUS SSH target and external artifact root. The following commands copy the results and original submission receipts, then generate a separate analysis directory.

```bash
root="$GNN_ARTIFACT_ROOT/meshgraphnet-channel-selection-20261113"
mkdir -p "$root/collected"
ssh -4 -B en0 "$POLUS_SSH_TARGET" \
  'cd "$HOME/gnn-p100-20261110/meshgraphnet-channel-selection-20261113" && tar -czf - results submission preflight.json source.sha256 source-commit.txt lsf.*' \
  > "$root/collected/run.tar.gz"
tar -xzf "$root/collected/run.tar.gz" -C "$root/collected"
python scripts/analyze_cylinderflow_channel_selection.py "$root/collected/results" "$root/analysis"
```

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

run_dense() {
  local config="$1"
  local out_dir="$2"
  echo "[presentation] Starting dense run: ${config}"
  rm -rf "$out_dir"
  python -m gnn_pruning run-dense --config "$config"
}

run_pruned() {
  local dense_ckpt="$1"
  local config="$2"
  local out_dir="$3"
  local method="$4"

  echo "[presentation] Starting prune run: ${config}"
  rm -rf "$out_dir"
  python -m gnn_pruning prune --checkpoint "$dense_ckpt" --config "$config"

  local pruned_ckpt="${out_dir}/pruned_${method}.pt"
  echo "[presentation] Starting finetune run: ${config}"
  python -m gnn_pruning finetune --checkpoint "$pruned_ckpt" --config "$config"
}

# Dense baselines first
run_dense "configs/experiments/presentation/cora_gcn_dense.yaml" "runs/presentation/cora_gcn_dense"
run_dense "configs/experiments/presentation/cora_graphsage_dense.yaml" "runs/presentation/cora_graphsage_dense"

GCN_DENSE_CKPT="runs/presentation/cora_gcn_dense/dense_checkpoint.pt"
GRAPHSAGE_DENSE_CKPT="runs/presentation/cora_graphsage_dense/dense_checkpoint.pt"

# GCN pruning experiments
run_pruned "$GCN_DENSE_CKPT" "configs/experiments/presentation/cora_gcn_random_50.yaml" "runs/presentation/cora_gcn_random_50" "random"
run_pruned "$GCN_DENSE_CKPT" "configs/experiments/presentation/cora_gcn_random_90.yaml" "runs/presentation/cora_gcn_random_90" "random"
run_pruned "$GCN_DENSE_CKPT" "configs/experiments/presentation/cora_gcn_global_magnitude_50.yaml" "runs/presentation/cora_gcn_global_magnitude_50" "global_magnitude"
run_pruned "$GCN_DENSE_CKPT" "configs/experiments/presentation/cora_gcn_global_magnitude_90.yaml" "runs/presentation/cora_gcn_global_magnitude_90" "global_magnitude"

# GraphSAGE pruning experiments
run_pruned "$GRAPHSAGE_DENSE_CKPT" "configs/experiments/presentation/cora_graphsage_random_50.yaml" "runs/presentation/cora_graphsage_random_50" "random"
run_pruned "$GRAPHSAGE_DENSE_CKPT" "configs/experiments/presentation/cora_graphsage_random_90.yaml" "runs/presentation/cora_graphsage_random_90" "random"
run_pruned "$GRAPHSAGE_DENSE_CKPT" "configs/experiments/presentation/cora_graphsage_global_magnitude_50.yaml" "runs/presentation/cora_graphsage_global_magnitude_50" "global_magnitude"
run_pruned "$GRAPHSAGE_DENSE_CKPT" "configs/experiments/presentation/cora_graphsage_global_magnitude_90.yaml" "runs/presentation/cora_graphsage_global_magnitude_90" "global_magnitude"

echo "[presentation] Benchmark runs complete."
echo "[presentation] Collect results with: python scripts/collect_presentation_results.py"

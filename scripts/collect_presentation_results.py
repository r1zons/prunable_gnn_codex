"""Collect presentation benchmark metrics into one CSV."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _get_value(metrics: Dict[str, Any], key: str, default: float = 0.0) -> float:
    return float(metrics.get("benchmark", {}).get(key, default))


def _collect_dense(run_dir: Path, model: str) -> Dict[str, Any]:
    metrics = _load_json(run_dir / "metrics_eval.json")
    checkpoint_path = run_dir / "dense_checkpoint.pt"
    checkpoint_size_bytes = float(os.path.getsize(checkpoint_path)) if checkpoint_path.exists() else 0.0
    return {
        "model": model,
        "method": "dense",
        "sparsity": 0.0,
        "test_accuracy": float(metrics["test"]["accuracy"]),
        "test_macro_f1": float(metrics["test"]["macro_f1"]),
        "inference_time_mean_sec": _get_value(metrics, "inference_time_mean_sec"),
        "parameter_count": _get_value(metrics, "parameter_count"),
        "checkpoint_size_bytes": checkpoint_size_bytes,
        "run_dir": str(run_dir),
    }


def _collect_pruned(run_dir: Path, model: str, method: str, sparsity: float) -> Dict[str, Any]:
    metrics_path = run_dir / "metrics_pruned_post_finetune.json"
    if not metrics_path.exists():
        metrics_path = run_dir / f"metrics_post_prune_{method}.json"

    metrics = _load_json(metrics_path)
    post_ckpt = run_dir / f"pruned_{method}_post_finetune.pt"
    checkpoint_size_bytes = float(os.path.getsize(post_ckpt)) if post_ckpt.exists() else 0.0

    return {
        "model": model,
        "method": method,
        "sparsity": sparsity,
        "test_accuracy": float(metrics["test"]["accuracy"]),
        "test_macro_f1": float(metrics["test"]["macro_f1"]),
        "inference_time_mean_sec": _get_value(metrics, "inference_time_mean_sec"),
        "parameter_count": _get_value(metrics, "parameter_count"),
        "checkpoint_size_bytes": checkpoint_size_bytes,
        "run_dir": str(run_dir),
    }


def build_rows(runs_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    rows.append(_collect_dense(runs_root / "cora_gcn_dense", "gcn"))
    rows.append(_collect_dense(runs_root / "cora_graphsage_dense", "graphsage"))

    specs = [
        ("gcn", "random", 0.5),
        ("gcn", "random", 0.9),
        ("gcn", "global_magnitude", 0.5),
        ("gcn", "global_magnitude", 0.9),
        ("graphsage", "random", 0.5),
        ("graphsage", "random", 0.9),
        ("graphsage", "global_magnitude", 0.5),
        ("graphsage", "global_magnitude", 0.9),
    ]

    for model, method, sparsity in specs:
        label = "50" if sparsity == 0.5 else "90"
        run_dir = runs_root / f"cora_{model}_{method}_{label}"
        rows.append(_collect_pruned(run_dir, model, method, sparsity))

    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect presentation benchmark metrics into CSV.")
    parser.add_argument("--runs-root", default="runs/presentation", help="Root directory containing presentation run folders.")
    parser.add_argument("--output", default="runs/presentation/presentation_results.csv", help="Output CSV path.")
    args = parser.parse_args()

    runs_root = Path(args.runs_root).expanduser()
    output_path = Path(args.output).expanduser()

    rows = build_rows(runs_root)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "method",
                "sparsity",
                "test_accuracy",
                "test_macro_f1",
                "inference_time_mean_sec",
                "parameter_count",
                "checkpoint_size_bytes",
                "run_dir",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[presentation] Wrote {len(rows)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

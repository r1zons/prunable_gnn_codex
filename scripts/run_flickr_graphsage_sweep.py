"""Run Flickr GraphSAGE architecture sweep and merge key metrics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gnn_pruning.config import load_yaml
from gnn_pruning.pipelines import run_pipeline


SWEEP_CONFIGS = [
    "configs/experiments/flickr_graphsage_l2_h64.yaml",
    "configs/experiments/flickr_graphsage_l2_h128.yaml",
    "configs/experiments/flickr_graphsage_l3_h64.yaml",
    "configs/experiments/flickr_graphsage_l3_h128.yaml",
    "configs/experiments/flickr_graphsage_l4_h128.yaml",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Flickr GraphSAGE architecture sweep.")
    parser.add_argument("--output-root", default="runs/flickr_graphsage_sweep")
    args = parser.parse_args()

    output_root = Path(args.output_root).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    merged_rows: List[Dict[str, Any]] = []
    for config_path in SWEEP_CONFIGS:
        cfg = load_yaml(config_path)
        model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
        print(f"[run_flickr_graphsage_sweep] running {config_path}")
        artifacts = run_pipeline(config_path)
        rows = _read_pipeline_rows(artifacts.csv_path)
        for row in rows:
            checkpoint_path = Path(str(row.get("checkpoint_path", "")))
            merged_rows.append(
                {
                    "dataset": row.get("dataset", ""),
                    "model": row.get("model", ""),
                    "num_layers": model_cfg.get("num_layers", ""),
                    "hidden_channels": model_cfg.get("hidden_channels", ""),
                    "method": row.get("pruning_method", ""),
                    "phase": row.get("phase", ""),
                    "sparsity": row.get("requested_sparsity", ""),
                    "test_accuracy": row.get("test_accuracy", ""),
                    "test_macro_f1": row.get("test_macro_f1", ""),
                    "inference_time_mean_sec": _ms_to_sec(row.get("inference_time_mean_ms", "")),
                    "parameter_count": row.get("parameter_count", ""),
                    "checkpoint_size_bytes": _checkpoint_size(checkpoint_path),
                    "run_dir": str(artifacts.output_dir),
                }
            )

    merged_csv = output_root / "sweep_results.csv"
    _write_rows(merged_rows, merged_csv)
    print(f"[run_flickr_graphsage_sweep] saved {merged_csv}")
    _write_best_summary(merged_rows, output_root / "best_summary.json")
    return 0


def _read_pipeline_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _ms_to_sec(value: Any) -> Any:
    if value in ("", None):
        return ""
    return float(value) / 1000.0


def _checkpoint_size(path: Path) -> int:
    if not path.exists():
        return 0
    return int(path.stat().st_size)


def _write_rows(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_best_summary(rows: List[Dict[str, Any]], path: Path) -> None:
    post_finetune = [row for row in rows if row.get("phase") == "post_finetune"]
    if not post_finetune:
        return
    best = max(post_finetune, key=lambda row: float(row.get("test_accuracy", 0.0) or 0.0))
    payload = {"best_post_finetune_accuracy": best}
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    raise SystemExit(main())

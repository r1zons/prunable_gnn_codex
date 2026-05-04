"""Run Flickr pruning study configs and merge results."""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

DEBUG_CONFIGS = [
    "configs/experiments/flickr_debug_graphsage_l2_h64.yaml",
    "configs/experiments/flickr_debug_graphsage_l2_h128.yaml",
    "configs/experiments/flickr_debug_graphsage_l3_h128.yaml",
]
REAL_CONFIGS = [
    "configs/experiments/flickr_graphsage_l2_h64.yaml",
    "configs/experiments/flickr_graphsage_l2_h128.yaml",
    "configs/experiments/flickr_graphsage_l3_h128.yaml",
    "configs/experiments/flickr_graphsage_l4_h128.yaml",
]
POST_PRUNE_ONLY_CONFIGS = [
    "configs/experiments/flickr_postprune_only_graphsage_l2_h128.yaml",
    "configs/experiments/flickr_postprune_only_graphsage_l3_h128.yaml",
]


def _run_config(config: str) -> Path:
    cmd = ["python", "-u", "-m", "gnn_pruning", "run-pipeline", "--config", config, "--progress"]
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    run_dir = Path("runs")
    # Resolve from config file output_dir to avoid assumptions.
    import yaml
    with Path(config).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return Path(payload["run"]["output_dir"])


def _merge_csvs(run_dirs: list[Path], merged_csv: Path) -> None:
    merged_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for run_dir in run_dirs:
        csv_path = run_dir / "pipeline_results.csv"
        if not csv_path.exists():
            continue
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            rows.extend(list(csv.DictReader(handle)))

    if not rows:
        print("[warn] no rows to merge")
        return

    fields = [
        "dataset", "model", "num_layers", "hidden_channels", "method", "phase", "sparsity", "achieved_sparsity",
        "test_accuracy", "test_macro_f1", "inference_time_mean_ms", "inference_time_std_ms", "parameter_count",
        "final_reward", "num_adaptive_steps", "stop_reason", "run_dir",
    ]
    with merged_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--post-prune-only", action="store_true")
    parser.add_argument("--configs", nargs="+", default=[])
    args = parser.parse_args()

    selected = []
    if args.debug:
        selected.extend(DEBUG_CONFIGS)
    if args.real:
        selected.extend(REAL_CONFIGS)
    if args.post_prune_only:
        selected.extend(POST_PRUNE_ONLY_CONFIGS)
    selected.extend(args.configs)
    if not selected:
        parser.error("Select at least one mode: --debug, --real, --post-prune-only, or --configs")

    run_dirs = []
    for cfg in selected:
        run_dirs.append(_run_config(cfg))

    merged_csv = Path("runs/flickr_pruning_study/flickr_results.csv")
    _merge_csvs(run_dirs, merged_csv)
    print(f"[done] merged results: {merged_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

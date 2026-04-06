"""Run pruning comparison pipelines on larger datasets (Flickr + Reddit)."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gnn_pruning.config import dump_yaml, load_yaml
from gnn_pruning.pipelines import run_pipeline


EXPERIMENT_CONFIGS = [
    "configs/experiments/presentation_flickr.yaml",
    "configs/experiments/presentation_reddit.yaml",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run large-dataset comparison experiments.")
    parser.add_argument("--output-root", default="runs/large_datasets")
    args = parser.parse_args()

    output_root = Path(args.output_root).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    merged_rows: List[Dict[str, Any]] = []
    for config_path in EXPERIMENT_CONFIGS:
        cfg = _prepare_config(config_path=config_path, output_root=output_root)
        print(f"[run_large_datasets] running {config_path} -> {cfg}")
        artifacts = run_pipeline(str(cfg))
        print(f"[run_large_datasets] complete: {artifacts.output_dir}")
        merged_rows.extend(_read_csv_rows(artifacts.csv_path))

    merged_csv = output_root / "large_dataset_results.csv"
    _write_merged_rows(merged_rows, merged_csv)
    print(f"[run_large_datasets] merged CSV: {merged_csv}")
    return 0


def _prepare_config(config_path: str, output_root: Path) -> Path:
    source = Path(config_path).expanduser()
    payload = load_yaml(source)
    payload["run"] = dict(payload.get("run", {}))
    payload["run"]["output_dir"] = str(output_root / source.stem)
    payload["run"]["experiment_name"] = str(payload["run"].get("experiment_name", source.stem))
    generated = output_root / f"{source.stem}.yaml"
    dump_yaml(payload, generated)
    return generated


def _read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_merged_rows(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    header = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())

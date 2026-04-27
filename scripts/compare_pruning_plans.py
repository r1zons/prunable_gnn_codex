"""Compare pruning plans for global vs layerwise magnitude on one config."""

import argparse
import json
import tempfile
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gnn_pruning.config import dump_yaml, load_yaml, resolve_config
from gnn_pruning.pruning.workflow import prune_from_checkpoint
from gnn_pruning.training.workflow import train_dense


def _variant_config(base_config_path: str, method: str, sparsity: float, output_dir: Path) -> Path:
    cfg = load_yaml(base_config_path)
    cfg["run"] = dict(cfg.get("run", {}))
    cfg["run"]["output_dir"] = str(output_dir)
    cfg["pruning"] = dict(cfg.get("pruning", {}))
    cfg["pruning"]["method"] = method
    cfg["pruning"]["target_sparsity"] = float(sparsity)
    cfg["pruning"]["structured"] = True
    path = output_dir / f"compare_{method}.yaml"
    dump_yaml(cfg, path)
    return path


def _load_plan(pruning_metrics_path: Path) -> Dict[str, Any]:
    with pruning_metrics_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare global_magnitude and layerwise_magnitude pruning plans.")
    parser.add_argument("--config", required=True, help="Experiment config path.")
    parser.add_argument("--sparsity", type=float, default=0.5, help="Target sparsity for structured pruning.")
    args = parser.parse_args()

    resolved = resolve_config(args.config)
    run_dir = Path(resolved.run.output_dir).expanduser()
    run_dir.mkdir(parents=True, exist_ok=True)
    dense_checkpoint = run_dir / "dense_checkpoint.pt"

    if not dense_checkpoint.exists():
        print(f"[compare_pruning_plans] dense checkpoint missing, training once at {run_dir}")
        artifacts = train_dense(args.config, resume=True, output_dir_override=str(run_dir))
        dense_checkpoint = artifacts.checkpoint_path

    plans: Dict[str, Dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix="compare_plans_") as tmp:
        tmp_root = Path(tmp)
        for method in ("global_magnitude", "layerwise_magnitude"):
            method_dir = tmp_root / method
            method_dir.mkdir(parents=True, exist_ok=True)
            cfg = _variant_config(args.config, method=method, sparsity=args.sparsity, output_dir=method_dir)
            artifacts = prune_from_checkpoint(str(dense_checkpoint), str(cfg))
            plans[method] = _load_plan(artifacts.pruning_metrics_path)

    print("=== pruning plan comparison ===")
    for method, plan in plans.items():
        details = plan.get("details", {}) if isinstance(plan.get("details", {}), dict) else {}
        print(f"method={method}")
        print(f"  requested_sparsity={plan.get('requested_sparsity')}")
        print(f"  achieved_sparsity={plan.get('achieved_sparsity')}")
        print(f"  selected_layer_indices={details.get('selected_layer_indices', [details.get('layer_index', 0)])}")
        print(f"  kept_channel_indices={details.get('kept_channel_indices', [])}")

    left = plans["global_magnitude"].get("details", {})
    right = plans["layerwise_magnitude"].get("details", {})
    if left == right:
        groups = int(left.get("prunable_channel_groups", 0) or 0)
        print(f"plans are identical; prunable_channel_groups={groups}")
    else:
        print("plans differ between global_magnitude and layerwise_magnitude")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

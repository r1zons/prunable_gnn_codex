"""Run dense once and compare random vs global_magnitude pruning diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gnn_pruning.pruning.workflow import finetune_pruned_checkpoint, prune_from_checkpoint
from gnn_pruning.training.workflow import train_dense


def _write_config(path: Path, dataset: str, model: str, method: str, sparsity: float, output_dir: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "base: base/default",
                f"dataset: {dataset}",
                f"model: {model}",
                "preset: fast_debug",
                "run:",
                f"  experiment_name: debug_compare_{dataset}_{model}_{method}_{int(sparsity*100)}",
                f"  output_dir: {output_dir.as_posix()}",
                "pruning:",
                f"  method: {method}",
                f"  target_sparsity: {sparsity}",
                "  structured: true",
                "  finetune_epochs: 5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _phase_acc(metrics: dict, phase: str) -> float:
    return float(metrics.get(phase, {}).get("test", {}).get("accuracy", float("nan")))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="pubmed")
    parser.add_argument("--model", default="graphsage")
    parser.add_argument("--sparsity", type=float, default=0.9)
    parser.add_argument("--output-root", default="runs/debug_compare_pruners")
    args = parser.parse_args()

    root = Path(args.output_root).expanduser()
    root.mkdir(parents=True, exist_ok=True)

    dense_cfg = _write_config(root / "dense.yaml", args.dataset, args.model, "random", args.sparsity, root / "dense")
    # dense config should not include pruning section, but train_dense ignores it.
    dense_artifacts = train_dense(str(dense_cfg), resume=False)

    for method in ("random", "global_magnitude"):
        run_dir = root / method
        cfg = _write_config(root / f"{method}.yaml", args.dataset, args.model, method, args.sparsity, run_dir)
        prune_artifacts = prune_from_checkpoint(str(dense_artifacts.checkpoint_path), str(cfg))
        finetune_pruned_checkpoint(str(prune_artifacts.pruned_checkpoint_path), str(cfg), finetune_epochs=5)

        diagnostics = json.loads((run_dir / "pruning_diagnostics.json").read_text(encoding="utf-8"))
        print(f"\n=== {method} ===")
        print("kept channels:", diagnostics.get("kept_channel_indices_per_layer"))
        print("hidden dims before/after:", diagnostics.get("dense_hidden_dimensions"), diagnostics.get("pruned_hidden_dimensions"))
        print("params before/after:", diagnostics.get("dense_parameter_count"), diagnostics.get("pruned_parameter_count"))
        print(
            "accuracy dense/post_prune/post_finetune:",
            _phase_acc(diagnostics["metrics"], "dense"),
            _phase_acc(diagnostics["metrics"], "post_prune"),
            _phase_acc(diagnostics["metrics"], "post_finetune"),
        )

    print(f"\nDiagnostics saved under: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

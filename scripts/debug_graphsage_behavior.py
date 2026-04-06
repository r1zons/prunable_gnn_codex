"""Run GraphSAGE diagnostics across cora/citeseer/pubmed at sparsity 0.9."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gnn_pruning.pruning.workflow import finetune_pruned_checkpoint, prune_from_checkpoint
from gnn_pruning.training.workflow import train_dense


def _write_config(path: Path, dataset: str, output_dir: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "base: base/default",
                f"dataset: {dataset}",
                "model: graphsage",
                "preset: fast_debug",
                "run:",
                f"  experiment_name: debug_graphsage_{dataset}",
                f"  output_dir: {output_dir.as_posix()}",
                "pruning:",
                "  method: global_magnitude",
                "  target_sparsity: 0.9",
                "  structured: true",
                "  finetune_epochs: 5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="runs/debug_graphsage_behavior")
    args = parser.parse_args()

    root = Path(args.output_root).expanduser()
    root.mkdir(parents=True, exist_ok=True)

    for dataset in ("cora", "citeseer", "pubmed"):
        run_dir = root / dataset
        cfg = _write_config(root / f"{dataset}.yaml", dataset, run_dir)
        dense_artifacts = train_dense(str(cfg), resume=False)
        prune_artifacts = prune_from_checkpoint(str(dense_artifacts.checkpoint_path), str(cfg))
        finetune_pruned_checkpoint(str(prune_artifacts.pruned_checkpoint_path), str(cfg))
        diagnostics = json.loads((run_dir / "pruning_diagnostics.json").read_text(encoding="utf-8"))

        print(f"\n=== GraphSAGE {dataset} @ 0.9 ===")
        print("dense accuracy:", diagnostics["metrics"]["dense"]["test"]["accuracy"])
        print("post_prune accuracy:", diagnostics["metrics"]["post_prune"]["test"]["accuracy"])
        print("post_finetune accuracy:", diagnostics["metrics"]["post_finetune"]["test"]["accuracy"])
        print("param counts before/after:", diagnostics["dense_parameter_count"], diagnostics["pruned_parameter_count"])
        print("hidden dims before/after:", diagnostics["dense_hidden_dimensions"], diagnostics["pruned_hidden_dimensions"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

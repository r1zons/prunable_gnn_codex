"""Run dense + prune + finetune across Cora/CiteSeer/PubMed."""

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
                f"  experiment_name: multidataset_{dataset}_{model}_{method}_{int(sparsity*100)}",
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="runs/multidataset")
    args = parser.parse_args()

    datasets = ["cora", "citeseer", "pubmed"]
    models = ["gcn", "graphsage"]
    methods = ["random", "global_magnitude"]
    sparsities = [0.5, 0.9]

    root = Path(args.output_root).expanduser()
    root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, list[dict]] = {}
    for dataset in datasets:
        summary[dataset] = []
        for model in models:
            dense_cfg = _write_config(root / f"{dataset}_{model}_dense.yaml", dataset, model, "random", 0.5, root / dataset / model / "dense")
            dense_artifacts = train_dense(str(dense_cfg), resume=False)

            for method in methods:
                for sparsity in sparsities:
                    run_dir = root / dataset / model / f"{method}_{int(sparsity*100)}"
                    cfg = _write_config(root / f"{dataset}_{model}_{method}_{int(sparsity*100)}.yaml", dataset, model, method, sparsity, run_dir)
                    prune_artifacts = prune_from_checkpoint(str(dense_artifacts.checkpoint_path), str(cfg))
                    finetune_pruned_checkpoint(str(prune_artifacts.pruned_checkpoint_path), str(cfg))
                    diagnostics = json.loads((run_dir / "pruning_diagnostics.json").read_text(encoding="utf-8"))
                    summary[dataset].append(
                        {
                            "model": model,
                            "method": method,
                            "sparsity": sparsity,
                            "dense_acc": diagnostics["metrics"]["dense"]["test"]["accuracy"],
                            "post_prune_acc": diagnostics["metrics"]["post_prune"]["test"]["accuracy"],
                            "post_finetune_acc": diagnostics["metrics"]["post_finetune"]["test"]["accuracy"],
                        }
                    )

        print(f"\n=== Dataset: {dataset} ===")
        for row in summary[dataset]:
            print(row)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Debug Flickr GraphSAGE sweep checkpoint/architecture consistency."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gnn_pruning.config import resolve_config
from gnn_pruning.models import build_model


CONFIGS = [
    "configs/experiments/flickr_graphsage_l2_h64.yaml",
    "configs/experiments/flickr_graphsage_l2_h128.yaml",
    "configs/experiments/flickr_graphsage_l3_h64.yaml",
    "configs/experiments/flickr_graphsage_l3_h128.yaml",
    "configs/experiments/flickr_graphsage_l4_h128.yaml",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-run-metadata", action="store_true")
    args = parser.parse_args()

    dense_paths_by_arch: dict[tuple[int, int], str] = {}
    for config_path in CONFIGS:
        resolved = resolve_config(config_path)
        model = build_model(
            resolved.model.name,
            in_channels=resolved.model.in_channels or 16,
            hidden_channels=resolved.model.hidden_channels,
            out_channels=resolved.model.out_channels or 7,
            num_layers=resolved.model.num_layers,
            dropout=resolved.model.dropout,
        )
        params = sum(p.numel() for p in model.parameters())
        run_dir = Path(resolved.run.output_dir).expanduser()
        dense_path = str(run_dir / "dense_checkpoint.pt")
        arch_key = (int(resolved.model.num_layers), int(resolved.model.hidden_channels))

        reused = False
        if args.check_run_metadata:
            meta_path = run_dir / "run_metadata.json"
            if meta_path.exists():
                reused = "checkpoint_reused\": true" in meta_path.read_text(encoding="utf-8")

        print(
            f"config={config_path} dataset={resolved.data.name} model={resolved.model.name} "
            f"num_layers={resolved.model.num_layers} hidden={resolved.model.hidden_channels} "
            f"param_count={params} run_dir={run_dir} dense_checkpoint={dense_path} reused={reused}"
        )

        if arch_key in dense_paths_by_arch and dense_paths_by_arch[arch_key] != dense_path:
            raise SystemExit(f"Architecture {arch_key} has inconsistent dense checkpoint paths.")
        dense_paths_by_arch[arch_key] = dense_path

    all_dense_paths = list(dense_paths_by_arch.values())
    if len(set(all_dense_paths)) != len(all_dense_paths):
        raise SystemExit("Different Flickr GraphSAGE architecture configs reuse the same dense checkpoint path.")
    print("Consistency check passed: architecture variants use distinct dense checkpoint paths.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Inspect resolved GraphSAGE configs and built model architectures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gnn_pruning.config import resolve_config
from gnn_pruning.data import load_dataset
from gnn_pruning.models import build_model


def _load_first_graph(config_path: str):
    resolved = resolve_config(config_path)
    if str(resolved.data.name).strip().lower() == "dblp":
        dataset = load_dataset(
            resolved.data.name,
            resolved.data.root,
            getattr(resolved.data, "dblp_strategy", "author_homogeneous"),
        )
    else:
        dataset = load_dataset(resolved.data.name, resolved.data.root)
    return resolved, dataset[0]


def _parameter_count(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def main() -> int:
    parser = argparse.ArgumentParser(description="Debug GraphSAGE config-to-model application.")
    parser.add_argument("config_paths", nargs="+", help="One or more experiment config paths.")
    args = parser.parse_args()

    for config_path in args.config_paths:
        resolved, data = _load_first_graph(config_path)
        model = build_model(
            resolved.model.name,
            in_channels=int(data.num_features),
            hidden_channels=int(resolved.model.hidden_channels),
            out_channels=int(data.y.max().item()) + 1,
            num_layers=int(resolved.model.num_layers),
            dropout=float(resolved.model.dropout),
        )

        print("=" * 80)
        print(f"config_path: {config_path}")
        print(f"dataset: {resolved.data.name}")
        print(f"model: {resolved.model.name}")
        print(f"num_layers: {resolved.model.num_layers}")
        print(f"hidden_channels: {resolved.model.hidden_channels}")
        print(f"dropout: {resolved.model.dropout}")
        print(f"parameter_count: {_parameter_count(model)}")
        print("model_summary:")
        print(model)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

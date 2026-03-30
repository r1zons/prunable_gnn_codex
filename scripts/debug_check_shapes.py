"""Verify structured pruning changes architecture and parameter counts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gnn_pruning.models import build_model


def _param_count(state_dict: dict) -> int:
    return int(sum(t.numel() for t in state_dict.values()))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-checkpoint", required=True)
    parser.add_argument("--pruned-checkpoint", required=True)
    args = parser.parse_args()

    dense_payload = torch.load(Path(args.dense_checkpoint).expanduser(), map_location="cpu")
    pruned_payload = torch.load(Path(args.pruned_checkpoint).expanduser(), map_location="cpu")

    dense_model = build_model(dense_payload["model_name"], **dense_payload["model_config"])
    pruned_model = build_model(pruned_payload["model_name"], **pruned_payload["model_config"])

    print("=== Dense model ===")
    print(dense_model)
    print("param_count:", _param_count(dense_payload["model_state_dict"]))

    print("\n=== Pruned model ===")
    print(pruned_model)
    print("param_count:", _param_count(pruned_payload["model_state_dict"]))

    if hasattr(dense_model, "convs") and hasattr(pruned_model, "convs"):
        dense_dims = [int(c.out_channels) for c in dense_model.convs[:-1]]
        pruned_dims = [int(c.out_channels) for c in pruned_model.convs[:-1]]
        print("hidden_dims_before:", dense_dims)
        print("hidden_dims_after:", pruned_dims)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

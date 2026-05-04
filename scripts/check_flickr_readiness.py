"""Quick readiness checks for Flickr pruning experiments."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch

from gnn_pruning.config import resolve_config
from gnn_pruning.data import generate_exact_ratio_split, load_dataset
from gnn_pruning.models import build_model
from gnn_pruning.pruning import list_pruners
from gnn_pruning.training.workflow import _build_run_signature


def _param_count(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def _generate_split(cfg: object, num_nodes: int):
    return generate_exact_ratio_split(
        num_nodes=num_nodes,
        seed=int(cfg.run.seed),
        train_ratio=float(cfg.data.train_ratio),
        val_ratio=float(cfg.data.val_ratio),
        test_ratio=float(cfg.data.test_ratio),
    )


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    cfg = resolve_config("configs/experiments/flickr_graphsage_l2_h64.yaml")
    dataset = load_dataset("flickr", cfg.data.root)
    data = dataset[0]
    num_classes = int(data.y.max().item() + 1) if getattr(data, "y", None) is not None and data.y.numel() else 0
    print("dataset loaded: flickr")
    print(f"nodes={data.num_nodes} edges={data.edge_index.size(1)} features={data.num_features} classes={num_classes}")

    split = _generate_split(cfg=cfg, num_nodes=int(data.num_nodes))
    print(f"split sizes: train={len(split.train)} val={len(split.val)} test={len(split.test)}")

    model_cfgs = [(2, 64), (2, 128), (3, 128), (4, 128)]
    counts: dict[tuple[int, int], int] = {}
    for layers, hidden in model_cfgs:
        model = build_model("graphsage", in_channels=data.num_features, hidden_channels=hidden, out_channels=num_classes, num_layers=layers, dropout=0.5)
        model = model.to(device)
        counts[(layers, hidden)] = _param_count(model)
        with torch.no_grad():
            out = model(data.to(device))
        assert out.shape[0] == data.num_nodes and out.shape[1] == num_classes
    print("tested model configs:", model_cfgs)
    print("parameter counts:", counts)

    dense_signatures = []
    for layers, hidden in model_cfgs:
        run_cfg = resolve_config("configs/experiments/flickr_graphsage_l2_h64.yaml")
        run_cfg.model.num_layers = layers
        run_cfg.model.hidden_channels = hidden
        signature = _build_run_signature(run_cfg, split_hash="dummy_split_hash")
        dense_signatures.append((layers, hidden, signature["config_hash"]))
    print("dense checkpoint signatures:", dense_signatures)
    assert len({row[2] for row in dense_signatures}) == len(dense_signatures)

    pruners = [row.get("name", "") for row in list_pruners()]
    print("registered pruners:", pruners)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

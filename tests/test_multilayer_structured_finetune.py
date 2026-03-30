"""Regression test for multi-layer structured GraphSAGE prune+finetune path."""

from __future__ import annotations

import importlib
from pathlib import Path

import torch
from torch_geometric.data import Data

from gnn_pruning.models import GraphSAGENodeClassifier
from gnn_pruning.pruning.workflow import finetune_pruned_checkpoint, prune_from_checkpoint


def _dummy_dataset():
    n = 40
    x = torch.randn((n, 8), dtype=torch.float32)
    y = torch.randint(0, 4, (n,), dtype=torch.long)
    edge_index = torch.vstack([torch.arange(n), torch.roll(torch.arange(n), shifts=-1)]).long()
    data = Data(x=x, y=y, edge_index=edge_index)

    class DummyDataset:
        def __getitem__(self, idx: int):
            _ = idx
            return data

    return DummyDataset()


def _write_config(path: Path, output_dir: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "base: base/default",
                "dataset: flickr",
                "model: graphsage",
                "preset: fast_debug",
                "model:",
                "  name: graphsage",
                "  num_layers: 3",
                "  hidden_channels: 64",
                "run:",
                f"  output_dir: {output_dir.as_posix()}",
                "pruning:",
                "  method: global_magnitude",
                "  target_sparsity: 0.5",
                "  structured: true",
                "  finetune_epochs: 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _make_checkpoint(path: Path) -> Path:
    model = GraphSAGENodeClassifier(in_channels=8, hidden_channels=64, out_channels=4, num_layers=3, dropout=0.0)
    torch.save(
        {
            "model_name": "graphsage",
            "model_config": {
                "in_channels": 8,
                "hidden_channels": 64,
                "out_channels": 4,
                "num_layers": 3,
                "dropout": 0.0,
            },
            "model_state_dict": model.state_dict(),
        },
        path,
    )
    return path


def test_structured_graphsage_l3_prune_then_finetune(monkeypatch, tmp_path: Path) -> None:
    pruning_module = importlib.import_module("gnn_pruning.pruning.workflow")
    training_module = importlib.import_module("gnn_pruning.training.workflow")
    monkeypatch.setattr(pruning_module, "load_dataset", lambda name, root: _dummy_dataset())
    monkeypatch.setattr(training_module, "load_dataset", lambda name, root: _dummy_dataset())

    cfg = _write_config(tmp_path / "cfg.yaml", tmp_path / "out")
    dense = _make_checkpoint(tmp_path / "dense.pt")

    prune_artifacts = prune_from_checkpoint(str(dense), str(cfg))
    finetune_artifacts = finetune_pruned_checkpoint(str(prune_artifacts.pruned_checkpoint_path), str(cfg), finetune_epochs=1)
    assert finetune_artifacts.post_finetune_checkpoint_path.exists()

"""Lightweight checks for adaptive experiment files and smoke behavior."""

from __future__ import annotations

import importlib
from pathlib import Path

import torch
from torch_geometric.data import Data

from gnn_pruning.config import resolve_config
from gnn_pruning.pipelines import run_pipeline
from gnn_pruning.pruning import get_pruner


def _dummy_dataset(num_nodes: int = 24, in_channels: int = 6, num_classes: int = 3):
    x = torch.randn((num_nodes, in_channels), dtype=torch.float32)
    y = torch.randint(0, num_classes, (num_nodes,), dtype=torch.long)
    edge_index = torch.vstack(
        [
            torch.arange(0, num_nodes, dtype=torch.long),
            torch.roll(torch.arange(0, num_nodes, dtype=torch.long), shifts=-1),
        ]
    )
    data = Data(x=x, y=y, edge_index=edge_index)

    class DummyDataset:
        def __getitem__(self, idx: int):
            _ = idx
            return data

    return DummyDataset()


def test_adaptive_config_exists_and_resolves() -> None:
    config_path = Path("configs/experiments/adaptive_graphsage_citeseer_debug.yaml")
    assert config_path.exists()
    resolved = resolve_config(config_path)
    assert resolved.data.name == "citeseer"
    assert resolved.model.name == "graphsage"


def test_adaptive_pruner_registered() -> None:
    pruner_cls = get_pruner("adaptive_layerwise")
    assert pruner_cls.name == "adaptive_layerwise"


def test_adaptive_debug_pipeline_smoke(monkeypatch, tmp_path: Path) -> None:
    training_module = importlib.import_module("gnn_pruning.training.workflow")
    pruning_module = importlib.import_module("gnn_pruning.pruning.workflow")
    dataset = _dummy_dataset()
    monkeypatch.setattr(training_module, "load_dataset", lambda name, root: dataset)
    monkeypatch.setattr(pruning_module, "load_dataset", lambda name, root: dataset)

    cfg = tmp_path / "adaptive_debug.yaml"
    cfg.write_text(
        "\n".join(
            [
                "base: base/default",
                "dataset: citeseer",
                "model: graphsage",
                "preset: fast_debug",
                "run:",
                f"  output_dir: {str((tmp_path / 'out').as_posix())}",
                "pruning:",
                "  methods: [adaptive_layerwise]",
                "  sparsity_levels: [0.5]",
                "  structured: true",
                "  finetune_epochs: 1",
                "adaptive_pruning:",
                "  step_prune_ratio: 0.25",
                "  max_steps: 4",
                "  max_accuracy_drop: 0.3",
                "  min_channels_per_layer: 4",
                "  reward:",
                "    alpha: 0.4",
                "    beta: 0.2",
                "    gamma: 0.4",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    artifacts = run_pipeline(str(cfg))
    assert artifacts.csv_path.exists()
    trace_path = artifacts.output_dir / "adaptive_layerwise_s0_5000" / "adaptive_trace.json"
    assert trace_path.exists()


def test_adaptive_comparison_config_resolves() -> None:
    suite_path = Path("configs/suites/adaptive_comparison_small.yaml")
    assert suite_path.exists()

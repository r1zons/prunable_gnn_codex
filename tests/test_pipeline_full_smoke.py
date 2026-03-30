"""Smoke test for full pruning benchmark pipeline."""

from __future__ import annotations

import csv
import importlib
from pathlib import Path

import torch
from torch_geometric.data import Data

from gnn_pruning.pipelines import run_pipeline


def _build_dummy_dataset(num_nodes: int = 36, in_channels: int = 10, num_classes: int = 4):
    x = torch.randn((num_nodes, in_channels), dtype=torch.float32)
    y = torch.randint(0, num_classes, (num_nodes,), dtype=torch.long)
    edge_index = torch.vstack([torch.arange(num_nodes), torch.roll(torch.arange(num_nodes), shifts=-1)]).long()
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
                "dataset: citeseer",
                "model: graphsage",
                "preset: fast_debug",
                "run:",
                "  seed: 42",
                f"  output_dir: {output_dir.as_posix()}",
                "pruning:",
                "  methods: [random]",
                "  sparsity_levels: [0.1, 0.3, 0.5, 0.7, 0.9]",
                "  structured: true",
                "  finetune_epochs: 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_run_pipeline_full_smoke_citeseer_graphsage_fast_debug(monkeypatch, tmp_path: Path) -> None:
    training_module = importlib.import_module("gnn_pruning.training.workflow")
    pruning_module = importlib.import_module("gnn_pruning.pruning.workflow")
    dataset = _build_dummy_dataset()
    monkeypatch.setattr(training_module, "load_dataset", lambda name, root: dataset)
    monkeypatch.setattr(pruning_module, "load_dataset", lambda name, root: dataset)

    cfg = _write_config(tmp_path / "pipeline.yaml", tmp_path / "pipeline_out")
    artifacts = run_pipeline(str(cfg))

    assert artifacts.config_snapshot.exists()
    assert artifacts.split_artifact.exists()
    assert artifacts.dense_checkpoint_path.exists()
    assert artifacts.dense_metrics_path.exists()
    assert artifacts.csv_path.exists()
    assert artifacts.summary_path.exists()

    with artifacts.csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    phases = [row["phase"] for row in rows]
    assert phases.count("dense") == 1
    assert phases.count("post_prune") == 5
    assert phases.count("post_finetune") == 5

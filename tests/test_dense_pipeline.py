"""Tests for full dense pipeline orchestration and CSV reporting."""

from __future__ import annotations

import importlib
from pathlib import Path

import torch
from torch_geometric.data import Data

from gnn_pruning.pipelines import run_dense_pipeline
from gnn_pruning.reporting import csv_columns


def _build_dummy_dataset(num_nodes: int = 12, in_channels: int = 5, num_classes: int = 3):
    x = torch.randn((num_nodes, in_channels), dtype=torch.float32)
    y = torch.randint(0, num_classes, (num_nodes,), dtype=torch.long)
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0],
        ],
        dtype=torch.long,
    )

    data = Data(x=x, y=y, edge_index=edge_index)

    class DummyDataset:
        def __getitem__(self, idx: int):
            _ = idx
            return data

    return DummyDataset()


def _write_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "dense_pipeline.yaml"
    config_path.write_text(
        "\n".join(
            [
                "base: base/default",
                "dataset: citeseer",
                "model: gcn",
                "preset: fast_debug",
                "run:",
                f"  output_dir: {tmp_path.as_posix()}/dense_run",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def test_run_dense_pipeline_smoke(monkeypatch, tmp_path: Path) -> None:
    workflow_module = importlib.import_module("gnn_pruning.training.workflow")
    monkeypatch.setattr(workflow_module, "load_dataset", lambda name, root: _build_dummy_dataset())

    config_path = _write_config(tmp_path)
    artifacts = run_dense_pipeline(str(config_path))

    assert artifacts.checkpoint_path.exists()
    assert artifacts.train_metrics_path.exists()
    assert artifacts.eval_metrics_path.exists()
    assert artifacts.config_snapshot_path.exists()
    assert artifacts.split_path.exists()
    assert artifacts.summary_path.exists()
    assert artifacts.csv_path.exists()


def test_dense_csv_schema_presence() -> None:
    expected = {
        "experiment_name",
        "dataset",
        "model",
        "seed",
        "train_accuracy",
        "train_macro_f1",
        "val_accuracy",
        "val_macro_f1",
        "test_accuracy",
        "test_macro_f1",
        "best_epoch",
        "epochs_ran",
        "checkpoint_path",
        "metrics_path",
        "config_path",
        "split_path",
    }
    assert expected.issubset(set(csv_columns()))

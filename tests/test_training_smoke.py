"""Smoke tests for dense training/evaluation workflows."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import torch
from torch_geometric.data import Data

from gnn_pruning.training.workflow import evaluate_dense_and_save, train_dense


def _build_dummy_dataset(num_nodes: int = 10, in_channels: int = 4, num_classes: int = 3):
    x = torch.randn((num_nodes, in_channels), dtype=torch.float32)
    y = torch.randint(0, num_classes, (num_nodes,), dtype=torch.long)
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
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
    config_path = tmp_path / "citeseer_fast_debug.yaml"
    config_path.write_text(
        "\n".join(
            [
                "base: base/default",
                "dataset: citeseer",
                "model: gcn",
                "preset: fast_debug",
                "run:",
                f"  output_dir: {tmp_path.as_posix()}/run_artifacts",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def test_dense_train_and_evaluate_smoke(monkeypatch, tmp_path: Path) -> None:
    workflow = importlib.import_module("gnn_pruning.training.workflow")
    monkeypatch.setattr(workflow, "load_dataset", lambda name, root: _build_dummy_dataset())

    config_path = _write_config(tmp_path)

    train_artifacts = train_dense(str(config_path), resume=True)
    assert train_artifacts.checkpoint_path.exists()
    assert train_artifacts.metrics_path.exists()
    assert train_artifacts.resolved_config_path.exists()
    assert train_artifacts.split_path.exists()

    with train_artifacts.metrics_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert "train" in payload and "val" in payload and "test" in payload
    assert "training" in payload

    eval_artifacts = evaluate_dense_and_save(str(config_path), checkpoint_path=str(train_artifacts.checkpoint_path))
    assert eval_artifacts.metrics_path.exists()
    assert eval_artifacts.resolved_config_path.exists()


def test_dense_training_resume_from_checkpoint(monkeypatch, tmp_path: Path) -> None:
    workflow = importlib.import_module("gnn_pruning.training.workflow")
    monkeypatch.setattr(workflow, "load_dataset", lambda name, root: _build_dummy_dataset())

    config_path = _write_config(tmp_path)

    first_run = train_dense(str(config_path), resume=True)
    second_run = train_dense(str(config_path), resume=True)

    assert first_run.checkpoint_path == second_run.checkpoint_path
    assert second_run.checkpoint_path.exists()

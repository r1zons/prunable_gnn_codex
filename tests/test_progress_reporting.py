"""Progress reporting tests."""

from __future__ import annotations

import importlib
from pathlib import Path

import torch
from torch_geometric.data import Data

from gnn_pruning.pipelines import run_dense_pipeline


def _build_dummy_dataset(num_nodes: int = 24, in_channels: int = 6, num_classes: int = 3):
    x = torch.randn((num_nodes, in_channels), dtype=torch.float32)
    y = torch.randint(0, num_classes, (num_nodes,), dtype=torch.long)
    edge_index = torch.vstack([torch.arange(num_nodes), torch.roll(torch.arange(num_nodes), shifts=-1)]).long()
    data = Data(x=x, y=y, edge_index=edge_index)

    class DummyDataset:
        def __getitem__(self, idx: int):
            _ = idx
            return data

    return DummyDataset()


def _write_config(path: Path, output_dir: Path, show_progress: bool) -> Path:
    path.write_text(
        "\n".join(
            [
                "base: base/default",
                "dataset: citeseer",
                "model: gcn",
                "preset: fast_debug",
                "run:",
                f"  output_dir: {output_dir.as_posix()}",
                "logging:",
                f"  show_progress: {'true' if show_progress else 'false'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_progress_flag_enables_stage_output(monkeypatch, tmp_path: Path, capsys) -> None:
    workflow_module = importlib.import_module("gnn_pruning.training.workflow")
    monkeypatch.setattr(workflow_module, "load_dataset", lambda name, root: _build_dummy_dataset())
    cfg = _write_config(tmp_path / "cfg.yaml", tmp_path / "run", show_progress=False)
    run_dense_pipeline(str(cfg), show_progress=True)
    out = capsys.readouterr().out
    assert "[1/6] Loading dataset" in out
    assert "Epoch 1/" in out


def test_progress_disabled_keeps_output_quiet(monkeypatch, tmp_path: Path, capsys) -> None:
    workflow_module = importlib.import_module("gnn_pruning.training.workflow")
    monkeypatch.setattr(workflow_module, "load_dataset", lambda name, root: _build_dummy_dataset())
    cfg = _write_config(tmp_path / "cfg.yaml", tmp_path / "run", show_progress=False)
    run_dense_pipeline(str(cfg), show_progress=False)
    out = capsys.readouterr().out
    assert "[1/6] Loading dataset" not in out
    assert "Epoch 1/" not in out


def test_progress_log_file_created_when_enabled(monkeypatch, tmp_path: Path) -> None:
    workflow_module = importlib.import_module("gnn_pruning.training.workflow")
    monkeypatch.setattr(workflow_module, "load_dataset", lambda name, root: _build_dummy_dataset())
    cfg = _write_config(tmp_path / "cfg.yaml", tmp_path / "run", show_progress=True)
    artifacts = run_dense_pipeline(str(cfg), show_progress=True)
    log_path = artifacts.output_dir / "progress.log"
    assert log_path.exists()


def test_epoch_progress_written_during_training(monkeypatch, tmp_path: Path) -> None:
    workflow_module = importlib.import_module("gnn_pruning.training.workflow")
    monkeypatch.setattr(workflow_module, "load_dataset", lambda name, root: _build_dummy_dataset())
    cfg = _write_config(tmp_path / "cfg.yaml", tmp_path / "run", show_progress=True)
    artifacts = run_dense_pipeline(str(cfg), show_progress=True)
    content = (artifacts.output_dir / "progress.log").read_text(encoding="utf-8")
    assert "Epoch 1/" in content

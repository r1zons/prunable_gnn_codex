"""Regression tests for structured pruning runtime execution path."""

from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path

import torch
from torch_geometric.data import Data

from gnn_pruning.models import GCNNodeClassifier
from gnn_pruning.pruning.workflow import finetune_pruned_checkpoint, prune_from_checkpoint


def _dummy_dataset(num_nodes: int = 20, in_channels: int = 6, num_classes: int = 3):
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


def _make_checkpoint(path: Path, hidden_channels: int = 8) -> Path:
    model = GCNNodeClassifier(in_channels=6, hidden_channels=hidden_channels, out_channels=3, num_layers=2, dropout=0.0)
    torch.save(
        {
            "model_name": "gcn",
            "model_config": {
                "in_channels": 6,
                "hidden_channels": hidden_channels,
                "out_channels": 3,
                "num_layers": 2,
                "dropout": 0.0,
            },
            "model_state_dict": model.state_dict(),
        },
        path,
    )
    return path


def _make_config(path: Path, output_dir: Path, method: str, sparsity: float) -> Path:
    path.write_text(
        "\n".join(
            [
                "base: base/default",
                "dataset: cora",
                "model: gcn",
                "run:",
                "  seed: 42",
                f"  output_dir: {output_dir.as_posix()}",
                "pruning:",
                f"  method: {method}",
                f"  target_sparsity: {sparsity}",
                "  structured: true",
                "  finetune_epochs: 2",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _param_count_from_checkpoint(path: Path) -> int:
    payload = torch.load(path, map_location="cpu")
    return int(sum(tensor.numel() for tensor in payload["model_state_dict"].values()))


def _load_collect_module():
    module_path = Path("scripts/collect_presentation_results.py")
    spec = importlib.util.spec_from_file_location("collect_presentation_results", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_structured_pruning_changes_runtime_model_parameter_count(monkeypatch, tmp_path: Path) -> None:
    training_workflow = importlib.import_module("gnn_pruning.training.workflow")
    pruning_workflow = importlib.import_module("gnn_pruning.pruning.workflow")
    monkeypatch.setattr(training_workflow, "load_dataset", lambda name, root: _dummy_dataset())
    monkeypatch.setattr(pruning_workflow, "load_dataset", lambda name, root: _dummy_dataset())

    dense_ckpt = _make_checkpoint(tmp_path / "dense.pt", hidden_channels=10)
    config = _make_config(tmp_path / "cfg.yaml", tmp_path / "out", method="global_magnitude", sparsity=0.5)
    dense_params = _param_count_from_checkpoint(dense_ckpt)

    artifacts = prune_from_checkpoint(str(dense_ckpt), str(config))
    metrics = json.loads(artifacts.post_prune_metrics_path.read_text(encoding="utf-8"))
    evaluated_params = int(metrics["benchmark"]["parameter_count"])

    assert evaluated_params < dense_params


def test_structured_pruning_changes_saved_checkpoint_model(monkeypatch, tmp_path: Path) -> None:
    training_workflow = importlib.import_module("gnn_pruning.training.workflow")
    pruning_workflow = importlib.import_module("gnn_pruning.pruning.workflow")
    monkeypatch.setattr(training_workflow, "load_dataset", lambda name, root: _dummy_dataset())
    monkeypatch.setattr(pruning_workflow, "load_dataset", lambda name, root: _dummy_dataset())

    dense_ckpt = _make_checkpoint(tmp_path / "dense.pt", hidden_channels=10)
    config = _make_config(tmp_path / "cfg.yaml", tmp_path / "out", method="random", sparsity=0.5)
    dense_params = _param_count_from_checkpoint(dense_ckpt)

    artifacts = prune_from_checkpoint(str(dense_ckpt), str(config))
    reloaded_params = _param_count_from_checkpoint(artifacts.pruned_checkpoint_path)

    assert reloaded_params < dense_params


def test_csv_reports_actual_pruned_model_parameter_count(monkeypatch, tmp_path: Path) -> None:
    training_workflow = importlib.import_module("gnn_pruning.training.workflow")
    pruning_workflow = importlib.import_module("gnn_pruning.pruning.workflow")
    monkeypatch.setattr(training_workflow, "load_dataset", lambda name, root: _dummy_dataset())
    monkeypatch.setattr(pruning_workflow, "load_dataset", lambda name, root: _dummy_dataset())

    run_dir = tmp_path / "cora_gcn_global_magnitude_50"
    dense_ckpt = _make_checkpoint(tmp_path / "dense.pt", hidden_channels=10)
    config = _make_config(tmp_path / "cfg.yaml", run_dir, method="global_magnitude", sparsity=0.5)
    artifacts = prune_from_checkpoint(str(dense_ckpt), str(config))
    finetune_pruned_checkpoint(str(artifacts.pruned_checkpoint_path), str(config), finetune_epochs=1)

    collector = _load_collect_module()
    row = collector._collect_pruned(run_dir=run_dir, model="gcn", method="global_magnitude", sparsity=0.5)
    checkpoint_params = _param_count_from_checkpoint(run_dir / "pruned_global_magnitude_post_finetune.pt")

    assert int(row["parameter_count"]) == checkpoint_params
    assert row["mode"] == "structured"


def test_50_and_90_percent_structured_pruning_produce_different_model_sizes(monkeypatch, tmp_path: Path) -> None:
    training_workflow = importlib.import_module("gnn_pruning.training.workflow")
    pruning_workflow = importlib.import_module("gnn_pruning.pruning.workflow")
    monkeypatch.setattr(training_workflow, "load_dataset", lambda name, root: _dummy_dataset())
    monkeypatch.setattr(pruning_workflow, "load_dataset", lambda name, root: _dummy_dataset())

    dense_ckpt = _make_checkpoint(tmp_path / "dense.pt", hidden_channels=12)
    cfg50 = _make_config(tmp_path / "cfg50.yaml", tmp_path / "out50", method="global_magnitude", sparsity=0.5)
    cfg90 = _make_config(tmp_path / "cfg90.yaml", tmp_path / "out90", method="global_magnitude", sparsity=0.9)

    artifacts50 = prune_from_checkpoint(str(dense_ckpt), str(cfg50))
    artifacts90 = prune_from_checkpoint(str(dense_ckpt), str(cfg90))

    p50 = _param_count_from_checkpoint(artifacts50.pruned_checkpoint_path)
    p90 = _param_count_from_checkpoint(artifacts90.pruned_checkpoint_path)
    assert p90 < p50

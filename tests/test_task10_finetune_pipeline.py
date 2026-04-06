"""Task 10 integration tests for post-pruning evaluation and fine-tuning."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from gnn_pruning.models import GCNNodeClassifier, build_model
from gnn_pruning.pruning.workflow import evaluate_pruned_checkpoint, finetune_pruned_checkpoint, prune_from_checkpoint


def _make_dense_checkpoint(path: Path) -> Path:
    model = GCNNodeClassifier(in_channels=6, hidden_channels=8, out_channels=3, num_layers=2, dropout=0.0)
    torch.save(
        {
            "model_name": "gcn",
            "model_config": {
                "in_channels": 6,
                "hidden_channels": 8,
                "out_channels": 3,
                "num_layers": 2,
                "dropout": 0.0,
            },
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {"dummy": True},
        },
        path,
    )
    return path


def _write_config(path: Path, output_dir: Path) -> Path:
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
                "  method: layerwise_magnitude",
                "  target_sparsity: 0.5",
                "  structured: true",
                "  finetune_epochs: 2",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _dummy_dataset() -> object:
    torch.manual_seed(42)
    num_nodes = 24
    x = torch.randn((num_nodes, 6), dtype=torch.float32)
    y = torch.randint(0, 3, (num_nodes,), dtype=torch.long)
    src = torch.arange(0, num_nodes, dtype=torch.long)
    dst = torch.roll(src, shifts=-1)
    data = Data(x=x, y=y, edge_index=torch.vstack([src, dst]))

    class DummyDataset:
        def __getitem__(self, idx: int) -> Data:
            _ = idx
            return data

    return DummyDataset()


@pytest.fixture
def task10_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, Path, Path]:
    training_workflow = importlib.import_module("gnn_pruning.training.workflow")
    pruning_workflow = importlib.import_module("gnn_pruning.pruning.workflow")
    monkeypatch.setattr(training_workflow, "load_dataset", lambda name, root: _dummy_dataset())
    monkeypatch.setattr(pruning_workflow, "load_dataset", lambda name, root: _dummy_dataset())

    output_dir = tmp_path / "artifacts"
    config_path = _write_config(tmp_path / "task10.yaml", output_dir)
    dense_ckpt = _make_dense_checkpoint(tmp_path / "dense.pt")
    return dense_ckpt, config_path, output_dir


def _load_model_from_checkpoint(checkpoint_path: Path) -> torch.nn.Module:
    payload = torch.load(checkpoint_path, map_location="cpu")
    model = build_model(payload["model_name"], **payload["model_config"])
    model.load_state_dict(payload["model_state_dict"])
    return model


def test_post_prune_evaluation_runs(task10_env: tuple[Path, Path, Path]) -> None:
    dense_ckpt, config_path, _ = task10_env
    prune_artifacts = prune_from_checkpoint(str(dense_ckpt), str(config_path))

    assert prune_artifacts.post_prune_metrics_path is not None
    assert prune_artifacts.post_prune_metrics_path.exists()
    metrics = json.loads(prune_artifacts.post_prune_metrics_path.read_text(encoding="utf-8"))
    assert "accuracy" in metrics["test"]
    assert "macro_f1" in metrics["test"]

    model = _load_model_from_checkpoint(prune_artifacts.pruned_checkpoint_path)
    data = _dummy_dataset()[0]
    with torch.no_grad():
        logits = model(data)
    assert logits.shape == (data.num_nodes, 3)


def test_finetune_from_pruned_checkpoint_runs(task10_env: tuple[Path, Path, Path]) -> None:
    dense_ckpt, config_path, _ = task10_env
    prune_artifacts = prune_from_checkpoint(str(dense_ckpt), str(config_path))
    finetune_artifacts = finetune_pruned_checkpoint(str(prune_artifacts.pruned_checkpoint_path), str(config_path))

    assert finetune_artifacts.post_finetune_checkpoint_path.exists()


def test_metrics_saved_for_all_task10_phases(task10_env: tuple[Path, Path, Path]) -> None:
    dense_ckpt, config_path, _ = task10_env
    prune_artifacts = prune_from_checkpoint(str(dense_ckpt), str(config_path))
    finetune_artifacts = finetune_pruned_checkpoint(str(prune_artifacts.pruned_checkpoint_path), str(config_path))

    assert prune_artifacts.post_prune_metrics_path is not None
    assert prune_artifacts.post_prune_metrics_path.exists()
    assert finetune_artifacts.post_finetune_metrics_path.exists()

    post_prune = json.loads(prune_artifacts.post_prune_metrics_path.read_text(encoding="utf-8"))
    post_finetune = json.loads(finetune_artifacts.post_finetune_metrics_path.read_text(encoding="utf-8"))
    assert "accuracy" in post_prune["test"] and "macro_f1" in post_prune["test"]
    assert "accuracy" in post_finetune["test"] and "macro_f1" in post_finetune["test"]


def test_structured_pruned_model_keeps_valid_output_shape_after_finetune(task10_env: tuple[Path, Path, Path]) -> None:
    dense_ckpt, config_path, _ = task10_env
    prune_artifacts = prune_from_checkpoint(str(dense_ckpt), str(config_path))
    finetune_artifacts = finetune_pruned_checkpoint(str(prune_artifacts.pruned_checkpoint_path), str(config_path))

    model = _load_model_from_checkpoint(finetune_artifacts.post_finetune_checkpoint_path)
    data = _dummy_dataset()[0]
    with torch.no_grad():
        logits = model(data)
    assert logits.shape == (data.num_nodes, 3)


def test_finetune_uses_fresh_optimizer_state(task10_env: tuple[Path, Path, Path]) -> None:
    dense_ckpt, config_path, _ = task10_env
    prune_artifacts = prune_from_checkpoint(str(dense_ckpt), str(config_path))
    finetune_artifacts = finetune_pruned_checkpoint(str(prune_artifacts.pruned_checkpoint_path), str(config_path))

    post_payload = torch.load(finetune_artifacts.post_finetune_checkpoint_path, map_location="cpu")
    assert "optimizer_state_dict" not in post_payload
    assert "finetune_training" in post_payload


def test_structured_pruning_reduces_parameter_count_before_finetune(task10_env: tuple[Path, Path, Path]) -> None:
    dense_ckpt, config_path, _ = task10_env
    dense_model = _load_model_from_checkpoint(dense_ckpt)
    prune_artifacts = prune_from_checkpoint(str(dense_ckpt), str(config_path))
    pruned_model = _load_model_from_checkpoint(prune_artifacts.pruned_checkpoint_path)

    dense_params = sum(parameter.numel() for parameter in dense_model.parameters())
    pruned_params = sum(parameter.numel() for parameter in pruned_model.parameters())
    assert pruned_params < dense_params


def test_invalid_pruned_checkpoint_fails_cleanly(task10_env: tuple[Path, Path, Path], tmp_path: Path) -> None:
    _, config_path, _ = task10_env
    bad_checkpoint = tmp_path / "bad_pruned.pt"
    torch.save({"model_name": "gcn"}, bad_checkpoint)

    with pytest.raises(ValueError, match="Invalid pruned checkpoint"):
        finetune_pruned_checkpoint(str(bad_checkpoint), str(config_path))


def test_missing_metrics_or_artifact_path_fails_cleanly(task10_env: tuple[Path, Path, Path], tmp_path: Path) -> None:
    _, config_path, _ = task10_env
    missing = tmp_path / "does_not_exist.pt"

    with pytest.raises(FileNotFoundError, match="Pruned checkpoint not found"):
        evaluate_pruned_checkpoint(str(missing), str(config_path))

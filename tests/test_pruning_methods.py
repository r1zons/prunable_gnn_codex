"""Tests for first concrete pruning methods and workflow."""

from __future__ import annotations

import json
import importlib
from pathlib import Path

import torch
from torch_geometric.data import Data

from gnn_pruning.models import GCNNodeClassifier
from gnn_pruning.pruning import PruningContext, get_pruner
from gnn_pruning.pruning.workflow import finetune_pruned_checkpoint, prune_from_checkpoint


def _make_checkpoint(path: Path) -> Path:
    model = GCNNodeClassifier(in_channels=6, hidden_channels=8, out_channels=3, num_layers=2, dropout=0.0)
    payload = {
        "model_name": "gcn",
        "model_config": {
            "in_channels": 6,
            "hidden_channels": 8,
            "out_channels": 3,
            "num_layers": 2,
            "dropout": 0.0,
        },
        "model_state_dict": model.state_dict(),
    }
    torch.save(payload, path)
    return path


def _make_config(path: Path, output_dir: Path, method: str, structured: bool) -> Path:
    path.write_text(
        "\n".join(
            [
                "base: base/default",
                "dataset: cora",
                "model: gcn",
                "run:",
                f"  output_dir: {output_dir.as_posix()}",
                "pruning:",
                f"  method: {method}",
                "  target_sparsity: 0.5",
                f"  structured: {'true' if structured else 'false'}",
                "  finetune_epochs: 3",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _param_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


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


def _saliency_context(model: GCNNodeClassifier, num_nodes: int = 20) -> PruningContext:
    data = _dummy_dataset(num_nodes=num_nodes, in_channels=model.in_channels, num_classes=model.out_channels)[0]
    train_count = max(2, num_nodes // 2)
    train_idx = torch.arange(train_count, dtype=torch.long)
    return PruningContext(config={}, data={"data": data, "train_idx": train_idx}, device="cpu", seed=42)


def test_pruner_outputs_valid_plan() -> None:
    model = GCNNodeClassifier(in_channels=6, hidden_channels=8, out_channels=3, num_layers=2, dropout=0.0)
    pruner = get_pruner("random")()
    context = PruningContext(config={}, data=None, device="cpu", seed=42)

    scores = pruner.score(model, context)
    _, plan = pruner.apply(model, scores, context, target_sparsity=0.5, structured=False)

    assert plan.name == "random"
    assert 0.0 <= plan.target_sparsity <= 1.0
    assert plan.pruning_time_sec is not None


def test_pruning_runs_end_to_end_on_small_model(monkeypatch, tmp_path: Path) -> None:
    training_workflow = importlib.import_module("gnn_pruning.training.workflow")
    pruning_workflow = importlib.import_module("gnn_pruning.pruning.workflow")
    monkeypatch.setattr(training_workflow, "load_dataset", lambda name, root: _dummy_dataset())
    monkeypatch.setattr(pruning_workflow, "load_dataset", lambda name, root: _dummy_dataset())

    ckpt = _make_checkpoint(tmp_path / "dense.pt")
    cfg = _make_config(tmp_path / "cfg.yaml", tmp_path / "artifacts", method="global_magnitude", structured=False)

    artifacts = prune_from_checkpoint(str(ckpt), str(cfg))

    assert artifacts.pruned_checkpoint_path.exists()
    assert artifacts.pruning_metrics_path.exists()
    assert artifacts.post_prune_metrics_path is not None
    assert artifacts.post_prune_metrics_path.exists()

    payload = json.loads(artifacts.pruning_metrics_path.read_text(encoding="utf-8"))
    assert payload["name"] == "global_magnitude"
    assert payload["target_sparsity"] == 0.5


def test_structured_mode_reduces_parameter_count() -> None:
    model = GCNNodeClassifier(in_channels=6, hidden_channels=8, out_channels=3, num_layers=2, dropout=0.0)
    pruner = get_pruner("layerwise_magnitude")()
    context = PruningContext(config={}, data=None, device="cpu", seed=42)

    scores = pruner.score(model, context)
    pruned_model, plan = pruner.apply(model, scores, context, target_sparsity=0.5, structured=True)

    assert _param_count(pruned_model) < _param_count(model)
    assert plan.details["mode"] == "structured"
    assert plan.achieved_sparsity is not None


def test_random_structured_pruning_applies_surgery_and_preserves_forward_shape() -> None:
    model = GCNNodeClassifier(in_channels=6, hidden_channels=8, out_channels=3, num_layers=2, dropout=0.0)
    pruner = get_pruner("random")()
    context = PruningContext(config={}, data=None, device="cpu", seed=42)

    scores = pruner.score(model, context, structured=True, target_sparsity=0.5)
    pruned_model, plan = pruner.apply(model, scores, context, structured=True, target_sparsity=0.5)

    assert plan.details["hidden_dim_after"] < plan.details["hidden_dim_before"]
    assert plan.details["parameter_count_after"] < plan.details["parameter_count_before"]
    assert len(plan.details["kept_channel_indices"]) == plan.details["kept_channels"]

    data = _dummy_dataset(num_nodes=16, in_channels=6, num_classes=3)[0]
    with torch.no_grad():
        logits = pruned_model(data)
    assert logits.shape == (data.num_nodes, 3)


def test_global_magnitude_structured_pruning_applies_surgery_and_preserves_forward_shape() -> None:
    model = GCNNodeClassifier(in_channels=6, hidden_channels=10, out_channels=3, num_layers=2, dropout=0.0)
    pruner = get_pruner("global_magnitude")()
    context = PruningContext(config={}, data=None, device="cpu", seed=42)

    scores = pruner.score(model, context, structured=True, target_sparsity=0.6)
    pruned_model, plan = pruner.apply(model, scores, context, structured=True, target_sparsity=0.6)

    assert plan.details["hidden_dim_after"] < plan.details["hidden_dim_before"]
    assert plan.details["parameter_count_after"] < plan.details["parameter_count_before"]
    assert plan.details["scope"] == "structured_hidden_channels"
    assert plan.achieved_sparsity is not None

    data = _dummy_dataset(num_nodes=18, in_channels=6, num_classes=3)[0]
    with torch.no_grad():
        logits = pruned_model(data)
    assert logits.shape == (data.num_nodes, 3)


def test_unstructured_mode_does_not_report_structural_compression() -> None:
    model = GCNNodeClassifier(in_channels=6, hidden_channels=8, out_channels=3, num_layers=2, dropout=0.0)
    pruner = get_pruner("global_magnitude")()
    context = PruningContext(config={}, data=None, device="cpu", seed=42)

    scores = pruner.score(model, context, structured=False, target_sparsity=0.5)
    _, plan = pruner.apply(model, scores, context, structured=False, target_sparsity=0.5)

    assert plan.details["scope"] == "global"
    assert plan.details["structural_compression"] is False


def test_snip_unstructured_pruning_runs_with_gradient_saliency() -> None:
    model = GCNNodeClassifier(in_channels=6, hidden_channels=8, out_channels=3, num_layers=2, dropout=0.0)
    pruner = get_pruner("snip")()
    context = _saliency_context(model)

    scores = pruner.score(model, context, structured=False, target_sparsity=0.5)
    _, plan = pruner.apply(model, scores, context, structured=False, target_sparsity=0.5)

    assert plan.name == "snip"
    assert plan.details["scope"] == "global"
    assert plan.achieved_sparsity is not None


def test_grasp_structured_pruning_reuses_surgery() -> None:
    model = GCNNodeClassifier(in_channels=6, hidden_channels=10, out_channels=3, num_layers=2, dropout=0.0)
    pruner = get_pruner("grasp")()
    context = _saliency_context(model)

    scores = pruner.score(model, context, structured=True, target_sparsity=0.6)
    pruned_model, plan = pruner.apply(model, scores, context, structured=True, target_sparsity=0.6)

    assert plan.name == "grasp"
    assert plan.details["scope"] == "structured_hidden_channels"
    assert plan.details["parameter_count_after"] < plan.details["parameter_count_before"]

    data = _dummy_dataset(num_nodes=14, in_channels=6, num_classes=3)[0]
    with torch.no_grad():
        logits = pruned_model(data)
    assert logits.shape == (data.num_nodes, 3)


def test_l1_threshold_regularized_scoring_path_records_threshold() -> None:
    model = GCNNodeClassifier(in_channels=6, hidden_channels=8, out_channels=3, num_layers=2, dropout=0.0)
    pruner = get_pruner("l1_threshold")()
    context = _saliency_context(model)

    scores = pruner.score(model, context, structured=False, target_sparsity=0.5, reg_strength=1e-3)
    _, plan = pruner.apply(model, scores, context, structured=False, target_sparsity=0.5, reg_strength=1e-3)

    assert plan.name == "l1_threshold"
    assert plan.details["regularization"] == "l1"
    assert "score_threshold" in plan.details
    assert plan.details["scope"] == "threshold"
    assert plan.achieved_sparsity is not None


def test_group_lasso_structured_pruning_shrinks_model() -> None:
    model = GCNNodeClassifier(in_channels=6, hidden_channels=12, out_channels=3, num_layers=2, dropout=0.0)
    pruner = get_pruner("group_lasso")()
    context = _saliency_context(model)

    scores = pruner.score(model, context, structured=True, target_sparsity=0.5, reg_strength=1e-3)
    pruned_model, plan = pruner.apply(model, scores, context, structured=True, target_sparsity=0.5, reg_strength=1e-3)

    assert plan.name == "group_lasso"
    assert plan.details["regularization"] == "group_lasso"
    assert plan.details["scope"] == "structured_hidden_channels"
    assert plan.details["parameter_count_after"] < plan.details["parameter_count_before"]
    assert _param_count(pruned_model) < _param_count(model)


def test_movement_pruning_reports_compaction_requirement() -> None:
    model = GCNNodeClassifier(in_channels=6, hidden_channels=10, out_channels=3, num_layers=2, dropout=0.0)
    pruner = get_pruner("movement")()
    context = _saliency_context(model)

    scores = pruner.score(model, context, structured=False, target_sparsity=0.5, movement_steps=2, movement_lr=1e-2)
    _, plan = pruner.apply(model, scores, context, structured=False, target_sparsity=0.5)

    assert plan.name == "movement"
    assert plan.details["scope"] == "global"
    assert plan.details["structural_compression_support"] == "requires_compaction_step"
    assert plan.achieved_sparsity is not None


def test_hard_concrete_l0_pruning_reports_compaction_requirement() -> None:
    model = GCNNodeClassifier(in_channels=6, hidden_channels=10, out_channels=3, num_layers=2, dropout=0.0)
    pruner = get_pruner("hard_concrete_l0")()
    context = _saliency_context(model)

    scores = pruner.score(model, context, structured=False, target_sparsity=0.5, gate_steps=2, gate_lr=1e-2, l0_lambda=1e-3)
    _, plan = pruner.apply(model, scores, context, structured=False, target_sparsity=0.5)

    assert plan.name == "hard_concrete_l0"
    assert plan.details["scope"] == "global"
    assert plan.details["structural_compression_support"] == "requires_compaction_step"
    assert plan.achieved_sparsity is not None


def test_finetuning_runs_and_saves_post_checkpoint(monkeypatch, tmp_path: Path) -> None:
    training_workflow = importlib.import_module("gnn_pruning.training.workflow")
    pruning_workflow = importlib.import_module("gnn_pruning.pruning.workflow")
    monkeypatch.setattr(training_workflow, "load_dataset", lambda name, root: _dummy_dataset())
    monkeypatch.setattr(pruning_workflow, "load_dataset", lambda name, root: _dummy_dataset())

    ckpt = _make_checkpoint(tmp_path / "dense.pt")
    cfg = _make_config(tmp_path / "cfg.yaml", tmp_path / "artifacts", method="random", structured=False)
    prune_artifacts = prune_from_checkpoint(str(ckpt), str(cfg))

    finetune_artifacts = finetune_pruned_checkpoint(str(prune_artifacts.pruned_checkpoint_path), str(cfg))
    assert finetune_artifacts.pre_finetune_checkpoint_path.exists()
    assert finetune_artifacts.post_finetune_checkpoint_path.exists()


def test_metric_files_exist_for_prune_and_finetune_phases(monkeypatch, tmp_path: Path) -> None:
    training_workflow = importlib.import_module("gnn_pruning.training.workflow")
    pruning_workflow = importlib.import_module("gnn_pruning.pruning.workflow")
    monkeypatch.setattr(training_workflow, "load_dataset", lambda name, root: _dummy_dataset())
    monkeypatch.setattr(pruning_workflow, "load_dataset", lambda name, root: _dummy_dataset())

    ckpt = _make_checkpoint(tmp_path / "dense.pt")
    cfg = _make_config(tmp_path / "cfg.yaml", tmp_path / "artifacts", method="layerwise_magnitude", structured=False)
    prune_artifacts = prune_from_checkpoint(str(ckpt), str(cfg))
    finetune_artifacts = finetune_pruned_checkpoint(str(prune_artifacts.pruned_checkpoint_path), str(cfg), finetune_epochs=2)

    assert prune_artifacts.pruning_metrics_path.exists()
    assert prune_artifacts.post_prune_metrics_path is not None
    assert prune_artifacts.post_prune_metrics_path.exists()
    assert finetune_artifacts.pre_finetune_metrics_path.exists()
    assert finetune_artifacts.post_finetune_metrics_path.exists()


def test_snip_pruning_workflow_smoke(monkeypatch, tmp_path: Path) -> None:
    training_workflow = importlib.import_module("gnn_pruning.training.workflow")
    pruning_workflow = importlib.import_module("gnn_pruning.pruning.workflow")
    monkeypatch.setattr(training_workflow, "load_dataset", lambda name, root: _dummy_dataset())
    monkeypatch.setattr(pruning_workflow, "load_dataset", lambda name, root: _dummy_dataset())

    ckpt = _make_checkpoint(tmp_path / "dense.pt")
    cfg = _make_config(tmp_path / "cfg.yaml", tmp_path / "artifacts", method="snip", structured=True)

    artifacts = prune_from_checkpoint(str(ckpt), str(cfg))

    payload = json.loads(artifacts.pruning_metrics_path.read_text(encoding="utf-8"))
    assert payload["name"] == "snip"
    assert payload["mode"] == "structured"


def test_movement_pruning_workflow_smoke(monkeypatch, tmp_path: Path) -> None:
    training_workflow = importlib.import_module("gnn_pruning.training.workflow")
    pruning_workflow = importlib.import_module("gnn_pruning.pruning.workflow")
    monkeypatch.setattr(training_workflow, "load_dataset", lambda name, root: _dummy_dataset())
    monkeypatch.setattr(pruning_workflow, "load_dataset", lambda name, root: _dummy_dataset())

    ckpt = _make_checkpoint(tmp_path / "dense.pt")
    cfg = _make_config(tmp_path / "cfg.yaml", tmp_path / "artifacts", method="movement", structured=False)

    artifacts = prune_from_checkpoint(str(ckpt), str(cfg))

    payload = json.loads(artifacts.pruning_metrics_path.read_text(encoding="utf-8"))
    assert payload["name"] == "movement"
    assert payload["mode"] == "unstructured"

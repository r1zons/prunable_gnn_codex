"""Tests for adaptive_layerwise structural pruning baseline."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import torch
from torch_geometric.data import Data

from gnn_pruning.models import GCNNodeClassifier, GraphSAGENodeClassifier
from gnn_pruning.pruning import PruningContext, get_pruner
from gnn_pruning.pruning.workflow import prune_from_checkpoint
from gnn_pruning.reporting.csv_reporter import PIPELINE_RESULTS_COLUMNS


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


def _make_checkpoint(path: Path) -> Path:
    model = GraphSAGENodeClassifier(in_channels=6, hidden_channels=16, out_channels=3, num_layers=3, dropout=0.0)
    torch.save(
        {
            "model_name": "graphsage",
            "model_config": {
                "in_channels": 6,
                "hidden_channels": 16,
                "out_channels": 3,
                "num_layers": 3,
                "dropout": 0.0,
            },
            "model_state_dict": model.state_dict(),
        },
        path,
    )
    return path


def _make_config(path: Path, output_dir: Path, target_sparsity: float) -> Path:
    path.write_text(
        "\n".join(
            [
                "base: base/default",
                "dataset: citeseer",
                "model: graphsage",
                "run:",
                f"  output_dir: {output_dir.as_posix()}",
                "pruning:",
                "  method: adaptive_layerwise",
                f"  target_sparsity: {target_sparsity}",
                "  structured: true",
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
    return path


def _param_count(model: torch.nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def test_adaptive_pruner_is_registered() -> None:
    pruner_cls = get_pruner("adaptive_layerwise")
    assert pruner_cls.name == "adaptive_layerwise"


def test_adaptive_pruning_reduces_parameter_count() -> None:
    model = GCNNodeClassifier(in_channels=6, hidden_channels=12, out_channels=3, num_layers=3, dropout=0.0)
    pruner = get_pruner("adaptive_layerwise")()
    context = PruningContext(config={"adaptive_pruning": {"step_prune_ratio": 0.25, "max_steps": 3, "max_accuracy_drop": 1.0, "min_channels_per_layer": 4}}, data=None, device="cpu", seed=42)

    score = pruner.score(model, context, structured=True, target_sparsity=0.5)
    pruned_model, plan = pruner.apply(model, score, context, structured=True, target_sparsity=0.5)

    assert _param_count(pruned_model) < _param_count(model)
    assert plan.details["num_adaptive_steps"] >= 1


def test_adaptive_pruning_stops_on_max_accuracy_drop(monkeypatch) -> None:
    methods_module = importlib.import_module("gnn_pruning.pruning.methods")
    values = iter([1.0, 0.5])
    monkeypatch.setattr(methods_module, "_validation_accuracy", lambda model, context: next(values, 0.5))

    model = GCNNodeClassifier(in_channels=6, hidden_channels=12, out_channels=3, num_layers=3, dropout=0.0)
    pruner = get_pruner("adaptive_layerwise")()
    context = PruningContext(config={"adaptive_pruning": {"step_prune_ratio": 0.25, "max_steps": 4, "max_accuracy_drop": 0.1, "min_channels_per_layer": 4}}, data=None, device="cpu", seed=42)

    score = pruner.score(model, context, structured=True, target_sparsity=0.9)
    _, plan = pruner.apply(model, score, context, structured=True, target_sparsity=0.9)

    assert plan.details["stop_reason"] == "max_accuracy_drop_exceeded"


def test_adaptive_pruning_stops_on_target_sparsity(monkeypatch) -> None:
    methods_module = importlib.import_module("gnn_pruning.pruning.methods")
    monkeypatch.setattr(methods_module, "_validation_accuracy", lambda model, context: 1.0)

    model = GCNNodeClassifier(in_channels=6, hidden_channels=12, out_channels=3, num_layers=3, dropout=0.0)
    pruner = get_pruner("adaptive_layerwise")()
    context = PruningContext(config={"adaptive_pruning": {"step_prune_ratio": 0.5, "max_steps": 6, "max_accuracy_drop": 1.0, "min_channels_per_layer": 4}}, data=None, device="cpu", seed=42)

    score = pruner.score(model, context, structured=True, target_sparsity=0.2)
    _, plan = pruner.apply(model, score, context, structured=True, target_sparsity=0.2)

    assert plan.details["stop_reason"] == "target_sparsity_reached"


def test_adaptive_pruned_model_supports_forward_pass() -> None:
    model = GraphSAGENodeClassifier(in_channels=6, hidden_channels=16, out_channels=3, num_layers=3, dropout=0.0)
    pruner = get_pruner("adaptive_layerwise")()
    context = PruningContext(config={"adaptive_pruning": {"step_prune_ratio": 0.25, "max_steps": 2, "max_accuracy_drop": 1.0, "min_channels_per_layer": 4}}, data=None, device="cpu", seed=42)

    score = pruner.score(model, context, structured=True, target_sparsity=0.5)
    pruned_model, _ = pruner.apply(model, score, context, structured=True, target_sparsity=0.5)

    data = _dummy_dataset(num_nodes=16, in_channels=6, num_classes=3)[0]
    with torch.no_grad():
        logits = pruned_model(data)
    assert logits.shape == (data.num_nodes, 3)


def test_adaptive_trace_is_saved(monkeypatch, tmp_path: Path) -> None:
    training_workflow = importlib.import_module("gnn_pruning.training.workflow")
    pruning_workflow = importlib.import_module("gnn_pruning.pruning.workflow")
    monkeypatch.setattr(training_workflow, "load_dataset", lambda name, root: _dummy_dataset())
    monkeypatch.setattr(pruning_workflow, "load_dataset", lambda name, root: _dummy_dataset())

    ckpt = _make_checkpoint(tmp_path / "dense.pt")
    cfg = _make_config(tmp_path / "cfg.yaml", tmp_path / "run", target_sparsity=0.5)
    artifacts = prune_from_checkpoint(str(ckpt), str(cfg))

    assert artifacts.adaptive_trace_path is not None
    assert artifacts.adaptive_trace_path.exists()
    trace = json.loads(artifacts.adaptive_trace_path.read_text(encoding="utf-8"))
    assert isinstance(trace, list)
    assert trace


def test_adaptive_csv_fields_exist() -> None:
    for column in ("final_reward", "num_adaptive_steps", "stop_reason"):
        assert column in PIPELINE_RESULTS_COLUMNS

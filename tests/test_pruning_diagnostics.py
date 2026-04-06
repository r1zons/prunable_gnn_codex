"""Diagnostics-focused regression tests for pruning pipeline."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from gnn_pruning.models import GCNNodeClassifier, GraphSAGENodeClassifier
from gnn_pruning.pruning import PruningContext, get_pruner
from gnn_pruning.pruning.workflow import finetune_pruned_checkpoint, prune_from_checkpoint


def _dummy_dataset(num_nodes: int = 30, in_channels: int = 6, num_classes: int = 3):
    x = torch.randn((num_nodes, in_channels), dtype=torch.float32)
    y = torch.randint(0, num_classes, (num_nodes,), dtype=torch.long)
    edge_index = torch.vstack(
        [torch.arange(num_nodes, dtype=torch.long), torch.roll(torch.arange(num_nodes, dtype=torch.long), shifts=-1)]
    )
    data = Data(x=x, y=y, edge_index=edge_index)

    class DummyDataset:
        def __getitem__(self, idx: int):
            _ = idx
            return data

    return DummyDataset()


def _make_checkpoint(path: Path, model_name: str = "gcn", hidden: int = 8) -> Path:
    if model_name == "graphsage":
        model = GraphSAGENodeClassifier(in_channels=6, hidden_channels=hidden, out_channels=3, num_layers=2, dropout=0.0)
    else:
        model = GCNNodeClassifier(in_channels=6, hidden_channels=hidden, out_channels=3, num_layers=2, dropout=0.0)
    torch.save(
        {
            "model_name": model_name,
            "model_config": {
                "in_channels": 6,
                "hidden_channels": hidden,
                "out_channels": 3,
                "num_layers": 2,
                "dropout": 0.0,
            },
            "model_state_dict": model.state_dict(),
        },
        path,
    )
    return path


def _make_config(path: Path, output_dir: Path, dataset: str, model: str, method: str, sparsity: float) -> Path:
    path.write_text(
        "\n".join(
            [
                "base: base/default",
                f"dataset: {dataset}",
                f"model: {model}",
                "preset: fast_debug",
                "run:",
                "  seed: 42",
                f"  output_dir: {output_dir.as_posix()}",
                "pruning:",
                f"  method: {method}",
                f"  target_sparsity: {sparsity}",
                "  structured: true",
                "  finetune_epochs: 2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _param_count(checkpoint_path: Path) -> int:
    payload = torch.load(checkpoint_path, map_location="cpu")
    return int(sum(t.numel() for t in payload["model_state_dict"].values()))


@pytest.fixture
def patched_data(monkeypatch):
    training_workflow = importlib.import_module("gnn_pruning.training.workflow")
    pruning_workflow = importlib.import_module("gnn_pruning.pruning.workflow")
    monkeypatch.setattr(training_workflow, "load_dataset", lambda name, root: _dummy_dataset())
    monkeypatch.setattr(pruning_workflow, "load_dataset", lambda name, root: _dummy_dataset())


def test_post_prune_and_post_finetune_are_saved_separately(patched_data, tmp_path: Path) -> None:
    ckpt = _make_checkpoint(tmp_path / "dense.pt")
    cfg = _make_config(tmp_path / "cfg.yaml", tmp_path / "run", "cora", "gcn", "global_magnitude", 0.5)
    prune_artifacts = prune_from_checkpoint(str(ckpt), str(cfg))
    finetune_artifacts = finetune_pruned_checkpoint(str(prune_artifacts.pruned_checkpoint_path), str(cfg))

    assert prune_artifacts.post_prune_metrics_path.exists()
    assert finetune_artifacts.post_finetune_metrics_path.exists()
    assert prune_artifacts.post_prune_metrics_path != finetune_artifacts.post_finetune_metrics_path


def test_random_and_global_magnitude_produce_different_channel_selections() -> None:
    model = GCNNodeClassifier(in_channels=6, hidden_channels=12, out_channels=3, num_layers=2, dropout=0.0)
    context = PruningContext(config={}, data=None, device="cpu", seed=42)

    random_pruner = get_pruner("random")()
    gm_pruner = get_pruner("global_magnitude")()

    random_scores = random_pruner.score(model, context, structured=True, target_sparsity=0.5)
    gm_scores = gm_pruner.score(model, context, structured=True, target_sparsity=0.5)

    _, random_plan = random_pruner.apply(model, random_scores, context, structured=True, target_sparsity=0.5)
    _, gm_plan = gm_pruner.apply(model, gm_scores, context, structured=True, target_sparsity=0.5)

    assert set(random_plan.details["kept_channel_indices"]) != set(gm_plan.details["kept_channel_indices"])


def test_structural_pruning_changes_real_model_dimensions_in_pipeline(patched_data, tmp_path: Path) -> None:
    ckpt = _make_checkpoint(tmp_path / "dense.pt", hidden=10)
    cfg = _make_config(tmp_path / "cfg.yaml", tmp_path / "run", "cora", "gcn", "random", 0.5)

    artifacts = prune_from_checkpoint(str(ckpt), str(cfg))
    diagnostics = json.loads((tmp_path / "run" / "pruning_diagnostics.json").read_text(encoding="utf-8"))

    assert diagnostics["pruned_hidden_dimensions"][0] < diagnostics["dense_hidden_dimensions"][0]
    assert diagnostics["pruned_parameter_count"] < diagnostics["dense_parameter_count"]
    assert artifacts.pruned_checkpoint_path.exists()


def test_reported_sparsity_matches_actual_model_shape(patched_data, tmp_path: Path) -> None:
    ckpt = _make_checkpoint(tmp_path / "dense.pt", hidden=10)
    cfg = _make_config(tmp_path / "cfg.yaml", tmp_path / "run", "cora", "gcn", "global_magnitude", 0.5)

    prune_from_checkpoint(str(ckpt), str(cfg))
    diag = json.loads((tmp_path / "run" / "pruning_diagnostics.json").read_text(encoding="utf-8"))
    requested = float(diag["requested_sparsity"])
    dense = int(diag["dense_hidden_dimensions"][0])
    pruned = int(diag["pruned_hidden_dimensions"][0])
    expected_achieved = 1.0 - (pruned / dense)

    assert abs(float(diag["achieved_sparsity"]) - expected_achieved) < 1e-6
    assert requested >= expected_achieved


def test_finetune_starts_from_pruned_checkpoint(patched_data, tmp_path: Path) -> None:
    ckpt = _make_checkpoint(tmp_path / "dense.pt", hidden=10)
    cfg = _make_config(tmp_path / "cfg.yaml", tmp_path / "run", "cora", "gcn", "global_magnitude", 0.5)
    prune_artifacts = prune_from_checkpoint(str(ckpt), str(cfg))
    finetune_pruned_checkpoint(str(prune_artifacts.pruned_checkpoint_path), str(cfg), finetune_epochs=1)

    diag = json.loads((tmp_path / "run" / "pruning_diagnostics.json").read_text(encoding="utf-8"))
    assert diag["finetune"]["parameter_count_before_finetune"] < diag["dense_parameter_count"]


def test_graphsage_pubmed_post_prune_behavior_is_recorded(patched_data, tmp_path: Path) -> None:
    ckpt = _make_checkpoint(tmp_path / "dense.pt", model_name="graphsage", hidden=16)
    cfg = _make_config(tmp_path / "cfg.yaml", tmp_path / "run", "pubmed", "graphsage", "global_magnitude", 0.9)

    prune_artifacts = prune_from_checkpoint(str(ckpt), str(cfg))
    finetune_pruned_checkpoint(str(prune_artifacts.pruned_checkpoint_path), str(cfg), finetune_epochs=1)
    diag = json.loads((tmp_path / "run" / "pruning_diagnostics.json").read_text(encoding="utf-8"))

    assert "dense" in diag["metrics"]
    assert "post_prune" in diag["metrics"]
    assert "post_finetune" in diag["metrics"]
    assert "pruned_hidden_dimensions" in diag
    assert "kept_channel_indices_per_layer" in diag


def test_method_outputs_are_not_reused_between_runs(patched_data, tmp_path: Path) -> None:
    ckpt = _make_checkpoint(tmp_path / "dense.pt", hidden=10)
    cfg_random = _make_config(tmp_path / "cfg_random.yaml", tmp_path / "random_run", "cora", "gcn", "random", 0.5)
    cfg_gm = _make_config(tmp_path / "cfg_gm.yaml", tmp_path / "gm_run", "cora", "gcn", "global_magnitude", 0.5)

    random_artifacts = prune_from_checkpoint(str(ckpt), str(cfg_random))
    gm_artifacts = prune_from_checkpoint(str(ckpt), str(cfg_gm))

    assert random_artifacts.pruned_checkpoint_path.parent != gm_artifacts.pruned_checkpoint_path.parent
    assert random_artifacts.pruning_metrics_path != gm_artifacts.pruning_metrics_path


def test_invalid_pruning_plan_fails_cleanly() -> None:
    model = GCNNodeClassifier(in_channels=6, hidden_channels=8, out_channels=3, num_layers=2, dropout=0.0)
    pruner = get_pruner("random")()

    with pytest.raises(ValueError, match="valid PruningPlan"):
        pruner.apply(model=model, pruning_plan=None, context={})

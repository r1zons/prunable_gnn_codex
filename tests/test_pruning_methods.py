"""Tests for first concrete pruning methods and workflow."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from gnn_pruning.models import GCNNodeClassifier
from gnn_pruning.pruning import PruningContext, get_pruner
from gnn_pruning.pruning.workflow import prune_from_checkpoint


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
            ]
        ),
        encoding="utf-8",
    )
    return path


def _param_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def test_pruner_outputs_valid_plan() -> None:
    model = GCNNodeClassifier(in_channels=6, hidden_channels=8, out_channels=3, num_layers=2, dropout=0.0)
    pruner = get_pruner("random")()
    context = PruningContext(config={}, data=None, device="cpu", seed=42)

    scores = pruner.score(model, context)
    _, plan = pruner.apply(model, scores, context, target_sparsity=0.5, structured=False)

    assert plan.name == "random"
    assert 0.0 <= plan.target_sparsity <= 1.0
    assert plan.pruning_time_sec is not None


def test_pruning_runs_end_to_end_on_small_model(tmp_path: Path) -> None:
    ckpt = _make_checkpoint(tmp_path / "dense.pt")
    cfg = _make_config(tmp_path / "cfg.yaml", tmp_path / "artifacts", method="global_magnitude", structured=False)

    artifacts = prune_from_checkpoint(str(ckpt), str(cfg))

    assert artifacts.pruned_checkpoint_path.exists()
    assert artifacts.pruning_metrics_path.exists()

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

"""Multi-dataset pipeline smoke tests for cora/citeseer/pubmed."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import torch
from torch_geometric.data import Data

from gnn_pruning.models import GCNNodeClassifier, GraphSAGENodeClassifier
from gnn_pruning.pruning.workflow import finetune_pruned_checkpoint, prune_from_checkpoint
from gnn_pruning.training.workflow import train_dense


def _dataset_by_name(name: str):
    seed_map = {"cora": 1, "citeseer": 2, "pubmed": 3}
    torch.manual_seed(seed_map[name])
    n = {"cora": 28, "citeseer": 30, "pubmed": 32}[name]
    x = torch.randn((n, 6), dtype=torch.float32)
    if name == "cora":
        y = torch.zeros((n,), dtype=torch.long)
    elif name == "citeseer":
        y = torch.randint(0, 3, (n,), dtype=torch.long)
    else:
        y = (torch.arange(n, dtype=torch.long) % 3).clone()
    edge_index = torch.vstack([torch.arange(n), torch.roll(torch.arange(n), shifts=-1)]).long()
    data = Data(x=x, y=y, edge_index=edge_index)

    class DummyDataset:
        def __getitem__(self, idx: int):
            _ = idx
            return data

    return DummyDataset()


def _write_config(path: Path, dataset: str, model: str, output_dir: Path) -> Path:
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
                "  method: global_magnitude",
                "  target_sparsity: 0.9",
                "  structured: true",
                "  finetune_epochs: 2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _make_dense_checkpoint(path: Path, model: str) -> Path:
    if model == "graphsage":
        net = GraphSAGENodeClassifier(in_channels=6, hidden_channels=12, out_channels=3, num_layers=2, dropout=0.0)
    else:
        net = GCNNodeClassifier(in_channels=6, hidden_channels=12, out_channels=3, num_layers=2, dropout=0.0)
    torch.save(
        {
            "model_name": model,
            "model_config": {
                "in_channels": 6,
                "hidden_channels": 12,
                "out_channels": 3,
                "num_layers": 2,
                "dropout": 0.0,
            },
            "model_state_dict": net.state_dict(),
        },
        path,
    )
    return path


def _patch_datasets(monkeypatch):
    training_workflow = importlib.import_module("gnn_pruning.training.workflow")
    pruning_workflow = importlib.import_module("gnn_pruning.pruning.workflow")
    monkeypatch.setattr(training_workflow, "load_dataset", lambda name, root: _dataset_by_name(name))
    monkeypatch.setattr(pruning_workflow, "load_dataset", lambda name, root: _dataset_by_name(name))


def test_pipeline_runs_on_all_datasets(monkeypatch, tmp_path: Path) -> None:
    _patch_datasets(monkeypatch)
    for dataset in ("cora", "citeseer", "pubmed"):
        cfg = _write_config(tmp_path / f"{dataset}.yaml", dataset, "gcn", tmp_path / dataset)
        dense = _make_dense_checkpoint(tmp_path / f"{dataset}_dense.pt", "gcn")
        artifacts = prune_from_checkpoint(str(dense), str(cfg))
        finetune_pruned_checkpoint(str(artifacts.pruned_checkpoint_path), str(cfg), finetune_epochs=1)


def test_structural_pruning_reduces_params_all_datasets(monkeypatch, tmp_path: Path) -> None:
    _patch_datasets(monkeypatch)
    for dataset in ("cora", "citeseer", "pubmed"):
        cfg = _write_config(tmp_path / f"{dataset}.yaml", dataset, "gcn", tmp_path / dataset)
        dense = _make_dense_checkpoint(tmp_path / f"{dataset}_dense.pt", "gcn")
        prune_from_checkpoint(str(dense), str(cfg))
        diag = json.loads((tmp_path / dataset / "pruning_diagnostics.json").read_text(encoding="utf-8"))
        assert diag["pruned_parameter_count"] < diag["dense_parameter_count"]


def test_metrics_saved_for_all_phases_all_datasets(monkeypatch, tmp_path: Path) -> None:
    _patch_datasets(monkeypatch)
    for dataset in ("cora", "citeseer", "pubmed"):
        cfg = _write_config(tmp_path / f"{dataset}.yaml", dataset, "gcn", tmp_path / dataset)
        dense = _make_dense_checkpoint(tmp_path / f"{dataset}_dense.pt", "gcn")
        artifacts = prune_from_checkpoint(str(dense), str(cfg))
        finetune_pruned_checkpoint(str(artifacts.pruned_checkpoint_path), str(cfg), finetune_epochs=1)
        diag = json.loads((tmp_path / dataset / "pruning_diagnostics.json").read_text(encoding="utf-8"))
        assert {"dense", "post_prune", "post_finetune"}.issubset(set(diag["metrics"]))


def test_graphsage_pubmed_not_identical_to_cora(monkeypatch, tmp_path: Path) -> None:
    _patch_datasets(monkeypatch)
    results = {}
    for dataset in ("cora", "pubmed"):
        cfg = _write_config(tmp_path / f"{dataset}.yaml", dataset, "graphsage", tmp_path / dataset)
        dense = _make_dense_checkpoint(tmp_path / f"{dataset}_dense.pt", "graphsage")
        artifacts = prune_from_checkpoint(str(dense), str(cfg))
        finetune_pruned_checkpoint(str(artifacts.pruned_checkpoint_path), str(cfg), finetune_epochs=1)
        diag = json.loads((tmp_path / dataset / "pruning_diagnostics.json").read_text(encoding="utf-8"))
        results[dataset] = diag["metrics"]["post_finetune"]["test"]["accuracy"]

    assert results["pubmed"] != results["cora"]

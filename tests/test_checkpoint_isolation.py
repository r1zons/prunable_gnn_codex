"""Checkpoint sharing/isolation tests for pruning experiments."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import torch
from torch_geometric.data import Data

from gnn_pruning.config import resolve_config
from gnn_pruning.models import build_model
from gnn_pruning.pipelines import run_pipeline
from gnn_pruning.training.workflow import train_dense


def _dummy_dataset():
    n = 36
    x = torch.randn((n, 8), dtype=torch.float32)
    y = torch.randint(0, 4, (n,), dtype=torch.long)
    edge_index = torch.vstack([torch.arange(n), torch.roll(torch.arange(n), shifts=-1)]).long()
    data = Data(x=x, y=y, edge_index=edge_index)

    class DummyDataset:
        def __getitem__(self, idx: int):
            _ = idx
            return data

    return DummyDataset()


def _patch_datasets(monkeypatch) -> None:
    training_module = importlib.import_module("gnn_pruning.training.workflow")
    pruning_module = importlib.import_module("gnn_pruning.pruning.workflow")
    dataset = _dummy_dataset()
    monkeypatch.setattr(training_module, "load_dataset", lambda name, root: dataset)
    monkeypatch.setattr(pruning_module, "load_dataset", lambda name, root: dataset)


def _write_config(path: Path, output_dir: Path, num_layers: int, hidden: int, methods: str = "[random, global_magnitude]") -> Path:
    path.write_text(
        "\n".join(
            [
                "base: base/default",
                "dataset: flickr",
                "model: graphsage",
                "preset: fast_debug",
                "model:",
                "  name: graphsage",
                f"  num_layers: {num_layers}",
                f"  hidden_channels: {hidden}",
                "run:",
                f"  output_dir: {output_dir.as_posix()}",
                "pruning:",
                f"  methods: {methods}",
                "  sparsity_levels: [0.5]",
                "  structured: true",
                "  finetune_epochs: 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_same_config_pruners_share_same_dense_checkpoint(monkeypatch, tmp_path: Path) -> None:
    _patch_datasets(monkeypatch)
    cfg = _write_config(tmp_path / "cfg.yaml", tmp_path / "run", num_layers=2, hidden=64)
    run_pipeline(str(cfg))

    paths = []
    for method in ("random", "global_magnitude"):
        metadata_path = tmp_path / "run" / "pruning" / method / "sparsity_0_5" / "run_metadata.json"
        records = json.loads(metadata_path.read_text(encoding="utf-8"))
        paths.extend(record["dense_checkpoint_path"] for record in records)
    assert len(set(paths)) == 1


def test_different_graphsage_architectures_do_not_share_dense_checkpoint(monkeypatch, tmp_path: Path) -> None:
    _patch_datasets(monkeypatch)
    cfg_a = _write_config(tmp_path / "cfg_a.yaml", tmp_path / "run_a", num_layers=2, hidden=64, methods="[random]")
    cfg_b = _write_config(tmp_path / "cfg_b.yaml", tmp_path / "run_b", num_layers=3, hidden=128, methods="[random]")
    art_a = run_pipeline(str(cfg_a))
    art_b = run_pipeline(str(cfg_b))
    assert art_a.dense_checkpoint_path != art_b.dense_checkpoint_path


def test_incompatible_checkpoint_is_rejected(monkeypatch, tmp_path: Path) -> None:
    _patch_datasets(monkeypatch)
    shared_out = tmp_path / "shared"
    cfg_a = _write_config(tmp_path / "cfg_a.yaml", shared_out, num_layers=2, hidden=64, methods="[random]")
    cfg_b = _write_config(tmp_path / "cfg_b.yaml", shared_out, num_layers=3, hidden=128, methods="[random]")

    first = train_dense(str(cfg_a), resume=True, output_dir_override=str(shared_out))
    assert first.checkpoint_reused is False
    second = train_dense(str(cfg_b), resume=True, output_dir_override=str(shared_out))
    assert second.checkpoint_reused is False


def test_run_directories_are_unique_for_architecture_variants() -> None:
    paths = []
    for config_path in [
        "configs/experiments/flickr_graphsage_l2_h64.yaml",
        "configs/experiments/flickr_graphsage_l2_h128.yaml",
        "configs/experiments/flickr_graphsage_l3_h64.yaml",
        "configs/experiments/flickr_graphsage_l3_h128.yaml",
        "configs/experiments/flickr_graphsage_l4_h128.yaml",
    ]:
        resolved = resolve_config(config_path)
        paths.append(resolved.run.output_dir)
    assert len(set(paths)) == len(paths)


def test_model_parameter_count_differs_across_architecture_configs() -> None:
    counts = []
    for config_path in [
        "configs/experiments/flickr_graphsage_l2_h64.yaml",
        "configs/experiments/flickr_graphsage_l3_h128.yaml",
        "configs/experiments/flickr_graphsage_l4_h128.yaml",
    ]:
        resolved = resolve_config(config_path)
        model = build_model(
            resolved.model.name,
            in_channels=16,
            hidden_channels=resolved.model.hidden_channels,
            out_channels=7,
            num_layers=resolved.model.num_layers,
            dropout=resolved.model.dropout,
        )
        counts.append(sum(parameter.numel() for parameter in model.parameters()))
    assert len(set(counts)) == len(counts)

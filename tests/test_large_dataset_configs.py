"""Config/script checks for large-dataset and Flickr GraphSAGE sweep experiments."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from gnn_pruning.config import load_yaml, resolve_config


def test_flickr_config_resolves() -> None:
    cfg = resolve_config("configs/experiments/presentation_flickr.yaml")
    raw = load_yaml("configs/experiments/presentation_flickr.yaml")
    pruning = raw.get("pruning", {})
    assert cfg.data.name == "flickr"
    assert cfg.model.name == "gcn"
    assert pruning["methods"] == ["random", "global_magnitude"]
    assert pruning["sparsity_levels"] == [0.5, 0.9]


def test_reddit_config_resolves() -> None:
    cfg = resolve_config("configs/experiments/presentation_reddit.yaml")
    raw = load_yaml("configs/experiments/presentation_reddit.yaml")
    pruning = raw.get("pruning", {})
    assert cfg.data.name == "reddit"
    assert cfg.model.name == "gcn"
    assert pruning["methods"] == ["random", "global_magnitude"]


def test_flickr_graphsage_sweep_configs_resolve() -> None:
    expectations = {
        "configs/experiments/flickr_graphsage_l2_h64.yaml": (2, 64),
        "configs/experiments/flickr_graphsage_l2_h128.yaml": (2, 128),
        "configs/experiments/flickr_graphsage_l3_h64.yaml": (3, 64),
        "configs/experiments/flickr_graphsage_l3_h128.yaml": (3, 128),
        "configs/experiments/flickr_graphsage_l4_h128.yaml": (4, 128),
    }
    for path, (layers, hidden) in expectations.items():
        cfg = resolve_config(path)
        raw = load_yaml(path)
        pruning = raw.get("pruning", {})
        assert cfg.data.name == "flickr"
        assert cfg.model.name == "graphsage"
        assert cfg.model.num_layers == layers
        assert cfg.model.hidden_channels == hidden
        assert pruning["methods"] == ["random", "global_magnitude"]
        assert pruning["sparsity_levels"] == [0.5, 0.9]


def test_large_dataset_scripts_exist() -> None:
    scripts = [
        Path("scripts/run_large_datasets.py"),
        Path("scripts/run_flickr_graphsage_sweep.py"),
        Path("scripts/summarize_flickr_graphsage_sweep.py"),
    ]
    for script in scripts:
        assert script.exists()
        spec = importlib.util.spec_from_file_location(script.stem, script)
        assert spec is not None
        assert spec.loader is not None


def test_flickr_config_light_smoke() -> None:
    cfg = resolve_config("configs/experiments/flickr_graphsage_l2_h64.yaml")
    assert cfg.training.epochs > 0

"""Tests for Flickr pruning study configs and scripts."""

from __future__ import annotations

from pathlib import Path

from gnn_pruning.config import resolve_config
from gnn_pruning.reporting.csv_reporter import PIPELINE_RESULTS_COLUMNS


def test_flickr_readiness_script_importable() -> None:
    path = Path("scripts/check_flickr_readiness.py")
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "def main()" in text


def test_flickr_debug_configs_resolve() -> None:
    for path in [
        "configs/experiments/flickr_debug_graphsage_l2_h64.yaml",
        "configs/experiments/flickr_debug_graphsage_l2_h128.yaml",
        "configs/experiments/flickr_debug_graphsage_l3_h128.yaml",
    ]:
        cfg = resolve_config(path)
        assert cfg.data.name == "flickr"
        assert cfg.model.name == "graphsage"


def test_flickr_real_configs_resolve() -> None:
    for path in [
        "configs/experiments/flickr_graphsage_l2_h64.yaml",
        "configs/experiments/flickr_graphsage_l2_h128.yaml",
        "configs/experiments/flickr_graphsage_l3_h128.yaml",
        "configs/experiments/flickr_graphsage_l4_h128.yaml",
    ]:
        cfg = resolve_config(path)
        assert cfg.training.epochs >= 150


def test_flickr_model_hyperparams_match_filename() -> None:
    cfg_2_64 = resolve_config("configs/experiments/flickr_graphsage_l2_h64.yaml")
    cfg_3_128 = resolve_config("configs/experiments/flickr_graphsage_l3_h128.yaml")
    assert cfg_2_64.model.num_layers == 2 and cfg_2_64.model.hidden_channels == 64
    assert cfg_3_128.model.num_layers == 3 and cfg_3_128.model.hidden_channels == 128


def test_postprune_only_config_resolves() -> None:
    for path in [
        "configs/experiments/flickr_postprune_only_graphsage_l2_h128.yaml",
        "configs/experiments/flickr_postprune_only_graphsage_l3_h128.yaml",
    ]:
        cfg = resolve_config(path)
        assert cfg.data.name == "flickr"


def test_result_schema_contains_phase() -> None:
    assert "phase" in PIPELINE_RESULTS_COLUMNS
    assert {"dense", "post_prune", "post_finetune", "skipped_finetune"}

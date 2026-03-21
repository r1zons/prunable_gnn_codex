"""Unit tests for config loading, merging, and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from gnn_pruning.config import load_yaml, resolve_config
from gnn_pruning.config.loader import deep_merge
from gnn_pruning.pipelines import run_pipeline


def test_load_yaml_reads_mapping(tmp_path: Path) -> None:
    config_file = tmp_path / "simple.yaml"
    config_file.write_text("run:\n  seed: 123\n", encoding="utf-8")

    payload = load_yaml(config_file)

    assert payload["run"]["seed"] == 123


def test_deep_merge_prefers_right_side() -> None:
    left = {"training": {"epochs": 100, "lr": 0.01}, "run": {"seed": 42}}
    right = {"training": {"epochs": 5}, "run": {"output_dir": "runs/x"}}

    merged = deep_merge(left, right)

    assert merged["training"]["epochs"] == 5
    assert merged["training"]["lr"] == 0.01
    assert merged["run"]["seed"] == 42
    assert merged["run"]["output_dir"] == "runs/x"


def test_resolve_config_merges_layers_and_user_override(tmp_path: Path) -> None:
    user_config = tmp_path / "experiment.yaml"
    user_config.write_text(
        "\n".join(
            [
                "base: base/default",
                "dataset: citeseer",
                "model: graphsage",
                "preset: fast_debug",
                "training:",
                "  epochs: 7",
                "run:",
                "  output_dir: runs/test_output",
            ]
        ),
        encoding="utf-8",
    )

    resolved = resolve_config(user_config)

    assert resolved.data.name == "citeseer"
    assert resolved.model.name == "graphsage"
    assert resolved.training.epochs == 7
    assert resolved.run.experiment_name == "fast_debug"
    assert resolved.run.output_dir == "runs/test_output"


def test_run_pipeline_saves_resolved_snapshot(tmp_path: Path) -> None:
    output_dir = tmp_path / "artifacts"
    user_config = tmp_path / "experiment.yaml"
    user_config.write_text(
        "\n".join(
            [
                "base: base/default",
                "dataset: cora",
                "model: gcn",
                "preset: standard",
                f"run:\n  output_dir: {output_dir.as_posix()}",
            ]
        ),
        encoding="utf-8",
    )

    snapshot = run_pipeline(str(user_config))

    assert snapshot.exists()
    loaded = load_yaml(snapshot)
    assert loaded["data"]["name"] == "cora"
    assert loaded["model"]["name"] == "gcn"


def test_invalid_split_ratio_raises(tmp_path: Path) -> None:
    user_config = tmp_path / "bad.yaml"
    user_config.write_text(
        "\n".join(
            [
                "base: base/default",
                "data:",
                "  train_ratio: 0.8",
                "  val_ratio: 0.3",
                "  test_ratio: 0.2",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sum to 1.0"):
        resolve_config(user_config)

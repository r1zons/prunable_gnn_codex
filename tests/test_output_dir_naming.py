"""Tests for run output directory naming and collision handling."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from gnn_pruning.utils.output_dir import resolve_output_dir


def test_different_configs_produce_different_directories(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    fixed = datetime(2026, 3, 22, 20, 30)

    dir_a = resolve_output_dir(
        configured_output_dir="runs/default",
        experiment_name="fast_debug",
        dataset_name="cora",
        model_name="gcn",
        seed=42,
        resume=False,
        now=fixed,
    )
    dir_b = resolve_output_dir(
        configured_output_dir="runs/default",
        experiment_name="fast_debug_graphsage",
        dataset_name="cora",
        model_name="graphsage",
        seed=42,
        resume=False,
        now=fixed,
    )

    assert dir_a != dir_b


def test_gcn_and_graphsage_runs_do_not_collide(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    fixed = datetime(2026, 3, 22, 20, 43)

    gcn_dir = resolve_output_dir(
        configured_output_dir="runs/default",
        experiment_name="fast_debug",
        dataset_name="cora",
        model_name="gcn",
        seed=42,
        resume=False,
        now=fixed,
    )
    sage_dir = resolve_output_dir(
        configured_output_dir="runs/default",
        experiment_name="fast_debug",
        dataset_name="cora",
        model_name="graphsage",
        seed=42,
        resume=False,
        now=fixed,
    )

    assert gcn_dir != sage_dir


def test_multiple_runs_append_index_to_avoid_overwrite(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    fixed = datetime(2026, 3, 22, 20, 30)

    first = resolve_output_dir(
        configured_output_dir="runs/default",
        experiment_name="fast_debug",
        dataset_name="cora",
        model_name="gcn",
        seed=42,
        resume=False,
        now=fixed,
    )
    first.mkdir(parents=True)

    second = resolve_output_dir(
        configured_output_dir="runs/default",
        experiment_name="fast_debug",
        dataset_name="cora",
        model_name="gcn",
        seed=42,
        resume=False,
        now=fixed,
    )

    assert second != first
    assert second.name.endswith("_001")


def test_resume_uses_existing_directory(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    fixed = datetime(2026, 3, 22, 20, 30)

    existing = resolve_output_dir(
        configured_output_dir="runs/default",
        experiment_name="fast_debug",
        dataset_name="cora",
        model_name="gcn",
        seed=42,
        resume=False,
        now=fixed,
    )
    existing.mkdir(parents=True)

    resumed = resolve_output_dir(
        configured_output_dir="runs/default",
        experiment_name="fast_debug",
        dataset_name="cora",
        model_name="gcn",
        seed=42,
        resume=True,
        now=fixed,
    )

    assert resumed == existing

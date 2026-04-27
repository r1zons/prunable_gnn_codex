"""Lightweight audit checks for core experiment correctness assumptions."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from gnn_pruning.config import resolve_config
from gnn_pruning.reporting.csv_reporter import DENSE_RESULTS_COLUMNS, PIPELINE_RESULTS_COLUMNS


def _load_audit_module():
    module_path = Path("scripts/audit_experiment_correctness.py")
    spec = importlib.util.spec_from_file_location("audit_experiment_correctness", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_audit_configs_resolve() -> None:
    left = resolve_config("configs/experiments/audit_graphsage_cora_l2_h32.yaml")
    right = resolve_config("configs/experiments/audit_graphsage_cora_l3_h64.yaml")

    assert left.data.name == "cora"
    assert right.data.name == "cora"
    assert left.model.name == "graphsage"
    assert right.model.name == "graphsage"


def test_graphsage_audit_configs_build_different_models() -> None:
    pytest.importorskip("torch")
    from gnn_pruning.models import build_model

    left = resolve_config("configs/experiments/audit_graphsage_cora_l2_h32.yaml")
    right = resolve_config("configs/experiments/audit_graphsage_cora_l3_h64.yaml")

    model_left = build_model(
        left.model.name,
        in_channels=16,
        hidden_channels=left.model.hidden_channels,
        out_channels=7,
        num_layers=left.model.num_layers,
        dropout=left.model.dropout,
    )
    model_right = build_model(
        right.model.name,
        in_channels=16,
        hidden_channels=right.model.hidden_channels,
        out_channels=7,
        num_layers=right.model.num_layers,
        dropout=right.model.dropout,
    )

    count_left = sum(parameter.numel() for parameter in model_left.parameters())
    count_right = sum(parameter.numel() for parameter in model_right.parameters())

    assert count_left != count_right


def test_audit_script_detects_distinct_run_paths() -> None:
    pytest.importorskip("torch")
    module = _load_audit_module()
    records = module.audit_configs(
        [
            "configs/experiments/audit_graphsage_cora_l2_h32.yaml",
            "configs/experiments/audit_graphsage_cora_l3_h64.yaml",
        ]
    )

    assert len(records) == 2
    assert records[0].run_dir != records[1].run_dir
    assert records[0].dense_checkpoint_path != records[1].dense_checkpoint_path


def test_audit_script_importable() -> None:
    pytest.importorskip("torch")
    module = _load_audit_module()
    assert hasattr(module, "main")
    assert hasattr(module, "audit_configs")


def test_result_identity_fields_present_in_csv_schemas() -> None:
    required_identity = {
        "dataset",
        "model",
        "num_layers",
        "hidden_channels",
        "phase",
        "method",
        "sparsity",
        "run_dir",
    }
    assert required_identity.issubset(set(DENSE_RESULTS_COLUMNS))
    assert required_identity.issubset(set(PIPELINE_RESULTS_COLUMNS))

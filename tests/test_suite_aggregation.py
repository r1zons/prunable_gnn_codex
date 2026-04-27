"""Tests for suite aggregation and CSV schemas."""

from __future__ import annotations

import csv
from pathlib import Path

from gnn_pruning.pipelines.suite import aggregate_suite_rows
from gnn_pruning.reporting.csv_reporter import write_suite_run_rows


def test_aggregate_suite_rows_computes_mean_std_ci95() -> None:
    rows = [
        {
            "suite_name": "default_small",
            "experiment_name": "pipeline_citeseer_graphsage",
            "dataset": "citeseer",
            "model": "graphsage",
            "phase": "post_finetune",
            "pruning_method": "random",
            "requested_sparsity": "0.5",
            "test_accuracy": "0.60",
            "test_macro_f1": "0.50",
        },
        {
            "suite_name": "default_small",
            "experiment_name": "pipeline_citeseer_graphsage",
            "dataset": "citeseer",
            "model": "graphsage",
            "phase": "post_finetune",
            "pruning_method": "random",
            "requested_sparsity": "0.5",
            "test_accuracy": "0.80",
            "test_macro_f1": "0.70",
        },
    ]

    aggregate = aggregate_suite_rows(rows)
    assert len(aggregate) == 1
    row = aggregate[0]
    assert row["num_runs"] == 2
    for field in ("num_layers", "hidden_channels", "seed", "method", "sparsity", "config_hash", "run_dir"):
        assert field in row
    assert abs(float(row["test_accuracy_mean"]) - 0.7) < 1e-9
    assert row["test_accuracy_std"] != ""
    assert row["test_accuracy_ci95"] != ""
    assert abs(float(row["test_macro_f1_mean"]) - 0.6) < 1e-9


def test_suite_run_csv_schema(tmp_path: Path) -> None:
    csv_path = write_suite_run_rows(
        [
            {
                "suite_name": "default_small",
                "run_index": 0,
                "run_seed": 42,
                "experiment_name": "pipeline_citeseer_graphsage",
                "dataset": "citeseer",
                "model": "graphsage",
                "phase": "dense",
                "pruning_method": "dense",
                "requested_sparsity": 0.0,
                "achieved_sparsity": 0.0,
                "test_accuracy": 0.75,
                "test_macro_f1": 0.74,
                "pipeline_csv_path": "runs/a/pipeline_results.csv",
            }
        ],
        tmp_path / "suite_runs.csv",
    )
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        rows = list(reader)

    required = {
        "suite_name",
        "run_index",
        "run_seed",
        "experiment_name",
        "dataset",
        "model",
        "num_layers",
        "hidden_channels",
        "seed",
        "phase",
        "method",
        "sparsity",
        "pruning_method",
        "requested_sparsity",
        "achieved_sparsity",
        "config_hash",
        "run_dir",
        "test_accuracy",
        "test_macro_f1",
        "pipeline_csv_path",
    }
    assert required.issubset(set(header))
    assert len(rows) == 1


def test_suite_run_csv_rows_match_header_width(tmp_path: Path) -> None:
    csv_path = write_suite_run_rows(
        [
            {
                "suite_name": "default_small",
                "run_index": 0,
                "run_seed": 42,
                "experiment_name": "pipeline_citeseer_graphsage",
                "dataset": "citeseer",
                "model": "graphsage",
                "num_layers": 2,
                "hidden_channels": 128,
                "seed": 42,
                "phase": "post_finetune",
                "method": "global_magnitude",
                "sparsity": 0.5,
                "pruning_method": "global_magnitude",
                "requested_sparsity": 0.5,
                "achieved_sparsity": 0.5,
                "config_hash": "abc123",
                "run_dir": "runs/example",
                "test_accuracy": 0.8,
                "test_macro_f1": 0.79,
                "pipeline_csv_path": "runs/example/pipeline_results.csv",
            },
            {
                "suite_name": "default_small",
                "run_index": 1,
                "run_seed": 43,
                "experiment_name": "pipeline_citeseer_graphsage",
                "dataset": "citeseer",
                "model": "graphsage",
                "num_layers": 2,
                "hidden_channels": 128,
                "seed": 43,
                "phase": "post_prune",
                "method": "layerwise_magnitude",
                "sparsity": 0.5,
                "pruning_method": "layerwise_magnitude",
                "requested_sparsity": 0.5,
                "achieved_sparsity": 0.52,
                "config_hash": "def456",
                "run_dir": "runs/example2",
                "test_accuracy": 0.74,
                "test_macro_f1": 0.7,
                "pipeline_csv_path": "runs/example2/pipeline_results.csv",
            },
        ],
        tmp_path / "suite_runs.csv",
    )
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    assert rows
    header_len = len(rows[0])
    assert header_len > 0
    assert all(len(row) == header_len for row in rows[1:])


def test_suite_run_csv_schema_mismatch_resets_file(tmp_path: Path) -> None:
    csv_path = tmp_path / "suite_runs.csv"
    csv_path.write_text("suite_name,phase,pruning_method\nold,post_prune,random\n", encoding="utf-8")
    write_suite_run_rows(
        [
            {
                "suite_name": "default_small",
                "run_index": 0,
                "run_seed": 42,
                "experiment_name": "pipeline_citeseer_graphsage",
                "dataset": "citeseer",
                "model": "graphsage",
                "phase": "dense",
                "pruning_method": "dense",
                "requested_sparsity": 0.0,
                "achieved_sparsity": 0.0,
                "test_accuracy": 0.75,
                "test_macro_f1": 0.74,
                "pipeline_csv_path": "runs/a/pipeline_results.csv",
            }
        ],
        csv_path,
    )
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    assert len(rows[0]) > 3
    assert "run_index" in rows[0]


def test_aggregate_separates_rows_by_dataset() -> None:
    rows = [
        {
            "suite_name": "default_medium",
            "experiment_name": "presentation_flickr",
            "dataset": "flickr",
            "model": "gcn",
            "phase": "post_finetune",
            "pruning_method": "random",
            "requested_sparsity": "0.5",
            "test_accuracy": "0.80",
            "test_macro_f1": "0.75",
        },
        {
            "suite_name": "default_medium",
            "experiment_name": "presentation_reddit",
            "dataset": "reddit",
            "model": "gcn",
            "phase": "post_finetune",
            "pruning_method": "random",
            "requested_sparsity": "0.5",
            "test_accuracy": "0.60",
            "test_macro_f1": "0.55",
        },
    ]
    aggregate = aggregate_suite_rows(rows)
    assert len(aggregate) == 2
    datasets = {row["dataset"] for row in aggregate}
    assert datasets == {"flickr", "reddit"}

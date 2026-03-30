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
        "phase",
        "pruning_method",
        "requested_sparsity",
        "achieved_sparsity",
        "test_accuracy",
        "test_macro_f1",
        "pipeline_csv_path",
    }
    assert required.issubset(set(header))
    assert len(rows) == 1


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

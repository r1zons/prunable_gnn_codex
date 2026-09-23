"""Tests for suite aggregation and CSV schemas."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from gnn_pruning.pipelines.suite import _stable_condition_id, aggregate_suite_rows
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


def test_suite_aggregation_preserves_timing_columns() -> None:
    rows = [
        {
            "suite_name": "default_small",
            "experiment_name": "pipeline_pubmed_graphsage",
            "dataset": "pubmed",
            "model": "graphsage",
            "phase": "post_prune",
            "pruning_method": "random",
            "requested_sparsity": "0.7",
            "test_accuracy": "0.80",
            "test_macro_f1": "0.78",
            "inference_time_mean_ms": "1.50",
            "parameter_count": "1000",
        },
        {
            "suite_name": "default_small",
            "experiment_name": "pipeline_pubmed_graphsage",
            "dataset": "pubmed",
            "model": "graphsage",
            "phase": "post_prune",
            "pruning_method": "random",
            "requested_sparsity": "0.7",
            "test_accuracy": "0.70",
            "test_macro_f1": "0.68",
            "inference_time_mean_ms": "2.50",
            "parameter_count": "800",
        },
    ]

    aggregate = aggregate_suite_rows(rows)
    assert len(aggregate) == 1
    row = aggregate[0]
    assert "inference_time_mean_ms_mean" in row
    assert "inference_time_mean_ms_std" in row
    assert "parameter_count_mean" in row
    assert "parameter_count_std" in row
    assert abs(float(row["inference_time_mean_ms_mean"]) - 2.0) < 1e-9


def test_suite_aggregation_combines_independent_seeds_despite_varying_outcomes_and_paths() -> None:
    rows = []
    for seed, achieved, accuracy, stop_reason in [
        (42, 0.61, 0.79, "agent_stop"),
        (43, 0.68, 0.77, "target_sparsity_reached"),
        (44, 0.65, 0.78, "agent_stop"),
    ]:
        rows.append(
            {
                "suite_name": "compact",
                "condition_id": "stable-condition",
                "experiment_name": "pubmed_graphsage_l3_h16",
                "dataset": "pubmed",
                "model": "graphsage",
                "num_layers": "3",
                "hidden_channels": "16",
                "run_seed": seed,
                "seed": seed,
                "phase": "post_prune",
                "method": "q_learning_tabular",
                "pruning_method": "q_learning_tabular",
                "requested_sparsity": "0.7",
                "achieved_sparsity": achieved,
                "stop_reason": stop_reason,
                "config_hash": f"seed-specific-{seed}",
                "run_dir": f"runs/compact/run_{seed}",
                "val_accuracy": accuracy + 0.01,
                "test_accuracy": accuracy,
                "test_macro_f1": accuracy - 0.02,
                "parameter_count": 1000 - seed,
            }
        )

    aggregate = aggregate_suite_rows(rows)

    assert len(aggregate) == 1
    row = aggregate[0]
    assert row["num_observations"] == 3
    assert row["num_runs"] == 3
    assert row["aggregation_status"] == "multi_run"
    assert abs(float(row["achieved_sparsity_mean"]) - (0.61 + 0.68 + 0.65) / 3) < 1e-12
    assert row["achieved_sparsity_std"] != ""
    assert row["achieved_sparsity_ci95"] != ""
    assert row["val_accuracy_mean"] != ""
    assert json.loads(row["stop_reason_counts"]) == {"agent_stop": 2, "target_sparsity_reached": 1}
    assert json.loads(row["seeds"]) == ["42", "43", "44"]
    assert len(json.loads(row["run_dirs"])) == 3


def test_suite_aggregation_keeps_materially_different_conditions_separate() -> None:
    common = {
        "suite_name": "compact",
        "experiment_name": "shared_name",
        "dataset": "pubmed",
        "model": "graphsage",
        "num_layers": "3",
        "hidden_channels": "16",
        "phase": "post_prune",
        "method": "q_learning_tabular",
        "pruning_method": "q_learning_tabular",
        "requested_sparsity": "0.7",
        "test_accuracy": "0.75",
    }
    rows = [
        {**common, "condition_id": "max-drop-005", "run_seed": 42},
        {**common, "condition_id": "max-drop-010", "run_seed": 42},
    ]

    aggregate = aggregate_suite_rows(rows)

    assert len(aggregate) == 2
    assert {row["condition_id"] for row in aggregate} == {"max-drop-005", "max-drop-010"}


def test_suite_aggregation_marks_single_run_without_dispersion_estimates() -> None:
    aggregate = aggregate_suite_rows(
        [
            {
                "suite_name": "single",
                "experiment_name": "one_seed",
                "dataset": "cora",
                "model": "gcn",
                "phase": "dense",
                "method": "dense",
                "requested_sparsity": "0.0",
                "run_seed": 42,
                "test_accuracy": "0.8",
            }
        ]
    )

    assert aggregate[0]["aggregation_status"] == "single_run"
    assert aggregate[0]["num_runs"] == 1
    assert aggregate[0]["test_accuracy_std"] == ""
    assert aggregate[0]["test_accuracy_ci95"] == ""


def test_condition_id_ignores_seed_and_output_path_but_keeps_material_hyperparameters(tmp_path: Path) -> None:
    config_a = tmp_path / "a.yaml"
    config_b = tmp_path / "b.yaml"
    config_c = tmp_path / "c.yaml"
    config_a.write_text(
        "run:\n  seed: 42\n  output_dir: runs/a\nmodel:\n  hidden_channels: 16\nq_learning:\n  max_accuracy_drop: 0.05\n",
        encoding="utf-8",
    )
    config_b.write_text(
        "run:\n  seed: 44\n  output_dir: runs/b\nmodel:\n  hidden_channels: 16\nq_learning:\n  max_accuracy_drop: 0.05\n",
        encoding="utf-8",
    )
    config_c.write_text(
        "run:\n  seed: 42\n  output_dir: runs/c\nmodel:\n  hidden_channels: 16\nq_learning:\n  max_accuracy_drop: 0.10\n",
        encoding="utf-8",
    )

    assert _stable_condition_id(str(config_a)) == _stable_condition_id(str(config_b))
    assert _stable_condition_id(str(config_a)) != _stable_condition_id(str(config_c))

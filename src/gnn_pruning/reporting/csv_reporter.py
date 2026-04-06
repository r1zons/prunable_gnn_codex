"""CSV reporting utilities."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Union

DENSE_RESULTS_COLUMNS = [
    "experiment_name",
    "dataset",
    "model",
    "seed",
    "train_accuracy",
    "train_macro_f1",
    "val_accuracy",
    "val_macro_f1",
    "test_accuracy",
    "test_macro_f1",
    "best_epoch",
    "epochs_ran",
    "checkpoint_path",
    "metrics_path",
    "config_path",
    "split_path",
]

PIPELINE_RESULTS_COLUMNS = [
    "experiment_name",
    "dataset",
    "model",
    "seed",
    "phase",
    "pruning_method",
    "requested_sparsity",
    "achieved_sparsity",
    "train_accuracy",
    "train_macro_f1",
    "val_accuracy",
    "val_macro_f1",
    "test_accuracy",
    "test_macro_f1",
    "inference_time_mean_ms",
    "inference_time_std_ms",
    "parameter_count",
    "checkpoint_path",
    "metrics_path",
    "config_path",
    "split_path",
]

SUITE_RUN_COLUMNS = [
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
]

SUITE_AGGREGATE_COLUMNS = [
    "suite_name",
    "experiment_name",
    "dataset",
    "model",
    "phase",
    "pruning_method",
    "requested_sparsity",
    "num_runs",
    "test_accuracy_mean",
    "test_accuracy_std",
    "test_accuracy_ci95",
    "test_macro_f1_mean",
    "test_macro_f1_std",
    "test_macro_f1_ci95",
]


def write_csv_row(metrics: Mapping[str, object], csv_path: Union[str, Path]) -> Path:
    """Append one metrics row to CSV, creating schema header if needed."""
    target = Path(csv_path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)

    row: Dict[str, object] = {column: metrics.get(column, "") for column in DENSE_RESULTS_COLUMNS}
    file_exists = target.exists()

    with target.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DENSE_RESULTS_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    return target


def csv_columns() -> Iterable[str]:
    """Expose dense CSV schema columns for tests and validation."""
    return list(DENSE_RESULTS_COLUMNS)


def write_pipeline_csv_rows(rows: Sequence[Mapping[str, object]], csv_path: Union[str, Path]) -> Path:
    """Append multiple pipeline rows to CSV, creating header if needed."""
    target = Path(csv_path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    file_exists = target.exists()

    with target.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PIPELINE_RESULTS_COLUMNS)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            serialized = {column: row.get(column, "") for column in PIPELINE_RESULTS_COLUMNS}
            writer.writerow(serialized)

    return target


def write_suite_run_rows(rows: Sequence[Mapping[str, object]], csv_path: Union[str, Path]) -> Path:
    """Write suite run-level rows."""
    return _write_rows(rows, csv_path, SUITE_RUN_COLUMNS)


def write_suite_aggregate_rows(rows: Sequence[Mapping[str, object]], csv_path: Union[str, Path]) -> Path:
    """Write suite aggregate rows."""
    return _write_rows(rows, csv_path, SUITE_AGGREGATE_COLUMNS)


def _write_rows(rows: Sequence[Mapping[str, object]], csv_path: Union[str, Path], columns: Sequence[str]) -> Path:
    target = Path(csv_path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    file_exists = target.exists()

    with target.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        if not file_exists:
            writer.writeheader()
        for row in rows:
            serialized = {column: row.get(column, "") for column in columns}
            writer.writerow(serialized)
    return target

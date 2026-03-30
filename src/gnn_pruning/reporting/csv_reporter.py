"""CSV reporting utilities."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, Mapping, Union

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

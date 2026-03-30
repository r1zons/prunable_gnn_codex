"""Multi-run suite orchestration and aggregate reporting."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from gnn_pruning.config import dump_yaml, load_yaml
from gnn_pruning.reporting import write_suite_aggregate_rows, write_suite_run_rows

from .run_pipeline import run_pipeline


@dataclass
class SuiteArtifacts:
    """Artifacts produced by `run-suite`."""

    output_dir: Path
    runs_csv_path: Path
    aggregate_csv_path: Path


def run_suite(config_path: str) -> SuiteArtifacts:
    """Run repeated pipeline benchmarks and export aggregate metrics."""
    suite_cfg = load_yaml(config_path)
    run_cfg = suite_cfg.get("run", {}) if isinstance(suite_cfg.get("run", {}), dict) else {}
    suite_name = str(suite_cfg.get("suite_name", Path(config_path).stem))
    num_runs = int(run_cfg.get("num_runs", 1))
    base_seed = int(run_cfg.get("base_seed", 42))
    output_dir = Path(str(run_cfg.get("output_dir", f"runs/suites/{suite_name}"))).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    experiment_refs = _resolve_experiment_configs(suite_cfg)
    run_rows: List[Dict[str, Any]] = []

    for run_index in range(num_runs):
        run_seed = base_seed + run_index
        run_dir = output_dir / f"run_{run_index:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        for experiment_config in experiment_refs:
            pipeline_config = _build_run_config(
                source_config=experiment_config,
                destination=run_dir / f"{Path(experiment_config).stem}.yaml",
                run_seed=run_seed,
                output_dir=run_dir / Path(experiment_config).stem,
            )
            pipeline_artifacts = run_pipeline(str(pipeline_config))
            run_rows.extend(
                _load_pipeline_rows(
                    suite_name=suite_name,
                    run_index=run_index,
                    run_seed=run_seed,
                    pipeline_csv_path=pipeline_artifacts.csv_path,
                )
            )

    runs_csv_path = write_suite_run_rows(run_rows, output_dir / "suite_runs.csv")
    aggregate_rows = aggregate_suite_rows(run_rows)
    aggregate_csv_path = write_suite_aggregate_rows(aggregate_rows, output_dir / "suite_aggregate.csv")
    return SuiteArtifacts(output_dir=output_dir, runs_csv_path=runs_csv_path, aggregate_csv_path=aggregate_csv_path)


def aggregate_suite_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate run-level rows into mean/std/95% CI summaries."""
    grouped: Dict[Tuple[str, ...], List[Dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row.get("suite_name", "")),
            str(row.get("experiment_name", "")),
            str(row.get("dataset", "")),
            str(row.get("model", "")),
            str(row.get("phase", "")),
            str(row.get("pruning_method", "")),
            str(row.get("requested_sparsity", "")),
        )
        grouped.setdefault(key, []).append(row)

    aggregate_rows: List[Dict[str, Any]] = []
    for key, members in grouped.items():
        suite_name, experiment_name, dataset, model, phase, pruning_method, requested_sparsity = key
        test_accuracy_values = _extract_numeric(members, "test_accuracy")
        test_macro_f1_values = _extract_numeric(members, "test_macro_f1")
        aggregate_rows.append(
            {
                "suite_name": suite_name,
                "experiment_name": experiment_name,
                "dataset": dataset,
                "model": model,
                "phase": phase,
                "pruning_method": pruning_method,
                "requested_sparsity": requested_sparsity,
                "num_runs": len(members),
                "test_accuracy_mean": _safe_mean(test_accuracy_values),
                "test_accuracy_std": _safe_std(test_accuracy_values),
                "test_accuracy_ci95": _ci95(test_accuracy_values),
                "test_macro_f1_mean": _safe_mean(test_macro_f1_values),
                "test_macro_f1_std": _safe_std(test_macro_f1_values),
                "test_macro_f1_ci95": _ci95(test_macro_f1_values),
            }
        )
    return aggregate_rows


def _safe_mean(values: Sequence[float]) -> float | str:
    return mean(values) if values else ""


def _safe_std(values: Sequence[float]) -> float | str:
    return stdev(values) if len(values) > 1 else ""


def _ci95(values: Sequence[float]) -> float | str:
    if len(values) <= 1:
        return ""
    std = stdev(values)
    return 1.96 * std / math.sqrt(len(values))


def _extract_numeric(rows: Iterable[Dict[str, Any]], key: str) -> List[float]:
    values: List[float] = []
    for row in rows:
        value = row.get(key, "")
        if value == "" or value is None:
            continue
        values.append(float(value))
    return values


def _resolve_experiment_configs(suite_cfg: Dict[str, Any]) -> List[str]:
    experiments = suite_cfg.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("Suite config must provide a non-empty 'experiments' list.")
    return [str(entry) for entry in experiments]


def _build_run_config(source_config: str, destination: Path, run_seed: int, output_dir: Path) -> Path:
    payload = load_yaml(source_config)
    payload["run"] = dict(payload.get("run", {}))
    payload["run"]["seed"] = int(run_seed)
    payload["run"]["output_dir"] = str(output_dir)
    dump_yaml(payload, destination)
    return destination


def _load_pipeline_rows(
    suite_name: str,
    run_index: int,
    run_seed: int,
    pipeline_csv_path: Path,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with pipeline_csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "suite_name": suite_name,
                    "run_index": run_index,
                    "run_seed": run_seed,
                    "experiment_name": row.get("experiment_name", ""),
                    "dataset": row.get("dataset", ""),
                    "model": row.get("model", ""),
                    "phase": row.get("phase", ""),
                    "pruning_method": row.get("pruning_method", ""),
                    "requested_sparsity": row.get("requested_sparsity", ""),
                    "achieved_sparsity": row.get("achieved_sparsity", ""),
                    "test_accuracy": row.get("test_accuracy", ""),
                    "test_macro_f1": row.get("test_macro_f1", ""),
                    "pipeline_csv_path": str(pipeline_csv_path),
                }
            )
    return rows

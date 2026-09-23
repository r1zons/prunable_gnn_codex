"""Multi-run suite orchestration and aggregate reporting."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from gnn_pruning.config import dump_yaml, load_yaml
from gnn_pruning.reporting import write_suite_aggregate_rows, write_suite_run_rows
from gnn_pruning.utils import ProgressReporter

from .run_pipeline import run_pipeline


@dataclass
class SuiteArtifacts:
    """Artifacts produced by `run-suite`."""

    output_dir: Path
    runs_csv_path: Path
    aggregate_csv_path: Path


def run_suite(config_path: str, show_progress: bool = False) -> SuiteArtifacts:
    """Run repeated pipeline benchmarks and export aggregate metrics."""
    suite_cfg = load_yaml(config_path)
    run_cfg = suite_cfg.get("run", {}) if isinstance(suite_cfg.get("run", {}), dict) else {}
    suite_name = str(suite_cfg.get("suite_name", Path(config_path).stem))
    num_runs = int(run_cfg.get("num_runs", 1))
    base_seed = int(run_cfg.get("base_seed", 42))
    output_dir = Path(str(run_cfg.get("output_dir", f"runs/suites/{suite_name}"))).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    reporter = ProgressReporter(enabled=show_progress, log_path=output_dir / "progress.log")

    experiment_refs = _resolve_experiment_configs(suite_cfg)
    run_rows: List[Dict[str, Any]] = []
    runs_csv_target = output_dir / "suite_runs.csv"
    aggregate_csv_target = output_dir / "suite_aggregate.csv"
    if runs_csv_target.exists():
        runs_csv_target.unlink()
    if aggregate_csv_target.exists():
        aggregate_csv_target.unlink()

    total_jobs = max(1, num_runs * len(experiment_refs))
    job_index = 0
    for run_index in range(num_runs):
        run_seed = base_seed + run_index
        run_dir = output_dir / f"run_{run_index:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        for experiment_config in experiment_refs:
            job_index += 1
            pipeline_config = _build_run_config(
                source_config=experiment_config,
                destination=run_dir / f"{Path(experiment_config).stem}.yaml",
                run_seed=run_seed,
                output_dir=run_dir / Path(experiment_config).stem,
            )
            reporter.info(
                f"[suite {job_index}/{total_jobs}] run={run_index + 1}/{num_runs} "
                f"config={Path(experiment_config).name} seed={run_seed}"
            )
            pipeline_artifacts = run_pipeline(str(pipeline_config), show_progress=show_progress)
            run_rows.extend(
                _load_pipeline_rows(
                    suite_name=suite_name,
                    run_index=run_index,
                    run_seed=run_seed,
                    pipeline_csv_path=pipeline_artifacts.csv_path,
                    condition_id=_stable_condition_id(experiment_config),
                )
            )

    runs_csv_path = write_suite_run_rows(run_rows, runs_csv_target)
    aggregate_rows = aggregate_suite_rows(run_rows)
    aggregate_csv_path = write_suite_aggregate_rows(aggregate_rows, aggregate_csv_target)
    return SuiteArtifacts(output_dir=output_dir, runs_csv_path=runs_csv_path, aggregate_csv_path=aggregate_csv_path)


def aggregate_suite_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate independent seeds by stable experimental condition."""
    grouped: Dict[Tuple[str, ...], List[Dict[str, Any]]] = {}
    for row in rows:
        method = str(row.get("method", row.get("pruning_method", "")))
        pruning_method = str(row.get("pruning_method", method))
        requested_sparsity = str(row.get("requested_sparsity", row.get("sparsity", "")))
        key = (
            str(row.get("suite_name", "")),
            str(row.get("condition_id", "")),
            str(row.get("experiment_name", "")),
            str(row.get("dataset", "")),
            str(row.get("model", "")),
            str(row.get("num_layers", "")),
            str(row.get("hidden_channels", "")),
            str(row.get("phase", "")),
            method,
            pruning_method,
            requested_sparsity,
        )
        grouped.setdefault(key, []).append(row)

    aggregate_rows: List[Dict[str, Any]] = []
    for key, members in grouped.items():
        (
            suite_name,
            condition_id,
            experiment_name,
            dataset,
            model,
            num_layers,
            hidden_channels,
            phase,
            method,
            pruning_method,
            requested_sparsity,
        ) = key
        run_keys = _independent_run_keys(members)
        num_runs = len(set(run_keys))
        stop_reason_counts = Counter(str(row.get("stop_reason", "")).strip() for row in members)
        stop_reason_counts.pop("", None)
        row_out: Dict[str, Any] = {
            "suite_name": suite_name,
            "condition_id": condition_id,
            "experiment_name": experiment_name,
            "dataset": dataset,
            "model": model,
            "num_layers": num_layers,
            "hidden_channels": hidden_channels,
            "seed": "",
            "seeds": json.dumps(_unique_values(members, "run_seed", fallback="seed")),
            "phase": phase,
            "method": method,
            "sparsity": requested_sparsity,
            "pruning_method": pruning_method,
            "requested_sparsity": requested_sparsity,
            "stop_reason": _single_value_or_blank(stop_reason_counts),
            "stop_reason_counts": json.dumps(dict(sorted(stop_reason_counts.items())), sort_keys=True),
            "config_hash": "",
            "config_hashes": json.dumps(_unique_values(members, "config_hash")),
            "run_dir": "",
            "run_dirs": json.dumps(_unique_values(members, "run_dir")),
            "num_observations": len(members),
            "num_runs": num_runs,
            "aggregation_status": "multi_run" if num_runs > 1 else "single_run",
        }
        for metric in (
            "achieved_sparsity",
            "final_reward",
            "num_adaptive_steps",
            "val_accuracy",
            "val_macro_f1",
            "test_accuracy",
            "test_macro_f1",
            "inference_time_mean_ms",
            "parameter_count",
        ):
            values = _extract_per_run_numeric(members, metric, run_keys)
            row_out[f"{metric}_mean"] = _safe_mean(values)
            row_out[f"{metric}_std"] = _safe_std(values)
            row_out[f"{metric}_ci95"] = _ci95(values)

        # Keep legacy outcome columns useful without implying they are grouping identifiers.
        row_out["achieved_sparsity"] = row_out["achieved_sparsity_mean"]
        row_out["final_reward"] = row_out["final_reward_mean"]
        row_out["num_adaptive_steps"] = row_out["num_adaptive_steps_mean"]
        aggregate_rows.append(row_out)
    return aggregate_rows


def _safe_mean(values: Sequence[float]) -> float | str:
    return mean(values) if values else ""


def _safe_std(values: Sequence[float]) -> float | str:
    return stdev(values) if len(values) > 1 else ""


def _ci95(values: Sequence[float]) -> float | str:
    if len(values) <= 1:
        return ""
    std = stdev(values)
    return _t_critical_95(len(values) - 1) * std / math.sqrt(len(values))


def _t_critical_95(degrees_of_freedom: int) -> float:
    """Return a two-sided 95% Student-t critical value without SciPy."""
    table = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        12: 2.179,
        15: 2.131,
        20: 2.086,
        25: 2.060,
        30: 2.042,
    }
    if degrees_of_freedom in table:
        return table[degrees_of_freedom]
    lower_degrees = [df for df in table if df < degrees_of_freedom]
    if lower_degrees and degrees_of_freedom <= 30:
        return table[max(lower_degrees)]
    return 1.96


def _extract_numeric(rows: Iterable[Dict[str, Any]], key: str) -> List[float]:
    values: List[float] = []
    for row in rows:
        value = row.get(key, "")
        if value == "" or value is None:
            continue
        values.append(float(value))
    return values


def _independent_run_keys(rows: Sequence[Dict[str, Any]]) -> List[str]:
    keys: List[str] = []
    for index, row in enumerate(rows):
        for field in ("run_seed", "seed", "run_index"):
            value = row.get(field, "")
            if value not in ("", None):
                keys.append(f"{field}:{value}")
                break
        else:
            keys.append(f"row:{index}")
    return keys


def _extract_per_run_numeric(
    rows: Sequence[Dict[str, Any]],
    key: str,
    run_keys: Sequence[str],
) -> List[float]:
    by_run: Dict[str, List[float]] = {}
    for row, run_key in zip(rows, run_keys):
        values = _extract_numeric([row], key)
        if values:
            by_run.setdefault(run_key, []).extend(values)
    return [mean(values) for values in by_run.values()]


def _unique_values(rows: Sequence[Dict[str, Any]], key: str, fallback: str = "") -> List[str]:
    values = {
        str(row.get(key, row.get(fallback, ""))).strip()
        for row in rows
        if row.get(key, row.get(fallback, "")) not in ("", None)
    }
    return sorted(values)


def _single_value_or_blank(counts: Counter[str]) -> str:
    return next(iter(counts)) if len(counts) == 1 else ""


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


def _stable_condition_id(config_path: str) -> str:
    """Hash experiment settings while excluding seed- and path-specific run fields."""
    payload = load_yaml(config_path)
    normalized = json.loads(json.dumps(payload))
    run_cfg = normalized.get("run")
    if isinstance(run_cfg, dict):
        run_cfg.pop("seed", None)
        run_cfg.pop("output_dir", None)
    serialized = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _load_pipeline_rows(
    suite_name: str,
    run_index: int,
    run_seed: int,
    pipeline_csv_path: Path,
    condition_id: str = "",
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with pipeline_csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "suite_name": suite_name,
                    "condition_id": condition_id,
                    "run_index": run_index,
                    "run_seed": run_seed,
                    "experiment_name": row.get("experiment_name", ""),
                    "dataset": row.get("dataset", ""),
                    "model": row.get("model", ""),
                    "num_layers": row.get("num_layers", ""),
                    "hidden_channels": row.get("hidden_channels", ""),
                    "seed": row.get("seed", ""),
                    "phase": row.get("phase", ""),
                    "method": row.get("method", row.get("pruning_method", "")),
                    "sparsity": row.get("sparsity", row.get("requested_sparsity", "")),
                    "pruning_method": row.get("pruning_method", ""),
                    "requested_sparsity": row.get("requested_sparsity", ""),
                    "achieved_sparsity": row.get("achieved_sparsity", ""),
                    "final_reward": row.get("final_reward", ""),
                    "num_adaptive_steps": row.get("num_adaptive_steps", ""),
                    "stop_reason": row.get("stop_reason", ""),
                    "config_hash": row.get("config_hash", ""),
                    "run_dir": row.get("run_dir", ""),
                    "test_accuracy": row.get("test_accuracy", ""),
                    "test_macro_f1": row.get("test_macro_f1", ""),
                    "val_accuracy": row.get("val_accuracy", ""),
                    "val_macro_f1": row.get("val_macro_f1", ""),
                    "inference_time_mean_ms": row.get("inference_time_mean_ms", ""),
                    "inference_time_std_ms": row.get("inference_time_std_ms", ""),
                    "parameter_count": row.get("parameter_count", ""),
                    "pipeline_csv_path": str(pipeline_csv_path),
                }
            )
    return rows

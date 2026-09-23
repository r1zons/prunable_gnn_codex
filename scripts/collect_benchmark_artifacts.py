"""Collect compact benchmark artifacts from selected suite run folders."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

DEFAULT_COMPLETED_RUNS = ("run_000", "run_001")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect compact pruning benchmark artifacts.")
    parser.add_argument("--runs-root", default="runs/compact_pruning_benchmark")
    parser.add_argument("--include-runs", nargs="*", default=None)
    parser.add_argument("--include-incomplete", action="store_true")
    parser.add_argument("--out-dir", default="exports/compact_pruning_benchmark_artifacts")
    parser.add_argument("--zip", action="store_true")
    parser.add_argument("--include-rl-trace", action="store_true")
    parser.add_argument("--include-q-table", action="store_true")
    parser.add_argument("--include-deployment-trace", action="store_true", default=True)
    parser.add_argument("--include-action-diagnostics", action="store_true", default=True)
    return parser


def _discover_run_dirs(runs_root: Path) -> Dict[str, Path]:
    return {path.name: path for path in sorted(runs_root.glob("run_*")) if path.is_dir()}


def _select_runs(
    *,
    run_dirs: Dict[str, Path],
    include_runs: Sequence[str] | None,
    include_incomplete: bool,
) -> List[str]:
    if include_runs:
        return [run for run in include_runs if run in run_dirs]
    if include_incomplete:
        return sorted(run_dirs.keys())
    return [run for run in DEFAULT_COMPLETED_RUNS if run in run_dirs]


def _copy_file(
    *,
    source: Path,
    destination_root: Path,
    relative_path: Path,
    manifest: List[Dict[str, Any]],
    artifact_type: str,
    benchmark_run_id: str,
) -> None:
    target = destination_root / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    manifest.append(
        {
            "artifact_type": artifact_type,
            "benchmark_run_id": benchmark_run_id,
            "source_path": str(source),
            "copied_path": str(target),
            "size_bytes": source.stat().st_size,
        }
    )


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _collect_global_files(runs_root: Path, out_dir: Path, manifest: List[Dict[str, Any]]) -> None:
    known = [runs_root / "suite_runs.csv", runs_root / "suite_aggregate.csv"]
    for path in known:
        if path.exists():
            _copy_file(
                source=path,
                destination_root=out_dir,
                relative_path=Path(path.name),
                manifest=manifest,
                artifact_type="suite_file",
                benchmark_run_id="suite",
            )
    for path in sorted(runs_root.glob("*.csv")):
        if path.name in {"suite_runs.csv", "suite_aggregate.csv"}:
            continue
        _copy_file(
            source=path,
            destination_root=out_dir,
            relative_path=Path(path.name),
            manifest=manifest,
            artifact_type="suite_csv",
            benchmark_run_id="suite",
        )


def _is_qlearning_artifact(path: Path, *, include_rl_trace: bool, include_q_table: bool, include_deployment_trace: bool, include_action_diagnostics: bool) -> bool:
    if path.name == "deployment_trace.json":
        return include_deployment_trace
    if path.name == "action_space_diagnostics.json":
        return include_action_diagnostics
    if path.name == "q_table.json":
        return include_q_table
    if path.name == "rl_trace.json":
        return include_rl_trace
    return False


def _collect_run_files(
    *,
    runs_root: Path,
    out_dir: Path,
    run_id: str,
    run_path: Path,
    include_rl_trace: bool,
    include_q_table: bool,
    include_deployment_trace: bool,
    include_action_diagnostics: bool,
    manifest: List[Dict[str, Any]],
) -> List[Path]:
    standard_names = {
        "pipeline_results.csv",
        "resolved_config.yaml",
        "summary_pipeline.md",
        "metrics_eval.json",
        "metrics_train.json",
    }
    pipeline_paths: List[Path] = []
    for path in sorted(run_path.rglob("*")):
        if not path.is_file():
            continue
        should_copy = path.name in standard_names or _is_qlearning_artifact(
            path,
            include_rl_trace=include_rl_trace,
            include_q_table=include_q_table,
            include_deployment_trace=include_deployment_trace,
            include_action_diagnostics=include_action_diagnostics,
        )
        if not should_copy:
            continue
        rel = path.relative_to(runs_root)
        artifact_type = "pipeline_file"
        if path.name in {"deployment_trace.json", "action_space_diagnostics.json", "q_table.json", "rl_trace.json"}:
            artifact_type = "q_learning_file"
        _copy_file(
            source=path,
            destination_root=out_dir,
            relative_path=rel,
            manifest=manifest,
            artifact_type=artifact_type,
            benchmark_run_id=run_id,
        )
        if path.name == "pipeline_results.csv":
            pipeline_paths.append(path)
    return pipeline_paths


def _infer_value(row: Dict[str, Any], key: str, source_pipeline_path: Path) -> str:
    value = str(row.get(key, "")).strip()
    if value:
        return value
    if key == "experiment_name":
        return source_pipeline_path.parent.name
    return ""


def _build_combined_rows(pipeline_paths: Sequence[Path], runs_root: Path) -> List[Dict[str, Any]]:
    combined_rows: List[Dict[str, Any]] = []
    for path in pipeline_paths:
        run_id = next((part for part in path.relative_to(runs_root).parts if part.startswith("run_")), "")
        for row in _read_csv(path):
            enriched = dict(row)
            enriched["source_pipeline_path"] = str(path)
            enriched["source_run_dir"] = str(path.parent)
            enriched["benchmark_run_id"] = run_id
            enriched["inferred_experiment_name"] = _infer_value(enriched, "experiment_name", path)
            enriched["inferred_dataset"] = _infer_value(enriched, "dataset", path)
            enriched["inferred_model"] = _infer_value(enriched, "model", path)
            combined_rows.append(enriched)
    return combined_rows


def _write_combined_csv(rows: List[Dict[str, Any]], out_dir: Path) -> Path:
    combined_path = out_dir / "combined_pipeline_results.csv"
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    if not fieldnames:
        fieldnames = [
            "source_pipeline_path",
            "source_run_dir",
            "benchmark_run_id",
            "inferred_experiment_name",
            "inferred_dataset",
            "inferred_model",
        ]
    _write_csv(combined_path, rows, fieldnames)
    return combined_path


def _write_manifest(manifest: List[Dict[str, Any]], out_dir: Path) -> Path:
    path = out_dir / "artifact_manifest.csv"
    fieldnames = ["artifact_type", "benchmark_run_id", "source_path", "copied_path", "size_bytes"]
    _write_csv(path, manifest, fieldnames)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    runs_root = Path(args.runs_root).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = _discover_run_dirs(runs_root)
    selected = _select_runs(run_dirs=run_dirs, include_runs=args.include_runs, include_incomplete=bool(args.include_incomplete))

    manifest: List[Dict[str, Any]] = []
    _collect_global_files(runs_root, out_dir, manifest)

    pipeline_paths: List[Path] = []
    for run_id, run_path in run_dirs.items():
        if run_id not in selected:
            if run_id not in DEFAULT_COMPLETED_RUNS:
                print(f"Skipping incomplete/unselected run: {run_id}")
            continue
        if (not args.include_incomplete) and (not args.include_runs) and (run_id not in DEFAULT_COMPLETED_RUNS):
            print(f"Skipping incomplete/unselected run: {run_id}")
            continue
        if (args.include_runs and run_id in args.include_runs) and (run_id not in DEFAULT_COMPLETED_RUNS):
            print(f"Including potentially incomplete run: {run_id}")
        paths = _collect_run_files(
            runs_root=runs_root,
            out_dir=out_dir,
            run_id=run_id,
            run_path=run_path,
            include_rl_trace=bool(args.include_rl_trace),
            include_q_table=bool(args.include_q_table),
            include_deployment_trace=bool(args.include_deployment_trace),
            include_action_diagnostics=bool(args.include_action_diagnostics),
            manifest=manifest,
        )
        pipeline_paths.extend(paths)

    combined_rows = _build_combined_rows(pipeline_paths, runs_root)
    combined_path = _write_combined_csv(combined_rows, out_dir)
    manifest.append(
        {
            "artifact_type": "derived_csv",
            "benchmark_run_id": "all",
            "source_path": "",
            "copied_path": str(combined_path),
            "size_bytes": combined_path.stat().st_size if combined_path.exists() else 0,
        }
    )
    manifest_path = _write_manifest(manifest, out_dir)

    if args.zip:
        archive_base = str(out_dir)
        shutil.make_archive(archive_base, "zip", root_dir=out_dir)

    print(f"Collected artifacts to {out_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Combined CSV: {combined_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

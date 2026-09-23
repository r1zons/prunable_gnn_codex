from __future__ import annotations

import csv
import importlib.util
from pathlib import Path


def _load_module():
    module_path = Path("scripts/collect_benchmark_artifacts.py")
    spec = importlib.util.spec_from_file_location("collect_benchmark_artifacts", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["phase"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _create_fake_runs(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    _write_csv(root / "suite_runs.csv", [{"phase": "post_prune", "dataset": "cora"}])
    _write_csv(root / "suite_aggregate.csv", [{"phase": "post_prune", "dataset": "cora"}])

    for run_id in ("run_000", "run_001", "run_002"):
        exp = root / run_id / "exp_a"
        _write_csv(
            exp / "pipeline_results.csv",
            [
                {
                    "experiment_name": "exp_a",
                    "dataset": "cora",
                    "model": "gcn",
                    "phase": "post_prune",
                    "pruning_method": "q_learning_tabular",
                    "requested_sparsity": "0.7",
                    "achieved_sparsity": "0.65",
                    "test_accuracy": "0.8",
                }
            ],
        )
        (exp / "resolved_config.yaml").write_text("run:\n  seed: 42\n", encoding="utf-8")
        (exp / "summary_pipeline.md").write_text("# summary\n", encoding="utf-8")
        (exp / "deployment_trace.json").write_text("[]\n", encoding="utf-8")
        (exp / "action_space_diagnostics.json").write_text("{}\n", encoding="utf-8")
        (exp / "rl_trace.json").write_text("[]\n", encoding="utf-8")
        (exp / "q_table.json").write_text("{}\n", encoding="utf-8")


def test_script_imports_and_parser() -> None:
    module = _load_module()
    parser = module.build_parser()
    args = parser.parse_args([])
    assert args.runs_root == "runs/compact_pruning_benchmark"
    assert args.include_runs is None


def test_default_collects_run_000_run_001_only(tmp_path: Path) -> None:
    module = _load_module()
    runs_root = tmp_path / "runs" / "compact_pruning_benchmark"
    _create_fake_runs(runs_root)
    out_dir = tmp_path / "exports" / "artifacts"

    code = module.main(["--runs-root", str(runs_root), "--out-dir", str(out_dir)])
    assert code == 0
    assert (out_dir / "run_000").exists()
    assert (out_dir / "run_001").exists()
    assert not (out_dir / "run_002").exists()
    assert (out_dir / "combined_pipeline_results.csv").exists()
    assert (out_dir / "artifact_manifest.csv").exists()
    assert (out_dir / "run_000" / "exp_a" / "action_space_diagnostics.json").exists()
    assert not (out_dir / "run_000" / "exp_a" / "rl_trace.json").exists()


def test_include_runs_can_collect_run_002(tmp_path: Path) -> None:
    module = _load_module()
    runs_root = tmp_path / "runs" / "compact_pruning_benchmark"
    _create_fake_runs(runs_root)
    out_dir = tmp_path / "exports" / "artifacts_including_002"

    code = module.main(
        [
            "--runs-root",
            str(runs_root),
            "--out-dir",
            str(out_dir),
            "--include-runs",
            "run_002",
        ]
    )
    assert code == 0
    assert (out_dir / "run_002" / "exp_a" / "pipeline_results.csv").exists()


def test_rl_trace_optional_flag(tmp_path: Path) -> None:
    module = _load_module()
    runs_root = tmp_path / "runs" / "compact_pruning_benchmark"
    _create_fake_runs(runs_root)
    out_dir = tmp_path / "exports" / "artifacts_with_trace"

    code = module.main(
        [
            "--runs-root",
            str(runs_root),
            "--out-dir",
            str(out_dir),
            "--include-runs",
            "run_000",
            "--include-rl-trace",
        ]
    )
    assert code == 0
    assert (out_dir / "run_000" / "exp_a" / "rl_trace.json").exists()

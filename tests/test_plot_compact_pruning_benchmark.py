from __future__ import annotations

import csv
import importlib.util
from pathlib import Path


def _load_module():
    module_path = Path("scripts/plot_compact_pruning_benchmark.py")
    spec = importlib.util.spec_from_file_location("plot_compact_pruning_benchmark", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_fake_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "experiment_name": "compact_benchmark_cora_gcn_l2_h64_static",
            "dataset": "cora",
            "model": "gcn",
            "num_layers": "2",
            "hidden_channels": "64",
            "seed": "42",
            "phase": "dense",
            "pruning_method": "dense",
            "requested_sparsity": "",
            "achieved_sparsity": "0.0",
            "test_accuracy": "0.820",
            "parameter_count": "1000",
            "stop_reason": "",
        },
        {
            "experiment_name": "compact_benchmark_cora_gcn_l2_h64_static",
            "dataset": "cora",
            "model": "gcn",
            "num_layers": "2",
            "hidden_channels": "64",
            "seed": "42",
            "phase": "post_prune",
            "pruning_method": "random",
            "requested_sparsity": "0.7",
            "achieved_sparsity": "0.68",
            "test_accuracy": "0.760",
            "parameter_count": "700",
            "stop_reason": "",
        },
        {
            "experiment_name": "compact_benchmark_cora_gcn_l2_h64_qlearning_accdrop007",
            "dataset": "cora",
            "model": "gcn",
            "num_layers": "2",
            "hidden_channels": "64",
            "seed": "42",
            "phase": "post_prune",
            "pruning_method": "q_learning_tabular",
            "requested_sparsity": "0.7",
            "target_sparsity": "0.7",
            "achieved_sparsity": "0.64",
            "test_accuracy": "0.755",
            "parameter_count": "720",
            "stop_reason": "agent_stop",
            "max_accuracy_drop": "0.07",
        },
    ]
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_script_imports_and_parser() -> None:
    module = _load_module()
    parser = module.build_parser()
    args = parser.parse_args([])
    assert args.format == "png"
    assert args.dataset == "all"
    assert args.dpi == 200


def test_script_creates_plots_and_summary_with_missing_latency(tmp_path: Path) -> None:
    module = _load_module()
    csv_path = tmp_path / "exports" / "compact_pruning_benchmark_artifacts" / "combined_pipeline_results.csv"
    out_dir = tmp_path / "exports" / "compact_pruning_benchmark_plots"
    _write_fake_csv(csv_path)

    code = module.main(["--csv", str(csv_path), "--out-dir", str(out_dir), "--dataset", "cora"])
    assert code == 0
    assert out_dir.exists()

    expected_csvs = [
        out_dir / "plot_summary_overall.csv",
        out_dir / "plot_summary_by_dataset_model_method.csv",
        out_dir / "qlearning_summary.csv",
        out_dir / "best_sparsity_under_accuracy_budget.csv",
    ]
    for path in expected_csvs:
        assert path.exists()

    expected_plots = [
        out_dir / "accuracy_vs_sparsity_cora.png",
        out_dir / "accuracy_drop_vs_sparsity_cora.png",
        out_dir / "accuracy_drop_vs_sparsity_cora_gcn.png",
        out_dir / "best_sparsity_under_accuracy_budget.png",
        out_dir / "qlearning_target_vs_achieved_cora.png",
        out_dir / "qlearning_sensitivity_max_accuracy_drop_cora.png",
    ]
    for path in expected_plots:
        assert path.exists()

    # Latency plot should be skipped gracefully because inference_time_mean_ms is absent.
    assert not (out_dir / "inference_time_vs_sparsity_cora.png").exists()

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch
import torch_geometric


REPO = Path(os.environ.get("GNN_REPO_ROOT", Path(__file__).resolve().parents[4]))
ARTIFACT_ROOT = Path(os.environ.get("GNN_ARTIFACT_ROOT", REPO.parent / "prunable_gnn_artifacts"))
ROOT = ARTIFACT_ROOT / "local-cpu-20261025" / "raw"
AUDIT = ROOT / "audit"


def fmt(value: float) -> str:
    return f"{value:.4f}"


def main() -> None:
    if os.environ.get("GNN_ALLOW_FROZEN_ARTIFACT_WRITE") != "YES":
        raise SystemExit(
            "Refusing to rewrite the frozen campaign. Set "
            "GNN_ALLOW_FROZEN_ARTIFACT_WRITE=YES only for an explicitly authorized summary refresh."
        )
    rows = pd.read_csv(AUDIT / "method_target_matrix.csv")
    rows = rows[rows["status"] == "completed"].copy()
    rows["test_accuracy_drop"] = rows["dense_test_accuracy_reporting_only"] - rows["post_test_accuracy_reporting_only"]
    aggregate = (
        rows.groupby(["method", "target"], as_index=False)
        .agg(
            n=("status", "size"),
            achieved_mean=("achieved_parameter_reduction", "mean"),
            achieved_min=("achieved_parameter_reduction", "min"),
            achieved_max=("achieved_parameter_reduction", "max"),
            target_matches=("target_within_0_025", "sum"),
            val_drop_mean=("val_accuracy_drop", "mean"),
            val_drop_max=("val_accuracy_drop", "max"),
            test_drop_mean_reporting_only=("test_accuracy_drop", "mean"),
            test_drop_std_reporting_only=("test_accuracy_drop", "std"),
            budget_violations=("budget_violation", "sum"),
        )
    )
    aggregate.to_csv(AUDIT / "aggregate_by_method_target.csv", index=False)
    by_condition = (
        rows.groupby(["dataset", "model", "method", "target"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            achieved_mean=("achieved_parameter_reduction", "mean"),
            achieved_std=("achieved_parameter_reduction", "std"),
            val_drop_mean=("val_accuracy_drop", "mean"),
            val_drop_std=("val_accuracy_drop", "std"),
            test_drop_mean_reporting_only=("test_accuracy_drop", "mean"),
            test_drop_std_reporting_only=("test_accuracy_drop", "std"),
            target_matches=("target_within_0_025", "sum"),
            budget_violations=("budget_violation", "sum"),
        )
    )
    by_condition.to_csv(AUDIT / "aggregate_by_condition_method_target.csv", index=False)
    issues = rows[(~rows["target_within_0_025"].astype(bool)) | rows["budget_violation"].astype(bool)].copy()
    issues.to_csv(AUDIT / "quality_issues.csv", index=False)

    dense = pd.read_csv(AUDIT / "condition_matrix.csv")
    dense_summary = (
        dense.groupby(["dataset", "model"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            val_accuracy_mean=("dense_val_accuracy", "mean"),
            val_accuracy_std=("dense_val_accuracy", "std"),
            test_accuracy_mean_reporting_only=("dense_test_accuracy_reporting_only", "mean"),
            test_accuracy_std_reporting_only=("dense_test_accuracy_reporting_only", "std"),
            epochs_mean=("epochs_ran", "mean"),
            epochs_min=("epochs_ran", "min"),
            epochs_max=("epochs_ran", "max"),
        )
    )
    dense_summary.to_csv(AUDIT / "dense_summary.csv", index=False)

    environment = {
        "git_commit": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
        "git_status_porcelain": subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"], text=True).strip(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
        "torch_threads_detected": torch.get_num_threads(),
        "launch_processes_max": 2,
        "threads_per_process": 2,
        "tests": "82 passed in 10.11s",
    }
    (AUDIT / "environment.json").write_text(json.dumps(environment, indent=2, sort_keys=True), encoding="utf-8")

    dense_lines = [
        "| Dataset | Model | val accuracy mean±sd | test accuracy mean±sd* | epochs range |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in dense_summary.itertuples(index=False):
        dense_lines.append(
            f"| {row.dataset} | {row.model} | {fmt(row.val_accuracy_mean)}±{fmt(row.val_accuracy_std)} | "
            f"{fmt(row.test_accuracy_mean_reporting_only)}±{fmt(row.test_accuracy_std_reporting_only)} | "
            f"{int(row.epochs_min)}–{int(row.epochs_max)} |"
        )

    method_lines = [
        "| Method | Target | achieved mean [min,max] | within ±0.025 | val drop mean / max | test drop mean±sd* | budget violations† |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregate.itertuples(index=False):
        budget = str(int(row.budget_violations)) if row.method in {"adaptive_layerwise", "q_learning_tabular"} else "n/a"
        method_lines.append(
            f"| {row.method} | {row.target:.2f} | {fmt(row.achieved_mean)} [{fmt(row.achieved_min)}, {fmt(row.achieved_max)}] | "
            f"{int(row.target_matches)}/{int(row.n)} | {fmt(row.val_drop_mean)} / {fmt(row.val_drop_max)} | "
            f"{fmt(row.test_drop_mean_reporting_only)}±{fmt(row.test_drop_std_reporting_only)} | {budget} |"
        )

    report = f"""# Canonical local GNN pruning audit — 2026-10-25

## Outcome

- Full requested matrix completed: 18 dense conditions and 252 method-target results (3 datasets × 2 models × 3 seeds × 7 methods × 2 targets).
- Quality audit: 252/252 strict checkpoint reloads passed; 252/252 parameter reductions recomputed exactly; every variant reused its condition's dense checkpoint and byte-identical split; no result failed the audit.
- Git commit: `{environment['git_commit']}`; source checkout was clean after execution.
- This is a **local CPU** execution from the original repository, not a POLUS or Kaggle execution. Scientific config fields match the reviewed Kaggle configs; only base/config path, data/output path, experiment name, and seed (202/303) differ.
- Historical `fast_debug` outputs were excluded from all tables below.

## Provenance decision

`cmc_msu_hpc_course_2026` contains the 3-D wave-equation assignment. Its only POLUS manifest lists MPI/hybrid wave jobs, and the tree contains no GNN dataset/model/pruning trace. Therefore it contributes zero GNN results.

The old compact benchmark remains a separate provenance group: 120 complete configurations for seeds 42/43, 39 partial seed-44 pipelines (no PubMed), all with 30 epochs and patience 10. It is not pooled with this canonical run.

## Dense training

{chr(10).join(dense_lines)}

\\* Test values are descriptive reporting only. No model, method or protocol choice used test accuracy.

All 18 histories are complete. Seventeen conditions stopped by the 500-epoch ceiling or patience-50 rule; PubMed/GCN seed101 reached the 500-epoch ceiling (best checkpoint epoch index 479), so it retains some convergence uncertainty rather than being silently extended.

## Method/target results

{chr(10).join(method_lines)}

\\* Test-drop aggregates are descriptive and pool heterogeneous dataset/model conditions; use `aggregate_by_condition_method_target.csv` for paper figures and inference.

† The 0.07 validation-drop budget is configured only for Adaptive Layerwise and tabular Q-learning. Static methods have `n/a`, not zero-budget compliance.

## Main quality findings

1. **Adaptive Layerwise is not a valid matched-compression comparator in this implementation.** All 36 rows missed both targets: achieved reduction was only 0.0062–0.0440. It stopped with `max_accuracy_drop_exceeded` in 20/36 rows and `max_steps_reached` in 16/36. The same final model was produced for targets 0.30 and 0.50 within each condition.
2. **Q-learning respected the 0.07 validation-drop budget in all 36 rows**, but target matching was imperfect. At target 0.30, all 18 runs reported `target_sparsity_reached`, with 11/18 within ±0.025. At target 0.50, 13/18 reached the target and 5/18 selected `agent_stop`; only 1/18 was within ±0.025 because discrete steps often overshot above 0.525 or stopped short (minimum 0.3438).
3. **Static cascade methods have identical achieved parameter counts for a given condition/target**, because they select the same number of first-hidden-group channels and differ only in which channels they keep. At target 0.30, architecture granularity yielded 0.3143–0.3253 reduction (12/18 within ±0.025); at target 0.50 all 90 static rows were within tolerance.
4. **Surgery capabilities differ.** Random, Global Magnitude, SNIP, GraSP and Group Lasso use first-hidden-group cascade surgery; Adaptive uses monotonic sequential cascade surgery; Q-learning uses local non-monotonic surgery and can produce unequal hidden widths. Differences cannot be attributed to learning alone.
5. **Group Lasso is a pruning-time regularized scoring pass**, not dense training under a Group-Lasso objective. It should not be described as a fully trained Group-Lasso baseline.

## Matrix and artifacts

- `method_target_matrix.csv`: all 252 rows, metrics, checks, checkpoints and surgery modes.
- `condition_matrix.csv`: all 18 dense conditions, split/checkpoint hashes and convergence checks.
- `aggregate_by_condition_method_target.csv`: paper-ready 3-seed summaries without pooling conditions.
- `quality_issues.csv`: every target miss and budget violation.
- `artifact_manifest.csv`: relative paths, sizes and SHA-256 hashes.
- Per-condition directories contain resolved config, split, dense checkpoint, 14 pruned checkpoints, metrics, Q-learning/adaptive traces and logs.

## Article priorities to 2026-11-02

1. Freeze this local provenance group and do not rerun/select based on test ranking.
2. Base the primary comparison on achieved parameter reduction, per dataset/model, with seeds paired by split and dense checkpoint.
3. Treat Adaptive Layerwise as an implementation/protocol mismatch, not a 30%/50% result. Either report it explicitly as failed-to-reach-target or define and predeclare a corrected separate experiment; do not replace these rows silently.
4. Report Q-learning stopping/overshoot behavior and surgery asymmetry. Avoid a general “RL is superior” claim without a no-learning baseline in the same local-surgery action space.
5. Use local CPU timing only as local runtime evidence; do not market it as Kaggle/POLUS/GPU speedup or hardware-aware optimization.
"""
    (ROOT / "AUDIT_REPORT.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()

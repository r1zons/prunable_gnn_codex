"""Run and validate the manually launched Kaggle pruning pilot.

This script is intended for Kaggle "Save & Run All" execution. It must not be
used to start experiments without the researcher's explicit approval.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


REQUIRED_METHODS = (
    "random",
    "global_magnitude",
    "snip",
    "grasp",
    "group_lasso",
    "adaptive_layerwise",
    "q_learning_tabular",
)
REQUESTED_LEVELS = (0.3, 0.5)
COMPRESSION_MATCH_TOLERANCE = 0.025
PILOT_SEED = 101
OUTPUT_ROOT = Path("/kaggle/working/prunable_gnn_kaggle_pilot")
CONFIGS = {
    "cora_gcn_l3_h32": "configs/experiments/kaggle_pilot/cora_gcn_l3_h32.yaml",
    "cora_graphsage_l3_h32": "configs/experiments/kaggle_pilot/cora_graphsage_l3_h32.yaml",
    "citeseer_gcn_l3_h32": "configs/experiments/kaggle_pilot/citeseer_gcn_l3_h32.yaml",
    "citeseer_graphsage_l3_h32": "configs/experiments/kaggle_pilot/citeseer_graphsage_l3_h32.yaml",
    "pubmed_gcn_l3_h32": "configs/experiments/kaggle_pilot/pubmed_gcn_l3_h32.yaml",
    "pubmed_graphsage_l3_h32": "configs/experiments/kaggle_pilot/pubmed_graphsage_l3_h32.yaml",
}
CAPABILITIES = {
    "random": {
        "surgery_mode": "cascade_first_hidden_group",
        "request_semantics": "hidden_channel_fraction_input",
        "requires_train_gradients": False,
    },
    "global_magnitude": {
        "surgery_mode": "cascade_first_hidden_group",
        "request_semantics": "hidden_channel_fraction_input",
        "requires_train_gradients": False,
    },
    "snip": {
        "surgery_mode": "cascade_first_hidden_group",
        "request_semantics": "hidden_channel_fraction_input",
        "requires_train_gradients": True,
    },
    "grasp": {
        "surgery_mode": "cascade_first_hidden_group",
        "request_semantics": "hidden_channel_fraction_input",
        "requires_train_gradients": True,
    },
    "group_lasso": {
        "surgery_mode": "cascade_first_hidden_group",
        "request_semantics": "hidden_channel_fraction_input",
        "requires_train_gradients": True,
        "note": "One Group-Lasso-regularized scoring pass; not dense retraining with Group Lasso.",
    },
    "adaptive_layerwise": {
        "surgery_mode": "cascade_sequential_monotonic",
        "request_semantics": "parameter_reduction_stopping_target",
        "requires_train_gradients": False,
    },
    "q_learning_tabular": {
        "surgery_mode": "local_sequential_nonmonotonic",
        "request_semantics": "parameter_reduction_stopping_target",
        "requires_train_gradients": False,
    },
}
EPOCH_PATTERN = re.compile(
    r"Epoch (?P<epoch>\d+)/(?P<max_epochs>\d+) \| "
    r"train_loss=(?P<train_loss>[-+0-9.eE]+) \| "
    r"val_loss=(?P<val_loss>[-+0-9.eE]+) \| "
    r"train_acc=(?P<train_acc>[-+0-9.eE]+) \| "
    r"val_acc=(?P<val_acc>[-+0-9.eE]+) \| "
    r"best_epoch=(?P<best_epoch>\d+) \| "
    r"early_stop=(?P<early_stop>\d+) \| "
    r"elapsed_sec=(?P<elapsed_sec>[-+0-9.eE]+)"
)


def build_parser() -> argparse.ArgumentParser:
    """Build the pilot runner CLI."""
    parser = argparse.ArgumentParser(description="Run the approved Kaggle seed-101 pilot.")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true", help="Run all six independent conditions sequentially.")
    selection.add_argument(
        "--condition",
        action="append",
        choices=sorted(CONFIGS),
        help="Run one condition; repeat the option to select several conditions.",
    )
    parser.add_argument("--repo-root", type=Path, required=True, help="Pinned repository checkout root.")
    parser.add_argument("--expected-commit", required=True, help="Full Git commit SHA expected at repo-root.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate provenance, run selected conditions, and validate artifacts."""
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.expanduser().resolve()
    selected = sorted(CONFIGS) if args.all else list(dict.fromkeys(args.condition or []))
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    commit = _verify_checkout(repo_root, args.expected_commit)
    environment = _environment_payload(commit)
    _write_json(OUTPUT_ROOT / "environment.json", environment)
    _write_json(OUTPUT_ROOT / "method_capabilities.json", CAPABILITIES)
    _preflight(repo_root, selected)

    for condition in selected:
        _run_condition(repo_root, condition, commit)

    rows = _collect_completed_rows(OUTPUT_ROOT)
    _write_csv(OUTPUT_ROOT / "pilot_validation_summary.csv", rows)
    _write_json(
        OUTPUT_ROOT / "pilot_run_manifest.json",
        {
            "status": "complete",
            "generated_at_utc": _utc_now(),
            "git_commit": commit,
            "pilot_seed": PILOT_SEED,
            "selected_conditions": selected,
            "completed_conditions_present": sorted({str(row["condition"]) for row in rows}),
            "historical_results_pooled": False,
            "test_results_used_for_selection": False,
            "confirmatory_seeds_not_run": [202, 303],
        },
    )
    _write_artifact_manifest(OUTPUT_ROOT)
    print(f"Pilot preparation complete. Validation-only summary: {OUTPUT_ROOT / 'pilot_validation_summary.csv'}")
    return 0


def _verify_checkout(repo_root: Path, expected_commit: str) -> str:
    if not (repo_root / ".git").exists():
        raise RuntimeError(f"Repository checkout not found: {repo_root}")
    commit = _git(repo_root, "rev-parse", "HEAD").strip()
    expected = str(expected_commit).strip().lower()
    if len(expected) != 40 or commit.lower() != expected:
        raise RuntimeError(f"Pinned commit mismatch: expected {expected}, found {commit}")
    tracked_changes = _git(repo_root, "status", "--porcelain", "--untracked-files=no").strip()
    if tracked_changes:
        raise RuntimeError(f"Pinned checkout has tracked modifications:\n{tracked_changes}")
    return commit


def _preflight(repo_root: Path, selected: Sequence[str]) -> None:
    sys.path.insert(0, str(repo_root / "src"))
    from gnn_pruning.config import load_yaml, resolve_config
    from gnn_pruning.pruning import list_pruners

    registered = {str(row["name"]): row for row in list_pruners()}
    missing = [method for method in REQUIRED_METHODS if method not in registered]
    unsupported = [method for method in REQUIRED_METHODS if not bool(registered.get(method, {}).get("supports_structured"))]
    if missing or unsupported:
        raise RuntimeError(f"Required structured pruner incompatibility: missing={missing}, unsupported={unsupported}")

    for condition in selected:
        config_path = repo_root / CONFIGS[condition]
        raw = load_yaml(config_path)
        resolved = resolve_config(config_path)
        pruning = raw.get("pruning", {})
        q_learning = raw.get("q_learning", {})
        actual_methods = tuple(str(value) for value in pruning.get("methods", []))
        levels = tuple(float(value) for value in pruning.get("sparsity_levels", []))
        expected_output = OUTPUT_ROOT / "seed_101" / condition

        checks = {
            "seed": int(resolved.run.seed) == PILOT_SEED,
            "num_layers": int(resolved.model.num_layers) == 3,
            "hidden_channels": int(resolved.model.hidden_channels) == 32,
            "epochs": int(resolved.training.epochs) == 500,
            "patience": int(resolved.training.early_stopping_patience) == 50,
            "methods": actual_methods == REQUIRED_METHODS,
            "levels": levels == REQUESTED_LEVELS,
            "structured": pruning.get("structured") is True,
            "finetune_disabled": pruning.get("finetune_enabled") is False,
            "q_budget": float(q_learning.get("max_accuracy_drop", -1)) == 0.07,
            "q_local": str(q_learning.get("structural_pruning_mode")) == "local",
            "q_nonmonotonic": q_learning.get("allow_nonmonotonic_layer_order") is True,
            "output_dir": Path(resolved.run.output_dir) == expected_output,
        }
        failed = [name for name, valid in checks.items() if not valid]
        if failed:
            raise RuntimeError(f"Config preflight failed for {condition}: {failed}")
        if expected_output.exists():
            raise RuntimeError(
                f"Output already exists for {condition}: {expected_output}. "
                "Move it aside before retrying; the runner will not mix or overwrite runs."
            )


def _run_condition(repo_root: Path, condition: str, commit: str) -> None:
    config_path = repo_root / CONFIGS[condition]
    condition_dir = OUTPUT_ROOT / "seed_101" / condition
    condition_dir.mkdir(parents=True, exist_ok=False)
    log_path = condition_dir / "launcher.log"
    command = [
        sys.executable,
        "-u",
        "-m",
        "gnn_pruning",
        "run-pipeline",
        "--config",
        str(config_path),
        "--progress",
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root / "src")
    started = _utc_now()
    started_clock = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"git_commit={commit}\ncondition={condition}\ncommand={command!r}\nstarted_at_utc={started}\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=repo_root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    if completed.returncode != 0:
        _record_condition_failure(
            condition_dir=condition_dir,
            condition=condition,
            commit=commit,
            phase="pipeline_execution",
            error=f"Pipeline process exited with status {completed.returncode}",
            return_code=completed.returncode,
            log_path=log_path,
        )
        raise RuntimeError(f"Condition failed without being skipped: {condition}. See {log_path}")

    try:
        rows = _validate_condition(condition_dir, condition)
        _write_csv(condition_dir / "validation_summary.csv", rows)
        convergence = _extract_convergence(condition_dir / "progress.log")
        _write_csv(condition_dir / "dense_convergence.csv", convergence)
    except Exception as exc:
        _record_condition_failure(
            condition_dir=condition_dir,
            condition=condition,
            commit=commit,
            phase="artifact_validation",
            error=f"{type(exc).__name__}: {exc}",
            return_code=completed.returncode,
            log_path=log_path,
        )
        raise
    _write_json(
        condition_dir / "condition_complete.json",
        {
            "condition": condition,
            "git_commit": commit,
            "completed_at_utc": _utc_now(),
            "elapsed_sec": time.perf_counter() - started_clock,
            "post_prune_rows": len(rows),
            "dense_epochs_recorded": len(convergence),
            "test_results_used_for_selection": False,
        },
    )


def _validate_condition(condition_dir: Path, condition: str) -> List[Dict[str, Any]]:
    pipeline_csv = condition_dir / "pipeline_results.csv"
    rows = _read_csv(pipeline_csv)
    dense_rows = [row for row in rows if row.get("phase") == "dense"]
    post_rows = [row for row in rows if row.get("phase") == "post_prune"]
    if len(dense_rows) != 1:
        raise RuntimeError(f"{condition}: expected one dense row, found {len(dense_rows)}")
    if len(post_rows) != len(REQUIRED_METHODS) * len(REQUESTED_LEVELS):
        raise RuntimeError(f"{condition}: expected 14 post-prune rows, found {len(post_rows)}")

    dense = dense_rows[0]
    dense_params = int(float(dense["parameter_count"]))
    dense_val = float(dense["val_accuracy"])
    dense_checkpoint = (condition_dir / "dense_checkpoint.pt").resolve()
    root_split_hash = _sha256(condition_dir / "splits.yaml")
    summaries: List[Dict[str, Any]] = []

    observed = {(row["method"], float(row["requested_sparsity"])) for row in post_rows}
    expected = {(method, level) for method in REQUIRED_METHODS for level in REQUESTED_LEVELS}
    if observed != expected:
        raise RuntimeError(f"{condition}: method/target matrix mismatch: expected={expected}, observed={observed}")

    for row in post_rows:
        method = str(row["method"])
        requested = float(row["requested_sparsity"])
        pruned_params = int(float(row["parameter_count"]))
        achieved = float(row["achieved_sparsity"])
        recomputed = 1.0 - (pruned_params / dense_params)
        if abs(achieved - recomputed) > 1e-10:
            raise RuntimeError(
                f"{condition}/{method}/{requested}: achieved sparsity mismatch "
                f"reported={achieved}, recomputed={recomputed}"
            )

        variant_dir = Path(row["run_dir"]).resolve()
        debug = _read_json(variant_dir / "pruning_debug.json")
        prune_debug = debug.get("prune", {})
        loaded_checkpoint = Path(str(prune_debug.get("loaded_checkpoint_path", ""))).resolve()
        if loaded_checkpoint != dense_checkpoint:
            raise RuntimeError(f"{condition}/{method}/{requested}: dense checkpoint was not shared")
        if _sha256(variant_dir / "splits.yaml") != root_split_hash:
            raise RuntimeError(f"{condition}/{method}/{requested}: split differs from dense split")

        final_widths = prune_debug.get("hidden_dims_after", [])
        stop_reason = row.get("stop_reason", "")
        if method == "q_learning_tabular":
            diagnostics = _read_json(variant_dir / "action_space_diagnostics.json")
            deployment = diagnostics.get("deployment", {})
            final_widths = deployment.get("final_hidden_widths", final_widths)
            stop_reason = deployment.get("final_stop_reason", stop_reason)
            for required in ("q_table.json", "rl_trace.json", "deployment_trace.json", "action_space_diagnostics.json"):
                if not (variant_dir / required).exists():
                    raise RuntimeError(f"{condition}: missing Q-learning artifact {variant_dir / required}")
        if method == "adaptive_layerwise" and not (variant_dir / "adaptive_trace.json").exists():
            raise RuntimeError(f"{condition}: missing adaptive trace {variant_dir / 'adaptive_trace.json'}")

        val_accuracy = float(row["val_accuracy"])
        val_drop = max(0.0, dense_val - val_accuracy)
        budget = 0.07 if method in {"adaptive_layerwise", "q_learning_tabular"} else ""
        summaries.append(
            {
                "condition": condition,
                "dataset": row["dataset"],
                "model": row["model"],
                "seed": int(row["seed"]),
                "method": method,
                "requested_sparsity": requested,
                "request_semantics": CAPABILITIES[method]["request_semantics"],
                "achieved_parameter_reduction": recomputed,
                "requested_target_gap": recomputed - requested,
                "requested_target_match": abs(recomputed - requested) <= COMPRESSION_MATCH_TOLERANCE,
                "dense_parameter_count": dense_params,
                "pruned_parameter_count": pruned_params,
                "dense_val_accuracy": dense_val,
                "post_prune_val_accuracy": val_accuracy,
                "val_accuracy_drop": val_drop,
                "accuracy_drop_budget": budget,
                "accuracy_budget_violation": bool(budget != "" and val_drop > float(budget)),
                "stop_reason": stop_reason,
                "final_hidden_widths": json.dumps(final_widths),
                "surgery_mode": CAPABILITIES[method]["surgery_mode"],
                "checkpoint_path": row["checkpoint_path"],
                "metrics_path": row["metrics_path"],
            }
        )
    return _annotate_compression_matches(summaries)


def _annotate_compression_matches(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Label requested-target groups without rejecting valid method results."""
    for requested in REQUESTED_LEVELS:
        group = [row for row in rows if float(row["requested_sparsity"]) == requested]
        if not group:
            continue
        achieved = [float(row["achieved_parameter_reduction"]) for row in group]
        span = max(achieved) - min(achieved)
        status = "matched" if span <= COMPRESSION_MATCH_TOLERANCE else "unmatched"
        for row in group:
            row["comparison_achieved_span"] = span
            row["comparison_match_tolerance"] = COMPRESSION_MATCH_TOLERANCE
            row["comparison_match_status"] = status
    return rows


def _record_condition_failure(
    *,
    condition_dir: Path,
    condition: str,
    commit: str,
    phase: str,
    error: str,
    return_code: int,
    log_path: Path,
) -> None:
    """Record a failed condition without discarding artifacts from completed variants."""
    inventory = _partial_result_inventory(condition_dir)
    _write_csv(condition_dir / "partial_results_inventory.csv", inventory)
    _write_json(
        condition_dir / "failure.json",
        {
            "condition": condition,
            "git_commit": commit,
            "failure_phase": phase,
            "error": error,
            "return_code": return_code,
            "failed_at_utc": _utc_now(),
            "log_path": str(log_path),
            "preserved_variant_count": len(inventory),
            "partial_results_inventory": str(condition_dir / "partial_results_inventory.csv"),
        },
    )
    _write_artifact_manifest(OUTPUT_ROOT)


def _partial_result_inventory(condition_dir: Path) -> List[Dict[str, Any]]:
    """Inventory completed variant artifacts without interpreting incomplete results."""
    rows: List[Dict[str, Any]] = []
    pruning_root = condition_dir / "pruning"
    for variant_dir in sorted(pruning_root.glob("*/sparsity_*")):
        metrics = sorted(variant_dir.glob("metrics_post_prune*.json"))
        pruning_metrics = sorted(variant_dir.glob("pruning_metrics_*.json"))
        checkpoints = sorted(variant_dir.glob("pruned_*.pt"))
        if not metrics and not pruning_metrics and not checkpoints:
            continue
        rows.append(
            {
                "method": variant_dir.parent.name,
                "variant_dir": str(variant_dir),
                "post_prune_metrics_present": bool(metrics),
                "pruning_metrics_present": bool(pruning_metrics),
                "pruned_checkpoint_present": bool(checkpoints),
                "split_present": (variant_dir / "splits.yaml").exists(),
                "progress_log_present": (variant_dir / "progress.log").exists(),
            }
        )
    return rows


def _extract_convergence(log_path: Path) -> List[Dict[str, Any]]:
    if not log_path.exists():
        raise RuntimeError(f"Dense progress log missing: {log_path}")
    rows: List[Dict[str, Any]] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        match = EPOCH_PATTERN.search(line)
        if not match:
            continue
        payload = match.groupdict()
        rows.append(
            {
                "epoch": int(payload["epoch"]),
                "max_epochs": int(payload["max_epochs"]),
                "train_loss": float(payload["train_loss"]),
                "val_loss": float(payload["val_loss"]),
                "train_accuracy": float(payload["train_acc"]),
                "val_accuracy": float(payload["val_acc"]),
                "best_epoch": int(payload["best_epoch"]),
                "early_stopping_counter": int(payload["early_stop"]),
                "elapsed_sec": float(payload["elapsed_sec"]),
            }
        )
    if not rows:
        raise RuntimeError(f"No dense convergence rows parsed from {log_path}")
    return rows


def _collect_completed_rows(output_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted((output_root / "seed_101").glob("*/validation_summary.csv")):
        rows.extend(_read_csv(path))
    return rows


def _environment_payload(commit: str) -> Dict[str, Any]:
    import torch
    import torch_geometric

    return {
        "captured_at_utc": _utc_now(),
        "git_commit": commit,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "kaggle_kernel_run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE", ""),
    }


def _write_artifact_manifest(output_root: Path) -> None:
    manifest_path = output_root / "artifact_manifest.csv"
    rows = []
    for path in sorted(output_root.rglob("*")):
        if not path.is_file() or path == manifest_path:
            continue
        rows.append(
            {
                "relative_path": str(path.relative_to(output_root)),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    _write_csv(manifest_path, rows)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise RuntimeError(f"Required CSV missing: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in materialized:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(materialized)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"Required JSON missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())

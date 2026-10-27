from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
from itertools import product
from pathlib import Path

import torch

from gnn_pruning.models import build_model


REPO = Path(os.environ.get("GNN_REPO_ROOT", Path(__file__).resolve().parents[4]))
ARTIFACT_ROOT = Path(os.environ.get("GNN_ARTIFACT_ROOT", REPO.parent / "prunable_gnn_artifacts"))
ROOT = ARTIFACT_ROOT / "local-cpu-20261025" / "raw"
DATASETS = ("cora", "citeseer", "pubmed")
MODELS = ("gcn", "graphsage")
METHODS = (
    "random",
    "global_magnitude",
    "snip",
    "grasp",
    "group_lasso",
    "adaptive_layerwise",
    "q_learning_tabular",
)
TARGETS = (0.3, 0.5)
SEEDS = (101, 202, 303)
SURGERY = {
    "random": "cascade_first_hidden_group",
    "global_magnitude": "cascade_first_hidden_group",
    "snip": "cascade_first_hidden_group",
    "grasp": "cascade_first_hidden_group",
    "group_lasso": "cascade_first_hidden_group",
    "adaptive_layerwise": "cascade_sequential_monotonic",
    "q_learning_tabular": "local_sequential_nonmonotonic",
}
EPOCH_RE = re.compile(r"Epoch (\d+)/(\d+).*best_epoch=(\d+).*early_stop=(\d+)")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def strict_reload(checkpoint: Path) -> tuple[bool, str, list[int]]:
    try:
        payload = torch.load(checkpoint, map_location="cpu")
        config = dict(payload["model_config"])
        hidden = config.get("hidden_channels")
        widths = [int(v) for v in hidden] if isinstance(hidden, list) else [int(hidden)] * (int(config["num_layers"]) - 1)
        model = build_model(str(payload["model_name"]), **config)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        return True, "", widths
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}", []


def epoch_summary(condition_dir: Path) -> dict[str, object]:
    matches = []
    for line in (condition_dir / "progress.log").read_text(encoding="utf-8").splitlines():
        match = EPOCH_RE.search(line)
        if match:
            matches.append(tuple(int(v) for v in match.groups()))
    metrics = json.loads((condition_dir / "metrics_train.json").read_text(encoding="utf-8"))["training"]
    epochs_ran = int(metrics["epochs_ran"])
    best_epoch = int(metrics["best_epoch"])
    return {
        "epochs_ran": epochs_ran,
        "best_epoch": best_epoch,
        "best_val_loss": float(metrics["best_val_loss"]),
        "history_rows": len(matches),
        "history_complete": len(matches) == epochs_ran,
        "max_epoch_500": epochs_ran <= 500,
        "patience50_consistent": epochs_ran == 500 or epochs_ran - best_epoch >= 50,
    }


def condition_name(dataset: str, model: str) -> str:
    return f"{dataset}_{model}_l3_h32"


def audit() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    results: list[dict[str, object]] = []
    conditions: list[dict[str, object]] = []
    for seed, dataset, model in product(SEEDS, DATASETS, MODELS):
        condition = condition_name(dataset, model)
        condition_dir = ROOT / "results" / f"seed_{seed}" / condition
        csv_path = condition_dir / "pipeline_results.csv"
        if not csv_path.exists():
            conditions.append({"seed": seed, "dataset": dataset, "model": model, "condition": condition, "status": "missing"})
            for method, target in product(METHODS, TARGETS):
                results.append(
                    {
                        "seed": seed,
                        "dataset": dataset,
                        "model": model,
                        "condition": condition,
                        "method": method,
                        "target": target,
                        "status": "missing",
                        "surgery_mode": SURGERY[method],
                    }
                )
            continue

        rows = read_csv(csv_path)
        dense_rows = [row for row in rows if row["phase"] == "dense"]
        post_rows = [row for row in rows if row["phase"] == "post_prune"]
        errors: list[str] = []
        if len(dense_rows) != 1:
            errors.append(f"dense_rows={len(dense_rows)}")
        if len(post_rows) != 14:
            errors.append(f"post_rows={len(post_rows)}")
        if errors:
            conditions.append(
                {"seed": seed, "dataset": dataset, "model": model, "condition": condition, "status": "failed", "errors": ";".join(errors)}
            )
            continue

        dense = dense_rows[0]
        dense_params = int(float(dense["parameter_count"]))
        dense_val = float(dense["val_accuracy"])
        dense_test = float(dense["test_accuracy"])
        dense_checkpoint = (condition_dir / "dense_checkpoint.pt").resolve()
        split_hash = sha256(condition_dir / "splits.yaml")
        epochs = epoch_summary(condition_dir)
        observed = {(row["method"], float(row["requested_sparsity"])) for row in post_rows}
        expected = set(product(METHODS, TARGETS))
        if observed != expected:
            errors.append("method_target_matrix_mismatch")

        for row in post_rows:
            method = row["method"]
            target = float(row["requested_sparsity"])
            variant = Path(row["run_dir"]).resolve()
            params = int(float(row["parameter_count"]))
            reported = float(row["achieved_sparsity"])
            achieved = 1.0 - params / dense_params
            debug = json.loads((variant / "pruning_debug.json").read_text(encoding="utf-8"))["prune"]
            loaded = Path(str(debug["loaded_checkpoint_path"])).resolve()
            checkpoint = Path(row["checkpoint_path"]).resolve()
            reloadable, reload_error, widths = strict_reload(checkpoint)
            artifacts_ok = True
            stop_reason = row.get("stop_reason", "")
            if method == "q_learning_tabular":
                required = ("q_table.json", "rl_trace.json", "deployment_trace.json", "action_space_diagnostics.json")
                artifacts_ok = all((variant / name).exists() for name in required)
                diagnostics = json.loads((variant / "action_space_diagnostics.json").read_text(encoding="utf-8"))
                deployment = diagnostics["deployment"]
                widths = [int(v) for v in deployment.get("final_hidden_widths", widths)]
                stop_reason = str(deployment.get("final_stop_reason", stop_reason))
            elif method == "adaptive_layerwise":
                artifacts_ok = (variant / "adaptive_trace.json").exists()
            val_accuracy = float(row["val_accuracy"])
            val_drop = max(0.0, dense_val - val_accuracy)
            budgeted = method in {"adaptive_layerwise", "q_learning_tabular"}
            checks = {
                "reduction_exact": abs(reported - achieved) <= 1e-10,
                "structural_reduction": params < dense_params,
                "shared_dense_checkpoint": loaded == dense_checkpoint,
                "shared_split": sha256(variant / "splits.yaml") == split_hash,
                "reloadable": reloadable,
                "required_artifacts": artifacts_ok,
            }
            status = "completed" if all(checks.values()) else "failed_quality_check"
            results.append(
                {
                    "seed": seed,
                    "dataset": dataset,
                    "model": model,
                    "condition": condition,
                    "method": method,
                    "target": target,
                    "status": status,
                    "dense_params": dense_params,
                    "pruned_params": params,
                    "achieved_parameter_reduction": achieved,
                    "target_gap": achieved - target,
                    "target_within_0_025": abs(achieved - target) <= 0.025,
                    "dense_val_accuracy": dense_val,
                    "post_val_accuracy": val_accuracy,
                    "val_accuracy_drop": val_drop,
                    "budget_0_07_applicable": budgeted,
                    "budget_violation": budgeted and val_drop > 0.07,
                    "dense_test_accuracy_reporting_only": dense_test,
                    "post_test_accuracy_reporting_only": float(row["test_accuracy"]),
                    "post_test_macro_f1_reporting_only": float(row["test_macro_f1"]),
                    "stop_reason": stop_reason,
                    "final_hidden_widths": json.dumps(widths),
                    "surgery_mode": SURGERY[method],
                    **checks,
                    "reload_error": reload_error,
                    "checkpoint": str(checkpoint),
                    "variant_dir": str(variant),
                }
            )

        condition_status = "completed" if not errors and all(r["status"] == "completed" for r in results if r["seed"] == seed and r["condition"] == condition) else "failed_quality_check"
        conditions.append(
            {
                "seed": seed,
                "dataset": dataset,
                "model": model,
                "condition": condition,
                "status": condition_status,
                "dense_params": dense_params,
                "dense_val_accuracy": dense_val,
                "dense_test_accuracy_reporting_only": dense_test,
                "split_sha256": split_hash,
                "dense_checkpoint_sha256": sha256(dense_checkpoint),
                **epochs,
                "errors": ";".join(errors),
            }
        )
    return results, conditions


def artifact_manifest() -> list[dict[str, object]]:
    manifest = ROOT / "audit/artifact_manifest.csv"
    rows = []
    for path in sorted(ROOT.rglob("*")):
        if path.is_file() and path != manifest:
            rows.append({"relative_path": str(path.relative_to(ROOT)), "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    return rows


def main() -> int:
    if os.environ.get("GNN_ALLOW_FROZEN_ARTIFACT_WRITE") != "YES":
        raise SystemExit(
            "Refusing to rewrite the frozen campaign. Set "
            "GNN_ALLOW_FROZEN_ARTIFACT_WRITE=YES only for an explicitly authorized audit refresh."
        )
    results, conditions = audit()
    audit_dir = ROOT / "audit"
    write_csv(audit_dir / "method_target_matrix.csv", results)
    write_csv(audit_dir / "condition_matrix.csv", conditions)
    completed = [row for row in results if row["status"] == "completed"]
    summary = {
        "git_commit": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
        "planned_method_target_results": len(SEEDS) * len(DATASETS) * len(MODELS) * len(METHODS) * len(TARGETS),
        "completed_method_target_results": len(completed),
        "failed_quality_check_results": sum(row["status"] == "failed_quality_check" for row in results),
        "missing_method_target_results": sum(row["status"] == "missing" for row in results),
        "completed_dense_conditions": sum(row["status"] == "completed" for row in conditions),
        "missing_dense_conditions": sum(row["status"] == "missing" for row in conditions),
        "target_matches_within_0_025": sum(bool(row.get("target_within_0_025")) for row in completed),
        "budget_violations": sum(bool(row.get("budget_violation")) for row in completed),
        "reloadable_results": sum(bool(row.get("reloadable")) for row in completed),
    }
    (audit_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(audit_dir / "artifact_manifest.csv", artifact_manifest())
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["failed_quality_check_results"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

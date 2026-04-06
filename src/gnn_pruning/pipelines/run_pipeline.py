"""Full pruning benchmark pipeline orchestration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from gnn_pruning.config import dump_yaml, load_yaml, resolve_config
from gnn_pruning.config.schema import snapshot_path
from gnn_pruning.data import load_split_indices
from gnn_pruning.pruning.workflow import finetune_pruned_checkpoint, prune_from_checkpoint
from gnn_pruning.reporting import write_pipeline_csv_rows
from gnn_pruning.training import evaluate_dense_and_save, train_dense
from gnn_pruning.utils import ProgressReporter, resolve_output_dir


@dataclass
class PipelineArtifacts:
    """Artifacts produced by one end-to-end pruning benchmark run."""

    output_dir: Path
    config_snapshot: Path
    split_artifact: Path
    dense_checkpoint_path: Path
    dense_metrics_path: Path
    csv_path: Path
    summary_path: Path


def run_pipeline(config_path: str, show_progress: bool = False) -> PipelineArtifacts:
    """Run dense + pruning benchmark pipeline with shared split/checkpoint."""
    resolved = resolve_config(config_path)
    output_dir = resolve_output_dir(
        configured_output_dir=resolved.run.output_dir,
        experiment_name=resolved.run.experiment_name,
        dataset_name=resolved.data.name,
        model_name=resolved.model.name,
        seed=resolved.run.seed,
        resume=False,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    reporter = ProgressReporter(enabled=show_progress, log_path=output_dir / "progress.log")

    reporter.stage(1, 6, "Training dense model")
    train_artifacts = train_dense(
        config_path=config_path,
        resume=True,
        output_dir_override=str(output_dir),
        progress_reporter=reporter,
    )
    reporter.stage(2, 6, "Evaluating dense model")
    eval_artifacts = evaluate_dense_and_save(
        config_path=config_path,
        checkpoint_path=str(train_artifacts.checkpoint_path),
        output_dir_override=str(output_dir),
        progress_reporter=reporter,
    )

    resolved.run.output_dir = str(output_dir)
    config_out = snapshot_path(output_dir)
    dump_yaml(resolved.to_dict(), config_out)

    raw_cfg = load_yaml(config_path)
    pruning_cfg = raw_cfg.get("pruning", {}) if isinstance(raw_cfg.get("pruning", {}), dict) else {}
    methods = _resolve_pruning_methods(pruning_cfg)
    sparsity_levels = _resolve_sparsity_levels(pruning_cfg)

    rows: List[Dict[str, Any]] = []
    dense_metrics = _read_json(eval_artifacts.metrics_path)
    rows.append(
        _build_csv_row(
            resolved=resolved,
            phase="dense",
            metrics=dense_metrics,
            pruning_method="dense",
            requested_sparsity=0.0,
            achieved_sparsity=0.0,
            checkpoint_path=train_artifacts.checkpoint_path,
            metrics_path=eval_artifacts.metrics_path,
            config_path=config_out,
            split_path=train_artifacts.split_path,
        )
    )
    _append_run_metadata(
        output_dir=output_dir,
        payload={
            "phase": "dense",
            "dataset": resolved.data.name,
            "model": resolved.model.name,
            "num_layers": resolved.model.num_layers,
            "hidden_channels": resolved.model.hidden_channels,
            "seed": resolved.run.seed,
            "split_hash": train_artifacts.split_hash,
            "dense_checkpoint_path": str(train_artifacts.checkpoint_path),
            "checkpoint_reused": train_artifacts.checkpoint_reused,
            "pruning_method": "dense",
            "sparsity": 0.0,
            "parameter_count": dense_metrics.get("benchmark", {}).get("parameter_count", ""),
        },
    )

    variant_summaries: List[Dict[str, Any]] = []
    total_variants = max(1, len(methods) * len(sparsity_levels))
    variant_index = 0
    for method in methods:
        for sparsity in sparsity_levels:
            variant_index += 1
            reporter.stage(3, 6, f"Pruning variant {variant_index}/{total_variants}: method={method} sparsity={sparsity}")
            variant_dir = output_dir / "pruning" / method / f"sparsity_{_slug_sparsity(sparsity)}"
            variant_dir.mkdir(parents=True, exist_ok=True)
            variant_reporter = ProgressReporter(enabled=show_progress, log_path=variant_dir / "progress.log")
            variant_config = _write_variant_config(
                source_cfg=raw_cfg,
                destination=variant_dir / "pipeline_variant.yaml",
                output_dir=variant_dir,
                method=method,
                sparsity=sparsity,
            )
            _copy_split(train_artifacts.split_path, variant_dir / "splits.yaml")

            prune_artifacts = prune_from_checkpoint(
                checkpoint_path=str(train_artifacts.checkpoint_path),
                config_path=str(variant_config),
                progress_reporter=variant_reporter,
            )
            reporter.stage(4, 6, f"Post-prune evaluation done for {method}@{sparsity}")
            finetune_artifacts = finetune_pruned_checkpoint(
                checkpoint_path=str(prune_artifacts.pruned_checkpoint_path),
                config_path=str(variant_config),
                finetune_epochs=None,
                progress_reporter=variant_reporter,
            )
            reporter.stage(5, 6, f"Post-finetune evaluation done for {method}@{sparsity}")

            prune_metrics = _read_json(prune_artifacts.post_prune_metrics_path) if prune_artifacts.post_prune_metrics_path else {}
            finetune_metrics = _read_json(finetune_artifacts.post_finetune_metrics_path)
            pruning_plan = _read_json(prune_artifacts.pruning_metrics_path)

            achieved_sparsity = float(pruning_plan.get("achieved_sparsity", sparsity))
            rows.append(
                _build_csv_row(
                    resolved=resolved,
                    phase="post_prune",
                    metrics=prune_metrics,
                    pruning_method=method,
                    requested_sparsity=sparsity,
                    achieved_sparsity=achieved_sparsity,
                    checkpoint_path=prune_artifacts.pruned_checkpoint_path,
                    metrics_path=prune_artifacts.post_prune_metrics_path or prune_artifacts.pruning_metrics_path,
                    config_path=variant_config,
                    split_path=variant_dir / "splits.yaml",
                )
            )
            _append_run_metadata(
                output_dir=variant_dir,
                payload={
                    "phase": "post_prune",
                    "dataset": resolved.data.name,
                    "model": resolved.model.name,
                    "num_layers": resolved.model.num_layers,
                    "hidden_channels": resolved.model.hidden_channels,
                    "seed": resolved.run.seed,
                    "split_hash": train_artifacts.split_hash,
                    "dense_checkpoint_path": str(train_artifacts.checkpoint_path),
                    "checkpoint_reused": train_artifacts.checkpoint_reused,
                    "pruning_method": method,
                    "sparsity": sparsity,
                    "parameter_count": prune_metrics.get("benchmark", {}).get("parameter_count", ""),
                },
            )
            rows.append(
                _build_csv_row(
                    resolved=resolved,
                    phase="post_finetune",
                    metrics=finetune_metrics,
                    pruning_method=method,
                    requested_sparsity=sparsity,
                    achieved_sparsity=achieved_sparsity,
                    checkpoint_path=finetune_artifacts.post_finetune_checkpoint_path,
                    metrics_path=finetune_artifacts.post_finetune_metrics_path,
                    config_path=variant_config,
                    split_path=variant_dir / "splits.yaml",
                )
            )
            _append_run_metadata(
                output_dir=variant_dir,
                payload={
                    "phase": "post_finetune",
                    "dataset": resolved.data.name,
                    "model": resolved.model.name,
                    "num_layers": resolved.model.num_layers,
                    "hidden_channels": resolved.model.hidden_channels,
                    "seed": resolved.run.seed,
                    "split_hash": train_artifacts.split_hash,
                    "dense_checkpoint_path": str(train_artifacts.checkpoint_path),
                    "checkpoint_reused": train_artifacts.checkpoint_reused,
                    "pruning_method": method,
                    "sparsity": sparsity,
                    "parameter_count": finetune_metrics.get("benchmark", {}).get("parameter_count", ""),
                },
            )
            variant_summaries.append(
                {
                    "method": method,
                    "sparsity": sparsity,
                    "achieved_sparsity": achieved_sparsity,
                    "post_prune_test_accuracy": _metric(prune_metrics, "test", "accuracy"),
                    "post_finetune_test_accuracy": _metric(finetune_metrics, "test", "accuracy"),
                    "variant_dir": str(variant_dir),
                }
            )

    csv_path = write_pipeline_csv_rows(rows, output_dir / "pipeline_results.csv")
    summary_path = output_dir / "summary_pipeline.md"
    summary_path.write_text(
        _build_summary_markdown(
            resolved.run.experiment_name,
            resolved.data.name,
            resolved.model.name,
            dense_metrics,
            variant_summaries,
        ),
        encoding="utf-8",
    )
    reporter.stage(6, 6, "Saving artifacts")

    return PipelineArtifacts(
        output_dir=output_dir,
        config_snapshot=config_out,
        split_artifact=train_artifacts.split_path,
        dense_checkpoint_path=train_artifacts.checkpoint_path,
        dense_metrics_path=eval_artifacts.metrics_path,
        csv_path=csv_path,
        summary_path=summary_path,
    )


def _resolve_pruning_methods(pruning_cfg: Dict[str, Any]) -> List[str]:
    methods = pruning_cfg.get("methods")
    if isinstance(methods, list) and methods:
        return [str(m) for m in methods]
    return [str(pruning_cfg.get("method", "global_magnitude"))]


def _resolve_sparsity_levels(pruning_cfg: Dict[str, Any]) -> List[float]:
    configured = pruning_cfg.get("sparsity_levels")
    if isinstance(configured, list) and configured:
        return [float(level) for level in configured]
    return [0.1, 0.3, 0.5, 0.7, 0.9]


def _copy_split(source: Path, destination: Path) -> None:
    split = load_split_indices(source)
    payload = {
        "train": [int(v) for v in split.train],
        "val": [int(v) for v in split.val],
        "test": [int(v) for v in split.test],
    }
    dump_yaml(payload, destination)


def _write_variant_config(
    source_cfg: Dict[str, Any],
    destination: Path,
    output_dir: Path,
    method: str,
    sparsity: float,
) -> Path:
    cfg = dict(source_cfg)
    cfg["run"] = dict(source_cfg.get("run", {}))
    cfg["run"]["output_dir"] = str(output_dir)
    cfg["pruning"] = dict(source_cfg.get("pruning", {}))
    cfg["pruning"]["method"] = method
    cfg["pruning"]["target_sparsity"] = float(sparsity)
    dump_yaml(cfg, destination)
    return destination


def _build_csv_row(
    resolved: Any,
    phase: str,
    metrics: Dict[str, Any],
    pruning_method: str,
    requested_sparsity: float,
    achieved_sparsity: float,
    checkpoint_path: Path,
    metrics_path: Path,
    config_path: Path,
    split_path: Path,
) -> Dict[str, Any]:
    return {
        "experiment_name": resolved.run.experiment_name,
        "dataset": resolved.data.name,
        "model": resolved.model.name,
        "seed": resolved.run.seed,
        "phase": phase,
        "pruning_method": pruning_method,
        "requested_sparsity": requested_sparsity,
        "achieved_sparsity": achieved_sparsity,
        "train_accuracy": _metric(metrics, "train", "accuracy"),
        "train_macro_f1": _metric(metrics, "train", "macro_f1"),
        "val_accuracy": _metric(metrics, "val", "accuracy"),
        "val_macro_f1": _metric(metrics, "val", "macro_f1"),
        "test_accuracy": _metric(metrics, "test", "accuracy"),
        "test_macro_f1": _metric(metrics, "test", "macro_f1"),
        "inference_time_mean_ms": _metric(metrics, "benchmark", "inference_time_mean_ms"),
        "inference_time_std_ms": _metric(metrics, "benchmark", "inference_time_std_ms"),
        "parameter_count": _metric(metrics, "benchmark", "parameter_count"),
        "checkpoint_path": str(checkpoint_path),
        "metrics_path": str(metrics_path),
        "config_path": str(config_path),
        "split_path": str(split_path),
    }


def _metric(metrics: Dict[str, Any], split: str, key: str) -> Any:
    split_payload = metrics.get(split, {})
    if isinstance(split_payload, dict):
        return split_payload.get(key, "")
    return ""


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _slug_sparsity(value: float) -> str:
    return str(value).replace(".", "_")


def _build_summary_markdown(
    experiment_name: str,
    dataset_name: str,
    model_name: str,
    dense_metrics: Dict[str, Any],
    variants: List[Dict[str, Any]],
) -> str:
    lines = [
        "# Pruning Benchmark Summary",
        "",
        f"- Experiment: `{experiment_name}`",
        f"- Dataset: `{dataset_name}`",
        f"- Model: `{model_name}`",
        "",
        "## Dense Baseline",
        "",
        f"- Test Accuracy: {_metric(dense_metrics, 'test', 'accuracy')}",
        f"- Test Macro F1: {_metric(dense_metrics, 'test', 'macro_f1')}",
        "",
        "## Pruning Variants",
        "",
    ]

    for variant in variants:
        lines.extend(
            [
                f"- `{variant['method']}` @ sparsity `{variant['sparsity']}` "
                f"(achieved `{variant['achieved_sparsity']}`): "
                f"post-prune acc={variant['post_prune_test_accuracy']}, "
                f"post-finetune acc={variant['post_finetune_test_accuracy']}",
            ]
        )

    return "\n".join(lines) + "\n"


def _append_run_metadata(output_dir: Path, payload: Dict[str, Any]) -> None:
    path = output_dir / "run_metadata.json"
    existing: list[Dict[str, Any]] = []
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
            if isinstance(loaded, list):
                existing = loaded
    existing.append(payload)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(existing, handle, indent=2)

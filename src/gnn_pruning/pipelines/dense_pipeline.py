"""Full dense experiment pipeline orchestration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from gnn_pruning.config import resolve_config
from gnn_pruning.reporting import write_csv_row
from gnn_pruning.training import evaluate_dense_and_save, train_dense
from gnn_pruning.utils import resolve_output_dir


@dataclass
class DensePipelineArtifacts:
    """Artifacts from `run-dense` pipeline."""

    output_dir: Path
    checkpoint_path: Path
    train_metrics_path: Path
    eval_metrics_path: Path
    config_snapshot_path: Path
    split_path: Path
    summary_path: Path
    csv_path: Path


def run_dense_pipeline(config_path: str) -> DensePipelineArtifacts:
    """Run full dense pipeline and save structured artifacts."""
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

    train_artifacts = train_dense(config_path=config_path, resume=True, output_dir_override=str(output_dir))
    eval_artifacts = evaluate_dense_and_save(
        config_path=config_path,
        checkpoint_path=str(train_artifacts.checkpoint_path),
        output_dir_override=str(output_dir),
    )

    train_metrics = _read_json(train_artifacts.metrics_path)
    eval_metrics = _read_json(eval_artifacts.metrics_path)

    summary_path = output_dir / "summary.md"
    summary_path.write_text(
        _build_summary_markdown(resolved.run.experiment_name, resolved.data.name, resolved.model.name, train_metrics, eval_metrics),
        encoding="utf-8",
    )

    csv_row = {
        "experiment_name": resolved.run.experiment_name,
        "dataset": resolved.data.name,
        "model": resolved.model.name,
        "seed": resolved.run.seed,
        "train_accuracy": eval_metrics["train"]["accuracy"],
        "train_macro_f1": eval_metrics["train"]["macro_f1"],
        "val_accuracy": eval_metrics["val"]["accuracy"],
        "val_macro_f1": eval_metrics["val"]["macro_f1"],
        "test_accuracy": eval_metrics["test"]["accuracy"],
        "test_macro_f1": eval_metrics["test"]["macro_f1"],
        "best_epoch": train_metrics.get("training", {}).get("best_epoch", ""),
        "epochs_ran": train_metrics.get("training", {}).get("epochs_ran", ""),
        "checkpoint_path": str(train_artifacts.checkpoint_path),
        "metrics_path": str(eval_artifacts.metrics_path),
        "config_path": str(train_artifacts.resolved_config_path),
        "split_path": str(train_artifacts.split_path),
    }
    csv_path = write_csv_row(csv_row, output_dir / "dense_results.csv")

    return DensePipelineArtifacts(
        output_dir=output_dir,
        checkpoint_path=train_artifacts.checkpoint_path,
        train_metrics_path=train_artifacts.metrics_path,
        eval_metrics_path=eval_artifacts.metrics_path,
        config_snapshot_path=train_artifacts.resolved_config_path,
        split_path=train_artifacts.split_path,
        summary_path=summary_path,
        csv_path=csv_path,
    )


def _read_json(path: Path) -> Dict[str, Dict[str, float]]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_summary_markdown(
    experiment_name: str,
    dataset_name: str,
    model_name: str,
    train_metrics: Dict[str, Dict[str, float]],
    eval_metrics: Dict[str, Dict[str, float]],
) -> str:
    training = train_metrics.get("training", {})
    return "\n".join(
        [
            "# Dense Experiment Summary",
            "",
            f"- Experiment: `{experiment_name}`",
            f"- Dataset: `{dataset_name}`",
            f"- Model: `{model_name}`",
            "",
            "## Evaluation Metrics",
            "",
            f"- Train Accuracy: {eval_metrics['train']['accuracy']:.4f}",
            f"- Val Accuracy: {eval_metrics['val']['accuracy']:.4f}",
            f"- Test Accuracy: {eval_metrics['test']['accuracy']:.4f}",
            f"- Train Macro F1: {eval_metrics['train']['macro_f1']:.4f}",
            f"- Val Macro F1: {eval_metrics['val']['macro_f1']:.4f}",
            f"- Test Macro F1: {eval_metrics['test']['macro_f1']:.4f}",
            "",
            "## Training",
            "",
            f"- Best Epoch: {training.get('best_epoch', 'n/a')}",
            f"- Epochs Ran: {training.get('epochs_ran', 'n/a')}",
            f"- Best Val Loss: {training.get('best_val_loss', 'n/a')}",
        ]
    ) + "\n"

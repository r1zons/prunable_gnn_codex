"""Command-line interface for the gnn_pruning package."""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

from .config import resolve_config
from .pipelines import run_dense_pipeline, run_pipeline, run_suite
from .pruning import list_pruners
from .pruning.workflow import evaluate_pruned_checkpoint, finetune_pruned_checkpoint, prune_from_checkpoint
from .training import evaluate_dense_and_save, train_dense
from .utils import ProgressReporter, resolve_show_progress


def build_parser() -> argparse.ArgumentParser:
    """Create and return the root CLI parser."""
    parser = argparse.ArgumentParser(prog="gnn_pruning")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the full pruning benchmark pipeline.")
    run_parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML config.")
    _add_progress_flag(run_parser)

    run_pipeline_parser = subparsers.add_parser("run-pipeline", help="Run the full pruning benchmark pipeline.")
    run_pipeline_parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML config.")
    _add_progress_flag(run_pipeline_parser)

    run_suite_parser = subparsers.add_parser("run-suite", help="Run repeated benchmark suite and aggregate results.")
    run_suite_parser.add_argument("--config", type=str, required=True, help="Path to suite YAML config.")
    _add_progress_flag(run_suite_parser)

    run_dense_parser = subparsers.add_parser("run-dense", help="Run full dense experiment pipeline.")
    run_dense_parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML config.")
    _add_progress_flag(run_dense_parser)

    train_parser = subparsers.add_parser("train", help="Train dense model.")
    train_parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML config.")
    train_parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable checkpoint resume behavior.",
    )
    _add_progress_flag(train_parser)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate dense checkpoint.")
    eval_parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML config.")
    eval_parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint path; defaults to run.output_dir/dense_checkpoint.pt",
    )

    prune_parser = subparsers.add_parser("prune", help="Apply pruning to a dense checkpoint.")
    prune_parser.add_argument("--checkpoint", type=str, required=True, help="Path to dense checkpoint.")
    prune_parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML config.")

    eval_pruned_parser = subparsers.add_parser("evaluate-pruned", help="Evaluate a pruned checkpoint.")
    eval_pruned_parser.add_argument("--checkpoint", type=str, required=True, help="Path to pruned checkpoint.")
    eval_pruned_parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML config.")

    finetune_parser = subparsers.add_parser("finetune", help="Fine-tune a pruned checkpoint.")
    finetune_parser.add_argument("--checkpoint", type=str, required=True, help="Path to pruned checkpoint.")
    finetune_parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML config.")
    finetune_parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional fine-tune epochs override (default: pruning.finetune_epochs or 50).",
    )
    _add_progress_flag(finetune_parser)

    show_parser = subparsers.add_parser("show-config", help="Resolve and print effective config.")
    show_parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML config.")

    subparsers.add_parser("version", help="Print package version.")
    subparsers.add_parser("list-pruners", help="List registered pruner placeholders.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Execute the CLI entrypoint and return process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "version":
        from . import __version__

        print(__version__)
        return 0

    if args.command == "show-config":
        resolved = resolve_config(args.config)
        print(resolved.to_dict())
        return 0

    if args.command == "list-pruners":
        records = list_pruners()
        if not records:
            print("[gnn_pruning] no pruners registered yet")
            return 0
        for row in records:
            print(row)
        return 0

    if args.command == "prune":
        artifacts = prune_from_checkpoint(checkpoint_path=args.checkpoint, config_path=args.config)
        print(f"[gnn_pruning] pruned checkpoint saved to: {artifacts.pruned_checkpoint_path}")
        print(f"[gnn_pruning] pruning metrics saved to: {artifacts.pruning_metrics_path}")
        if artifacts.post_prune_metrics_path is not None:
            print(f"[gnn_pruning] post-prune metrics saved to: {artifacts.post_prune_metrics_path}")
        return 0

    if args.command == "evaluate-pruned":
        artifacts = evaluate_pruned_checkpoint(checkpoint_path=args.checkpoint, config_path=args.config)
        print(f"[gnn_pruning] pruned evaluation metrics saved to: {artifacts.metrics_path}")
        return 0

    if args.command == "finetune":
        show_progress = resolve_show_progress(args.config, args.progress)
        if show_progress:
            reporter = ProgressReporter(enabled=True)
            artifacts = finetune_pruned_checkpoint(
                checkpoint_path=args.checkpoint,
                config_path=args.config,
                finetune_epochs=args.epochs,
                progress_reporter=reporter,
            )
        else:
            artifacts = finetune_pruned_checkpoint(
                checkpoint_path=args.checkpoint,
                config_path=args.config,
                finetune_epochs=args.epochs,
            )
        print(f"[gnn_pruning] pre-finetune checkpoint saved to: {artifacts.pre_finetune_checkpoint_path}")
        print(f"[gnn_pruning] post-finetune checkpoint saved to: {artifacts.post_finetune_checkpoint_path}")
        print(f"[gnn_pruning] pre-finetune metrics saved to: {artifacts.pre_finetune_metrics_path}")
        print(f"[gnn_pruning] post-finetune metrics saved to: {artifacts.post_finetune_metrics_path}")
        return 0

    if args.command in {"run", "run-pipeline"}:
        show_progress = resolve_show_progress(args.config, args.progress)
        artifacts = run_pipeline(args.config, show_progress=True) if show_progress else run_pipeline(args.config)
        print(f"[gnn_pruning] pipeline invoked with config: {args.config}")
        print(f"[gnn_pruning] output dir: {artifacts.output_dir}")
        print(f"[gnn_pruning] resolved config snapshot saved to: {artifacts.config_snapshot}")
        print(f"[gnn_pruning] split artifact saved to: {artifacts.split_artifact}")
        print(f"[gnn_pruning] dense checkpoint: {artifacts.dense_checkpoint_path}")
        print(f"[gnn_pruning] dense metrics: {artifacts.dense_metrics_path}")
        print(f"[gnn_pruning] csv: {artifacts.csv_path}")
        print(f"[gnn_pruning] summary: {artifacts.summary_path}")
        return 0

    if args.command == "run-suite":
        show_progress = resolve_show_progress(args.config, args.progress)
        artifacts = run_suite(args.config, show_progress=True) if show_progress else run_suite(args.config)
        print(f"[gnn_pruning] suite output dir: {artifacts.output_dir}")
        print(f"[gnn_pruning] runs csv: {artifacts.runs_csv_path}")
        print(f"[gnn_pruning] aggregate csv: {artifacts.aggregate_csv_path}")
        return 0

    if args.command == "run-dense":
        show_progress = resolve_show_progress(args.config, args.progress)
        artifacts = run_dense_pipeline(args.config, show_progress=True) if show_progress else run_dense_pipeline(args.config)
        print(f"[gnn_pruning] dense pipeline complete at: {artifacts.output_dir}")
        print(f"[gnn_pruning] checkpoint: {artifacts.checkpoint_path}")
        print(f"[gnn_pruning] eval metrics: {artifacts.eval_metrics_path}")
        print(f"[gnn_pruning] summary: {artifacts.summary_path}")
        print(f"[gnn_pruning] csv: {artifacts.csv_path}")
        return 0

    if args.command == "train":
        show_progress = resolve_show_progress(args.config, args.progress)
        if show_progress:
            reporter = ProgressReporter(enabled=True)
            artifacts = train_dense(config_path=args.config, resume=not args.no_resume, progress_reporter=reporter)
        else:
            artifacts = train_dense(config_path=args.config, resume=not args.no_resume)
        print(f"[gnn_pruning] dense checkpoint saved to: {artifacts.checkpoint_path}")
        print(f"[gnn_pruning] metrics saved to: {artifacts.metrics_path}")
        print(f"[gnn_pruning] resolved config saved to: {artifacts.resolved_config_path}")
        return 0

    if args.command == "evaluate":
        artifacts = evaluate_dense_and_save(config_path=args.config, checkpoint_path=args.checkpoint)
        print(f"[gnn_pruning] evaluation metrics saved to: {artifacts.metrics_path}")
        print(f"[gnn_pruning] resolved config saved to: {artifacts.resolved_config_path}")
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


def _add_progress_flag(parser: argparse.ArgumentParser) -> None:
    """Add argparse-3.8-compatible --progress/--no-progress flags."""
    parser.add_argument(
        "--progress",
        dest="progress",
        action="store_const",
        const=True,
        default=None,
        help="Enable live progress output.",
    )
    parser.add_argument(
        "--no-progress",
        dest="progress",
        action="store_const",
        const=False,
        default=None,
        help="Disable live progress output (overrides config).",
    )

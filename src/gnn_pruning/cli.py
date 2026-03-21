"""Command-line interface for the gnn_pruning package."""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

from .config import resolve_config
from .pipelines import run_dense_pipeline, run_pipeline
from .training import evaluate_dense_and_save, train_dense


def build_parser() -> argparse.ArgumentParser:
    """Create and return the root CLI parser."""
    parser = argparse.ArgumentParser(prog="gnn_pruning")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the default pipeline scaffold.")
    run_parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML config.")

    run_dense_parser = subparsers.add_parser("run-dense", help="Run full dense experiment pipeline.")
    run_dense_parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML config.")

    train_parser = subparsers.add_parser("train", help="Train dense model.")
    train_parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML config.")
    train_parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable checkpoint resume behavior.",
    )

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate dense checkpoint.")
    eval_parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML config.")
    eval_parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint path; defaults to run.output_dir/dense_checkpoint.pt",
    )

    show_parser = subparsers.add_parser("show-config", help="Resolve and print effective config.")
    show_parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML config.")

    subparsers.add_parser("version", help="Print package version.")
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

    if args.command == "run":
        artifacts = run_pipeline(args.config)
        print(f"[gnn_pruning] scaffold pipeline invoked with config: {args.config}")
        print(f"[gnn_pruning] resolved config snapshot saved to: {artifacts.config_snapshot}")
        print(f"[gnn_pruning] split artifact saved to: {artifacts.split_artifact}")
        return 0

    if args.command == "run-dense":
        artifacts = run_dense_pipeline(args.config)
        print(f"[gnn_pruning] dense pipeline complete at: {artifacts.output_dir}")
        print(f"[gnn_pruning] checkpoint: {artifacts.checkpoint_path}")
        print(f"[gnn_pruning] eval metrics: {artifacts.eval_metrics_path}")
        print(f"[gnn_pruning] summary: {artifacts.summary_path}")
        print(f"[gnn_pruning] csv: {artifacts.csv_path}")
        return 0

    if args.command == "train":
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

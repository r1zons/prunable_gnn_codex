"""Command-line interface for the gnn_pruning package."""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

from .config import resolve_config
from .pipelines import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    """Create and return the root CLI parser."""
    parser = argparse.ArgumentParser(prog="gnn_pruning")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the default pipeline scaffold.")
    run_parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML config.")

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

    parser.error(f"Unsupported command: {args.command}")
    return 2

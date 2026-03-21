"""Command-line interface for the gnn_pruning package."""

from __future__ import annotations

import argparse
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    """Create and return the root CLI parser."""
    parser = argparse.ArgumentParser(prog="gnn_pruning")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the default pipeline scaffold.")
    run_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment YAML config.",
    )

    subparsers.add_parser("version", help="Print package version.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the CLI entrypoint and return process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "version":
        from . import __version__

        print(__version__)
        return 0

    if args.command == "run":
        print(f"[gnn_pruning] scaffold pipeline invoked with config: {args.config}")
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2

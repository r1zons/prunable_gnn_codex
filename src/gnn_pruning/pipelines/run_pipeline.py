"""Pipeline orchestration scaffolding."""

from __future__ import annotations

from pathlib import Path

from gnn_pruning.config import load_config


def run_pipeline(config_path: str) -> None:
    """Load config and execute placeholder pipeline steps."""
    _ = load_config(Path(config_path))
    return None

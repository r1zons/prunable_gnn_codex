"""Pipeline orchestration scaffolding."""

from __future__ import annotations

from pathlib import Path

from gnn_pruning.config import dump_yaml, resolve_config
from gnn_pruning.config.schema import snapshot_path


def run_pipeline(config_path: str) -> Path:
    """Load config and execute placeholder pipeline steps.

    Saves the fully resolved config to the run output directory.
    """
    resolved = resolve_config(config_path)
    output_path = snapshot_path(resolved.run.output_dir)
    dump_yaml(resolved.to_dict(), output_path)
    return output_path

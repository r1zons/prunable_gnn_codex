"""Run output directory naming and collision handling."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional, Union


def resolve_output_dir(
    configured_output_dir: str,
    experiment_name: str,
    dataset_name: str,
    model_name: str,
    seed: int,
    resume: bool,
    now: Optional[datetime] = None,
) -> Path:
    """Resolve output directory with naming and collision-avoidance rules."""
    configured = Path(configured_output_dir).expanduser()

    if configured_output_dir and configured_output_dir != "runs/default":
        if resume:
            return configured
        if configured.exists():
            return _with_run_index(configured)
        return configured

    timestamp = (now or datetime.utcnow()).strftime("%H_%M_%d%m%Y")
    root = Path("runs")
    base_name = f"{experiment_name}_{dataset_name}_{model_name}_{timestamp}_{seed}"
    candidate = root / base_name

    if resume and candidate.exists():
        return candidate
    if candidate.exists():
        return _with_run_index(candidate)
    return candidate


def _with_run_index(path: Path) -> Path:
    idx = 1
    while True:
        candidate = Path(f"{path}_{idx:03d}")
        if not candidate.exists():
            return candidate
        idx += 1

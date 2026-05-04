"""Convenience runner for adaptive pruning experiments."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEBUG_CONFIG = REPO_ROOT / "configs/experiments/adaptive_graphsage_citeseer_debug.yaml"
FULL_CONFIG = REPO_ROOT / "configs/experiments/adaptive_graphsage_citeseer.yaml"
SUITE_CONFIG = REPO_ROOT / "configs/suites/adaptive_comparison_small.yaml"


def _run_command(command: list[str]) -> None:
    print(f"\n[run] {' '.join(command)}", flush=True)
    subprocess.run(command, check=True, cwd=REPO_ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run adaptive pruning comparison pipelines.")
    parser.add_argument("--debug-only", action="store_true", help="Run only the lightweight adaptive debug pipeline.")
    parser.add_argument("--full", action="store_true", help="Run debug first, then full citeseer and suite comparisons.")
    args = parser.parse_args()

    if not args.debug_only and not args.full:
        parser.error("Choose one: --debug-only or --full")

    _run_command(["python", "-u", "-m", "gnn_pruning", "run-pipeline", "--config", str(DEBUG_CONFIG), "--progress"])
    print(f"[ok] Debug run finished. Output under: runs/adaptive_graphsage_citeseer_debug", flush=True)

    if args.full:
        _run_command(["python", "-u", "-m", "gnn_pruning", "run-pipeline", "--config", str(FULL_CONFIG), "--progress"])
        print(f"[ok] Full Citeseer comparison finished. Output under: runs/adaptive_graphsage_citeseer", flush=True)
        _run_command(["python", "-u", "-m", "gnn_pruning", "run-suite", "--config", str(SUITE_CONFIG), "--progress"])
        print(f"[ok] Suite comparison finished. Output under: runs/suites/adaptive_comparison_small", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

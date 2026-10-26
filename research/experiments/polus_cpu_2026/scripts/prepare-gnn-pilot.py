"""Prepare external Polus configs; optionally invoke the existing pilot validator.

Default operation only prepares configuration/provenance files. No model, dataset
or training imports occur until --run is explicitly supplied by the researcher.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import yaml


CONDITIONS = tuple(
    "{}_{}_l3_h32".format(dataset, model)
    for dataset in ("cora", "citeseer", "pubmed")
    for model in ("gcn", "graphsage")
)


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def merge(left: dict, right: dict) -> dict:
    result = dict(left)
    for key, value in right.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = value
    return result


def adapt_runner_log_directory(runner) -> None:
    """Keep launcher logs outside the output directory until the pipeline creates it.

    The pinned runner pre-creates the output, while run_pipeline(resume=False)
    then chooses a suffixed directory. Only launcher filesystem orchestration
    is adapted here; training, pruning and artifact validators stay unchanged.
    """
    source = inspect.getsource(runner._run_condition)
    original = ('    condition_dir.mkdir(parents=True, exist_ok=False)\n'
                '    log_path = condition_dir / "launcher.log"\n')
    replacement = ('    log_dir = OUTPUT_ROOT / "launcher-logs"\n'
                   '    log_dir.mkdir(parents=True, exist_ok=True)\n'
                   '    log_path = log_dir / (condition + ".log")\n')
    if source.count(original) != 1:
        raise RuntimeError("Pinned runner log-directory code changed; review adapter")
    exec(compile(source.replace(original, replacement), '<polus-launcher-adapter>', 'exec'), runner.__dict__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--run", action="store_true", help="Explicitly start one pilot condition.")
    args = parser.parse_args()
    repo = args.repo_root.expanduser().resolve()
    output = args.output_root.expanduser().resolve()
    data = args.data_root.expanduser().resolve()
    commit = git(repo, "rev-parse", "HEAD")
    if commit != args.expected_commit:
        raise RuntimeError("Repository commit differs from --expected-commit")
    if git(repo, "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("Repository has tracked changes")
    if args.run and not os.environ.get("LSB_JOBID"):
        raise RuntimeError("Start an LSF CPU allocation before using --run")
    if output == repo or repo in output.parents:
        raise RuntimeError("Keep outputs/configs outside the pinned repository")

    base_path = repo / "configs/experiments/kaggle_pilot/base.yaml"
    source_path = repo / "configs/experiments/kaggle_pilot" / (args.condition + ".yaml")
    base = yaml.safe_load(base_path.read_text())
    original = yaml.safe_load(source_path.read_text())
    prepared = merge(base, original)
    prepared["base"] = None
    prepared["data"]["root"] = str(data)
    prepared["run"]["output_dir"] = str(output / "seed_101" / args.condition)
    prepared["run"]["experiment_name"] = "polus_cpu_{}_seed101".format(args.condition)
    prepared["device"] = {"device": "cpu"}
    config_dir = output / "prepared-configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    target = config_dir / (args.condition + ".yaml")
    text = yaml.safe_dump(prepared, sort_keys=False)
    if target.exists() and target.read_text() != text:
        raise RuntimeError("Existing prepared config differs; choose a fresh output root")
    target.write_text(text)
    manifest = {
        "platform": "polus_cpu",
        "git_commit": commit,
        "condition": args.condition,
        "config": str(target),
        "config_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "source_config_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "source_base_sha256": hashlib.sha256(base_path.read_bytes()).hexdigest(),
        "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "machine": platform.machine(),
        "python": sys.version,
        "execution_requested": args.run,
        "historical_or_kaggle_results_pooled": False,
        "runner_log_directory_fix": "launcher-logs outside condition output; scientific pipeline unchanged",
        "changed_fields": ["base flattened", "data.root", "run.output_dir", "run.experiment_name", "device.device"],
    }
    (config_dir / (args.condition + ".provenance.json")).write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    if not args.run:
        return 0

    # Reuse protocol checks, checkpoint pairing, artifact validation and failures.
    module_path = repo / "scripts/run_kaggle_pilot.py"
    spec = importlib.util.spec_from_file_location("polus_existing_pilot", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the existing pilot runner")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    adapt_runner_log_directory(runner)
    runner.OUTPUT_ROOT = output
    runner.CONFIGS = {args.condition: str(target)}
    return runner.main([
        "--repo-root", str(repo), "--expected-commit", commit,
        "--condition", args.condition,
    ])


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


REPO = Path(os.environ.get("GNN_REPO_ROOT", Path(__file__).resolve().parents[4]))
ARTIFACT_ROOT = Path(os.environ.get("GNN_ARTIFACT_ROOT", REPO.parent / "prunable_gnn_artifacts"))
ROOT = ARTIFACT_ROOT / "local-cpu-20261025" / "raw"
PYTHON = Path(os.environ.get("GNN_PYTHON", sys.executable))
SOURCE_CONFIG_DIR = REPO / "configs/experiments/kaggle_pilot"
CONDITIONS = (
    "cora_graphsage_l3_h32",
    "citeseer_gcn_l3_h32",
    "citeseer_graphsage_l3_h32",
    "pubmed_gcn_l3_h32",
    "pubmed_graphsage_l3_h32",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_config(condition: str) -> Path:
    source = SOURCE_CONFIG_DIR / f"{condition}.yaml"
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload["base"] = str(SOURCE_CONFIG_DIR / "base.yaml")
    payload["run"]["experiment_name"] = f"local_pilot_{condition}_seed101"
    payload["run"]["output_dir"] = str(ROOT / "results/seed_101" / condition)
    payload.setdefault("data", {})["root"] = str(REPO / "data")
    destination = ROOT / "configs" / f"{condition}_seed101.yaml"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return destination


def run_condition(condition: str) -> dict[str, object]:
    config = prepare_config(condition)
    output_dir = ROOT / "results/seed_101" / condition
    log_path = ROOT / "logs" / f"{condition}_seed101.launch.log"
    status_path = ROOT / "status" / f"{condition}.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        result = {
            "condition": condition,
            "status": "skipped_existing_output",
            "output_dir": str(output_dir),
            "config": str(config),
            "checked_at_utc": utc_now(),
        }
        status_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        return result
    command = [
        str(PYTHON),
        "-u",
        "-m",
        "gnn_pruning",
        "run-pipeline",
        "--config",
        str(config),
        "--progress",
    ]
    env = dict(os.environ)
    env.update(
        {
            "PYTHONPATH": str(REPO / "src"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
        }
    )
    started = utc_now()
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=REPO,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    result = {
        "condition": condition,
        "status": "completed" if completed.returncode == 0 else "failed",
        "return_code": completed.returncode,
        "started_at_utc": started,
        "finished_at_utc": utc_now(),
        "output_dir": str(output_dir),
        "config": str(config),
        "config_sha256": sha256(config),
        "log": str(log_path),
        "log_sha256": sha256(log_path),
    }
    status_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> int:
    if os.environ.get("GNN_ALLOW_COMPLETED_CAMPAIGN_RERUN") != "YES":
        raise SystemExit(
            "Campaign LOCAL-CPU-20261025 is complete. Refusing to rerun without "
            "GNN_ALLOW_COMPLETED_CAMPAIGN_RERUN=YES and explicit researcher authorization."
        )
    ROOT.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(run_condition, CONDITIONS))
    summary_path = ROOT / "status/remaining_seed101.json"
    summary_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(results, indent=2, sort_keys=True))
    return 1 if any(row["status"] == "failed" for row in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())

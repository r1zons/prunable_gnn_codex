"""Lightweight correctness audit for experiment identity and checkpoint isolation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gnn_pruning.config import resolve_config
from gnn_pruning.models import build_model
from gnn_pruning.training.checkpoints import load_checkpoint


@dataclass
class AuditRecord:
    """Resolved audit metadata for one config path."""

    config_path: str
    dataset: str
    model: str
    num_layers: int
    hidden_channels: int
    seed: int
    run_dir: Path
    dense_checkpoint_path: Path
    config_hash: str
    parameter_count: int


def audit_configs(config_paths: Sequence[str]) -> List[AuditRecord]:
    """Audit one or more configs and return resolved records."""
    records = [_audit_single(config_path) for config_path in config_paths]

    if len(records) > 1:
        _assert_multi_config_isolation(records)

    return records


def _audit_single(config_path: str) -> AuditRecord:
    resolved = resolve_config(config_path)
    run_dir = Path(resolved.run.output_dir).expanduser()
    dense_checkpoint_path = run_dir / "dense_checkpoint.pt"
    config_hash = hashlib.sha256(json.dumps(resolved.to_dict(), sort_keys=True).encode("utf-8")).hexdigest()

    model = build_model(
        resolved.model.name,
        in_channels=int(resolved.model.in_channels or 16),
        hidden_channels=int(resolved.model.hidden_channels),
        out_channels=int(resolved.model.out_channels or 7),
        num_layers=int(resolved.model.num_layers),
        dropout=float(resolved.model.dropout),
    )
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))

    print("=" * 90)
    print(f"config_path: {config_path}")
    print(f"dataset: {resolved.data.name}")
    print(f"model: {resolved.model.name}")
    print(f"num_layers: {resolved.model.num_layers}")
    print(f"hidden_channels: {resolved.model.hidden_channels}")
    print(f"seed: {resolved.run.seed}")
    print(f"run_dir: {run_dir}")
    print(f"dense_checkpoint_path: {dense_checkpoint_path}")
    print(f"config_hash: {config_hash}")
    print(f"parameter_count: {parameter_count}")

    if dense_checkpoint_path.exists():
        payload = load_checkpoint(dense_checkpoint_path, map_location=resolved.device.device)
        compatible, reason = _checkpoint_compatibility(payload=payload, expected=_expected_signature(resolved, config_hash))
        verdict = "ALLOWED" if compatible else "REJECTED"
        print(f"checkpoint_reuse: {verdict} ({reason or 'signature matches'})")
    else:
        print("checkpoint_reuse: checkpoint does not exist yet (would require fresh training run)")

    return AuditRecord(
        config_path=str(config_path),
        dataset=str(resolved.data.name),
        model=str(resolved.model.name),
        num_layers=int(resolved.model.num_layers),
        hidden_channels=int(resolved.model.hidden_channels),
        seed=int(resolved.run.seed),
        run_dir=run_dir,
        dense_checkpoint_path=dense_checkpoint_path,
        config_hash=config_hash,
        parameter_count=parameter_count,
    )


def _expected_signature(resolved: Any, config_hash: str) -> Dict[str, Any]:
    return {
        "dataset": resolved.data.name,
        "model_name": resolved.model.name,
        "num_layers": int(resolved.model.num_layers),
        "hidden_channels": int(resolved.model.hidden_channels),
        "dropout": float(resolved.model.dropout),
        "seed": int(resolved.run.seed),
        "config_hash": config_hash,
    }


def _checkpoint_compatibility(payload: Dict[str, Any], expected: Dict[str, Any]) -> tuple[bool, str]:
    saved = payload.get("run_signature")
    if not isinstance(saved, dict):
        return False, "missing run_signature"
    for key in ["dataset", "model_name", "num_layers", "hidden_channels", "dropout", "seed", "config_hash"]:
        if saved.get(key) != expected.get(key):
            return False, f"{key} differs (saved={saved.get(key)} expected={expected.get(key)})"
    return True, ""


def _assert_multi_config_isolation(records: Iterable[AuditRecord]) -> None:
    items = list(records)
    print("\nMulti-config audit summary:")
    print(f"- parameter_counts: {[item.parameter_count for item in items]}")
    print(f"- run_dirs: {[str(item.run_dir) for item in items]}")
    print(f"- dense_checkpoint_paths: {[str(item.dense_checkpoint_path) for item in items]}")

    for idx, left in enumerate(items):
        for right in items[idx + 1 :]:
            different_architecture = (
                left.model == right.model
                and (
                    left.num_layers != right.num_layers
                    or left.hidden_channels != right.hidden_channels
                )
            )
            if different_architecture and left.run_dir == right.run_dir:
                raise ValueError(
                    "Different architecture configs share the same run directory: "
                    f"{left.config_path} and {right.config_path} -> {left.run_dir}"
                )
            if different_architecture and left.dense_checkpoint_path == right.dense_checkpoint_path:
                raise ValueError(
                    "Different architecture configs share the same dense checkpoint path: "
                    f"{left.config_path} and {right.config_path} -> {left.dense_checkpoint_path}"
                )


def _collect_config_paths(config: Sequence[str] | None, configs: Sequence[str] | None) -> List[str]:
    collected: List[str] = []
    if config:
        collected.extend(str(path) for path in config)
    if configs:
        collected.extend(str(path) for path in configs)
    if not collected:
        raise ValueError("Provide at least one config path using --config or --configs.")
    return collected


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit experiment correctness assumptions.")
    parser.add_argument("--config", action="append", default=None, help="Single config path (can be repeated).")
    parser.add_argument("--configs", nargs="+", default=None, help="One or more config paths.")
    args = parser.parse_args(argv)

    config_paths = _collect_config_paths(args.config, args.configs)
    audit_configs(config_paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

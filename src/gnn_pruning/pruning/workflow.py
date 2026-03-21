"""Pruning workflow for applying pruners to dense checkpoints."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

import torch

from gnn_pruning.config import load_yaml, resolve_config
from gnn_pruning.models import build_model

from .base import PruningContext
from .registry import get_pruner


@dataclass
class PruningArtifacts:
    """Artifacts produced by prune workflow."""

    pruned_checkpoint_path: Path
    pruning_metrics_path: Path


def prune_from_checkpoint(checkpoint_path: str, config_path: str) -> PruningArtifacts:
    """Load dense checkpoint, apply selected pruner, and save pruned artifacts."""
    resolved = resolve_config(config_path)
    raw_cfg = load_yaml(config_path)
    prune_cfg: Dict[str, Any] = raw_cfg.get("pruning", {}) if isinstance(raw_cfg.get("pruning", {}), dict) else {}

    method = str(prune_cfg.get("method", "random"))
    target_sparsity = float(prune_cfg.get("target_sparsity", 0.5))
    structured = bool(prune_cfg.get("structured", True))

    checkpoint = torch.load(Path(checkpoint_path).expanduser(), map_location=resolved.device.device)

    model_name = checkpoint.get("model_name", resolved.model.name)
    model_cfg = checkpoint.get("model_config", {})
    if not model_cfg:
        raise ValueError("Checkpoint missing model_config required for pruning rebuild.")

    model = build_model(model_name, **model_cfg)
    model.load_state_dict(checkpoint["model_state_dict"])

    pruner_cls = get_pruner(method)
    pruner = pruner_cls()

    context = PruningContext(config=resolved.to_dict(), data=None, device=resolved.device.device, seed=resolved.run.seed)
    scores = pruner.score(model, context)
    pruned_model, plan = pruner.apply(
        model=model,
        scores=scores,
        context=context,
        target_sparsity=target_sparsity,
        structured=structured,
    )

    output_dir = Path(resolved.run.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    pruned_checkpoint_path = output_dir / f"pruned_{method}.pt"
    torch.save(
        {
            "model_name": model_name,
            "model_config": model_cfg,
            "source_checkpoint": str(Path(checkpoint_path).expanduser()),
            "model_state_dict": pruned_model.state_dict(),
            "pruning_plan": asdict(plan),
        },
        pruned_checkpoint_path,
    )

    pruning_metrics_path = output_dir / f"pruning_metrics_{method}.json"
    with pruning_metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(plan), handle, indent=2)

    return PruningArtifacts(
        pruned_checkpoint_path=pruned_checkpoint_path,
        pruning_metrics_path=pruning_metrics_path,
    )

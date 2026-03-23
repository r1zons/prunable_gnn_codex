"""Pruning workflow for applying pruners to dense checkpoints."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

import torch

from gnn_pruning.config import load_yaml, resolve_config
from gnn_pruning.data import generate_exact_ratio_split, load_dataset, load_split_indices, save_split_indices, to_index_tensors
from gnn_pruning.models import build_model
from gnn_pruning.training.trainer import DenseTrainer
from gnn_pruning.training.workflow import evaluate_dense

from .base import PruningContext
from .registry import get_pruner


@dataclass
class PruningArtifacts:
    """Artifacts produced by prune workflow."""

    pruned_checkpoint_path: Path
    pruning_metrics_path: Path
    post_prune_metrics_path: Path | None = None


@dataclass
class PrunedEvalArtifacts:
    """Artifacts produced by evaluate-pruned workflow."""

    metrics_path: Path


@dataclass
class FineTuneArtifacts:
    """Artifacts produced by finetuning a pruned checkpoint."""

    pre_finetune_checkpoint_path: Path
    post_finetune_checkpoint_path: Path
    pre_finetune_metrics_path: Path
    post_finetune_metrics_path: Path


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
            "model_config": _model_config_from_model(pruned_model, fallback=model_cfg),
            "source_checkpoint": str(Path(checkpoint_path).expanduser()),
            "model_state_dict": pruned_model.state_dict(),
            "pruning_plan": asdict(plan),
        },
        pruned_checkpoint_path,
    )

    pruning_metrics_path = output_dir / f"pruning_metrics_{method}.json"
    with pruning_metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(plan), handle, indent=2)

    split = _load_or_create_split(config_path=config_path, output_dir=output_dir)
    post_prune_metrics = evaluate_dense(
        config_path=config_path,
        model_override=pruned_model,
        split_override=split,
        output_dir_override=str(output_dir),
    )
    post_prune_metrics_path = output_dir / f"metrics_post_prune_{method}.json"
    with post_prune_metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(post_prune_metrics, handle, indent=2)

    return PruningArtifacts(
        pruned_checkpoint_path=pruned_checkpoint_path,
        pruning_metrics_path=pruning_metrics_path,
        post_prune_metrics_path=post_prune_metrics_path,
    )


def evaluate_pruned_checkpoint(checkpoint_path: str, config_path: str) -> PrunedEvalArtifacts:
    """Evaluate a pruned checkpoint and save metrics."""
    checkpoint = Path(checkpoint_path).expanduser()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Pruned checkpoint not found: {checkpoint}")
    output_dir = checkpoint.parent
    payload = _load_pruned_checkpoint_payload(checkpoint, resolve_config(config_path).device.device)
    model = build_model(payload["model_name"], **payload["model_config"])
    model.load_state_dict(payload["model_state_dict"])
    split = _load_or_create_split(config_path=config_path, output_dir=output_dir)
    metrics = evaluate_dense(config_path=config_path, model_override=model, split_override=split, output_dir_override=str(output_dir))
    metrics_path = output_dir / f"metrics_evaluate_{checkpoint.stem}.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    return PrunedEvalArtifacts(metrics_path=metrics_path)


def finetune_pruned_checkpoint(
    checkpoint_path: str,
    config_path: str,
    finetune_epochs: int | None = None,
) -> FineTuneArtifacts:
    """Fine-tune a pruned checkpoint, saving pre/post checkpoints and metrics."""
    resolved = resolve_config(config_path)
    raw_cfg = load_yaml(config_path)
    prune_cfg: Dict[str, Any] = raw_cfg.get("pruning", {}) if isinstance(raw_cfg.get("pruning", {}), dict) else {}
    epochs = int(finetune_epochs if finetune_epochs is not None else prune_cfg.get("finetune_epochs", 50))

    checkpoint = _load_pruned_checkpoint_payload(Path(checkpoint_path).expanduser(), resolved.device.device)
    output_dir = Path(checkpoint_path).expanduser().parent
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(resolved.data.name, resolved.data.root)
    data = dataset[0]

    split = _load_or_create_split(config_path=config_path, output_dir=output_dir, data=data)
    indices = to_index_tensors(split, device=resolved.device.device)

    model = build_model(checkpoint["model_name"], **checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])

    pre_finetune_checkpoint_path = output_dir / f"{Path(checkpoint_path).stem}_pre_finetune.pt"
    torch.save(checkpoint, pre_finetune_checkpoint_path)

    pre_metrics = evaluate_dense(
        config_path=config_path,
        model_override=model,
        split_override=split,
        output_dir_override=str(output_dir),
    )
    pre_finetune_metrics_path = output_dir / "metrics_pruned_pre_finetune.json"
    with pre_finetune_metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(pre_metrics, handle, indent=2)

    trainer = DenseTrainer(
        model=model,
        device=resolved.device.device,
        lr=resolved.training.lr,
        weight_decay=resolved.training.weight_decay,
        max_epochs=epochs,
        early_stopping_patience=resolved.training.early_stopping_patience,
    )
    train_result = trainer.fit(data=data, train_idx=indices["train"], val_idx=indices["val"])

    post_metrics = evaluate_dense(
        config_path=config_path,
        model_override=trainer.model,
        split_override=split,
        output_dir_override=str(output_dir),
    )
    post_metrics["finetune_training"] = train_result.to_dict()
    post_finetune_metrics_path = output_dir / "metrics_pruned_post_finetune.json"
    with post_finetune_metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(post_metrics, handle, indent=2)

    post_finetune_checkpoint_path = output_dir / f"{Path(checkpoint_path).stem}_post_finetune.pt"
    torch.save(
        {
            "model_name": checkpoint["model_name"],
            "model_config": _model_config_from_model(trainer.model, fallback=checkpoint["model_config"]),
            "source_checkpoint": str(Path(checkpoint_path).expanduser()),
            "model_state_dict": trainer.model.state_dict(),
            "finetune_epochs": epochs,
            "finetune_training": train_result.to_dict(),
        },
        post_finetune_checkpoint_path,
    )

    return FineTuneArtifacts(
        pre_finetune_checkpoint_path=pre_finetune_checkpoint_path,
        post_finetune_checkpoint_path=post_finetune_checkpoint_path,
        pre_finetune_metrics_path=pre_finetune_metrics_path,
        post_finetune_metrics_path=post_finetune_metrics_path,
    )


def _load_or_create_split(config_path: str, output_dir: Path, data: Any | None = None) -> Any:
    resolved = resolve_config(config_path)
    graph = data
    if graph is None:
        dataset = load_dataset(resolved.data.name, resolved.data.root)
        graph = dataset[0]

    split_path = output_dir / "splits.yaml"
    if split_path.exists():
        return load_split_indices(split_path)

    split = generate_exact_ratio_split(
        num_nodes=int(graph.num_nodes),
        seed=resolved.run.seed,
        train_ratio=resolved.data.train_ratio,
        val_ratio=resolved.data.val_ratio,
        test_ratio=resolved.data.test_ratio,
    )
    save_split_indices(split, output_dir)
    return split


def _load_pruned_checkpoint_payload(checkpoint_path: Path, map_location: str) -> Dict[str, Any]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Pruned checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=map_location)
    required = ("model_name", "model_config", "model_state_dict")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Invalid pruned checkpoint. Missing required keys: {missing}")
    return payload


def _model_config_from_model(model: torch.nn.Module, fallback: Dict[str, Any]) -> Dict[str, Any]:
    if hasattr(model, "convs") and len(model.convs) >= 2:
        conv0 = model.convs[0]
        conv1 = model.convs[1]
        if hasattr(conv0, "lin") and hasattr(conv1, "lin"):
            return {
                "in_channels": int(conv0.lin.weight.shape[1]),
                "hidden_channels": int(conv0.lin.weight.shape[0]),
                "out_channels": int(conv1.lin.weight.shape[0]),
                "num_layers": int(len(model.convs)),
                "dropout": float(getattr(model, "dropout", fallback.get("dropout", 0.0))),
            }
    return dict(fallback)

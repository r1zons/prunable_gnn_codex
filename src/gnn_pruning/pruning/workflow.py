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
from gnn_pruning.utils import ProgressReporter

from .base import PruningContext
from .registry import get_pruner


@dataclass
class PruningArtifacts:
    """Artifacts produced by prune workflow."""

    pruned_checkpoint_path: Path
    pruning_metrics_path: Path
    post_prune_metrics_path: Path | None = None
    diagnostics_path: Path | None = None


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
    diagnostics_path: Path | None = None


def prune_from_checkpoint(
    checkpoint_path: str,
    config_path: str,
    progress_reporter: ProgressReporter | None = None,
) -> PruningArtifacts:
    """Load dense checkpoint, apply selected pruner, and save pruned artifacts."""
    resolved = resolve_config(config_path)
    reporter = progress_reporter or ProgressReporter(enabled=False)
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
    dense_param_count = _parameter_count(model)
    dense_hidden_dims = _hidden_dims(model)

    pruner_cls = get_pruner(method)
    pruner = pruner_cls()

    output_dir = Path(resolved.run.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        resolved.data.name,
        resolved.data.root,
        dblp_strategy=getattr(resolved.data, "dblp_strategy", "author_homogeneous"),
    )
    data = dataset[0]
    split = _load_or_create_split(config_path=config_path, output_dir=output_dir, data=data)
    indices = to_index_tensors(split, device=resolved.device.device)

    context = PruningContext(
        config=resolved.to_dict(),
        data={"data": data, "train_idx": indices["train"]},
        device=resolved.device.device,
        seed=resolved.run.seed,
    )
    reporter.info(f"Scoring channels for {method} pruning...")
    scores = pruner.score(model, context)
    reporter.info("Applying pruning...")
    pruned_model, plan = pruner.apply(
        model=model,
        scores=scores,
        context=context,
        target_sparsity=target_sparsity,
        structured=structured,
    )

    pruned_param_count = _parameter_count(pruned_model)
    pruned_hidden_dims = _hidden_dims(pruned_model)

    if structured:
        if pruned_param_count >= dense_param_count:
            raise RuntimeError(
                "Structured pruning must reduce parameter count, "
                f"but got before={dense_param_count}, after={pruned_param_count}."
            )
        if not any(after < before for before, after in zip(dense_hidden_dims, pruned_hidden_dims)):
            raise RuntimeError(
                "Structured pruning must reduce at least one hidden dimension, "
                f"but got before={dense_hidden_dims}, after={pruned_hidden_dims}."
            )

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

    reporter.info("Evaluating dense checkpoint metrics...")
    dense_metrics = evaluate_dense(
        config_path=config_path,
        model_override=model,
        split_override=split,
        output_dir_override=str(output_dir),
    )
    reporter.info("Evaluating post-prune model...")
    post_prune_metrics = evaluate_dense(
        config_path=config_path,
        model_override=pruned_model,
        split_override=split,
        output_dir_override=str(output_dir),
    )
    post_prune_metrics_path = output_dir / f"metrics_post_prune_{method}.json"
    with post_prune_metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(post_prune_metrics, handle, indent=2)
    reporter.phase_metrics("post_prune", post_prune_metrics)

    eval_param_count = int(post_prune_metrics.get("benchmark", {}).get("parameter_count", -1))
    if structured and eval_param_count != pruned_param_count:
        raise RuntimeError(
            "Structured pruned evaluation must run with the compressed model, "
            f"but benchmark parameter_count={eval_param_count} and pruned_model parameter_count={pruned_param_count}."
        )

    debug_payload = {
        "stage": "post_prune",
        "structured": structured,
        "method": method,
        "model_class_before": model.__class__.__name__,
        "model_class_after": pruned_model.__class__.__name__,
        "hidden_dims_before": dense_hidden_dims,
        "hidden_dims_after": pruned_hidden_dims,
        "parameter_count_before": dense_param_count,
        "parameter_count_after": pruned_param_count,
        "evaluated_parameter_count": eval_param_count,
        "achieved_sparsity": plan.achieved_sparsity,
        "kept_channel_indices": list(plan.details.get("kept_channel_indices", [])),
        "saved_checkpoint_path": str(pruned_checkpoint_path),
        "loaded_checkpoint_path": str(Path(checkpoint_path).expanduser()),
    }
    _write_debug_artifact(output_dir, "prune", debug_payload)
    diagnostics_path = _write_pruning_diagnostics(
        output_dir=output_dir,
        payload={
            "dataset": resolved.data.name,
            "model": model_name,
            "method": method,
            "mode": "structured" if structured else "unstructured",
            "requested_sparsity": target_sparsity,
            "achieved_sparsity": plan.achieved_sparsity,
            "dense_hidden_dimensions": dense_hidden_dims,
            "pruned_hidden_dimensions": pruned_hidden_dims,
            "dense_parameter_count": dense_param_count,
            "pruned_parameter_count": pruned_param_count,
            "dense_checkpoint_size": _file_size_bytes(Path(checkpoint_path).expanduser()),
            "pruned_checkpoint_size": _file_size_bytes(pruned_checkpoint_path),
            "kept_channel_indices_per_layer": {str(plan.details.get("layer_index", 0)): list(plan.details.get("kept_channel_indices", []))},
            "dropped_channel_indices_per_layer": {
                str(plan.details.get("layer_index", 0)): _dropped_indices(
                    dense_hidden_dims[0] if dense_hidden_dims else 0,
                    list(plan.details.get("kept_channel_indices", [])),
                )
            },
            "metrics": {
                "dense": dense_metrics,
                "post_prune": post_prune_metrics,
            },
        },
    )

    return PruningArtifacts(
        pruned_checkpoint_path=pruned_checkpoint_path,
        pruning_metrics_path=pruning_metrics_path,
        post_prune_metrics_path=post_prune_metrics_path,
        diagnostics_path=diagnostics_path,
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
    progress_reporter: ProgressReporter | None = None,
) -> FineTuneArtifacts:
    """Fine-tune a pruned checkpoint, saving pre/post checkpoints and metrics."""
    resolved = resolve_config(config_path)
    reporter = progress_reporter or ProgressReporter(enabled=False)
    raw_cfg = load_yaml(config_path)
    prune_cfg: Dict[str, Any] = raw_cfg.get("pruning", {}) if isinstance(raw_cfg.get("pruning", {}), dict) else {}
    epochs = int(finetune_epochs if finetune_epochs is not None else prune_cfg.get("finetune_epochs", 50))

    checkpoint = _load_pruned_checkpoint_payload(Path(checkpoint_path).expanduser(), resolved.device.device)
    output_dir = Path(checkpoint_path).expanduser().parent
    output_dir.mkdir(parents=True, exist_ok=True)
    structured = str(checkpoint.get("pruning_plan", {}).get("mode", "")) == "structured"

    dataset = load_dataset(
        resolved.data.name,
        resolved.data.root,
        dblp_strategy=getattr(resolved.data, "dblp_strategy", "author_homogeneous"),
    )
    data = dataset[0]

    split = _load_or_create_split(config_path=config_path, output_dir=output_dir, data=data)
    indices = to_index_tensors(split, device=resolved.device.device)

    model = build_model(checkpoint["model_name"], **checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    pre_finetune_param_count = _parameter_count(model)
    pre_finetune_hidden_dims = _hidden_dims(model)

    source_checkpoint = checkpoint.get("source_checkpoint")
    source_param_count = None
    source_hidden_dims = None
    if source_checkpoint:
        source_payload = torch.load(Path(source_checkpoint).expanduser(), map_location=resolved.device.device)
        source_model = build_model(source_payload["model_name"], **source_payload["model_config"])
        source_model.load_state_dict(source_payload["model_state_dict"])
        source_param_count = _parameter_count(source_model)
        source_hidden_dims = _hidden_dims(source_model)
        if structured and pre_finetune_param_count >= source_param_count:
            raise RuntimeError(
                "Finetune must continue from compressed structured model, "
                f"but source parameter_count={source_param_count}, finetune start parameter_count={pre_finetune_param_count}."
            )

    pre_finetune_checkpoint_path = output_dir / f"{Path(checkpoint_path).stem}_pre_finetune.pt"
    torch.save(checkpoint, pre_finetune_checkpoint_path)

    reporter.info("Evaluating pre-finetune model...")
    pre_metrics = evaluate_dense(
        config_path=config_path,
        model_override=model,
        split_override=split,
        output_dir_override=str(output_dir),
    )
    pre_finetune_metrics_path = output_dir / "metrics_pruned_pre_finetune.json"
    with pre_finetune_metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(pre_metrics, handle, indent=2)
    reporter.phase_metrics("pre_finetune", pre_metrics)

    trainer = DenseTrainer(
        model=model,
        device=resolved.device.device,
        lr=resolved.training.lr,
        weight_decay=resolved.training.weight_decay,
        max_epochs=epochs,
        early_stopping_patience=resolved.training.early_stopping_patience,
    )
    reporter.info(f"Starting fine-tuning for {epochs} epochs...")
    train_result = trainer.fit(
        data=data,
        train_idx=indices["train"],
        val_idx=indices["val"],
        progress_callback=lambda payload: reporter.epoch(
            epoch=int(payload["epoch"]),
            max_epochs=int(payload["max_epochs"]),
            train_loss=float(payload["train_loss"]),
            val_loss=float(payload["val_loss"]),
            train_acc=float(payload["train_acc"]),
            val_acc=float(payload["val_acc"]),
            best_epoch=int(payload["best_epoch"]),
            early_stopping_counter=int(payload["early_stopping_counter"]),
            elapsed_sec=float(payload["elapsed_sec"]),
        ),
    )

    reporter.info("Evaluating post-finetune model...")
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
    reporter.phase_metrics("post_finetune", post_metrics)

    post_finetune_checkpoint_path = output_dir / f"{Path(checkpoint_path).stem}_post_finetune.pt"
    post_finetune_param_count = _parameter_count(trainer.model)
    post_finetune_hidden_dims = _hidden_dims(trainer.model)
    if structured and source_param_count is not None and post_finetune_param_count >= source_param_count:
        raise RuntimeError(
            "Post-finetune model must remain structurally compressed, "
            f"but source parameter_count={source_param_count}, post-finetune parameter_count={post_finetune_param_count}."
        )
    post_eval_param_count = int(post_metrics.get("benchmark", {}).get("parameter_count", -1))
    if structured and post_eval_param_count != post_finetune_param_count:
        raise RuntimeError(
            "Post-finetune evaluation must run with compressed model, "
            f"but benchmark parameter_count={post_eval_param_count}, model parameter_count={post_finetune_param_count}."
        )

    torch.save(
        {
            "model_name": checkpoint["model_name"],
            "model_config": _model_config_from_model(trainer.model, fallback=checkpoint["model_config"]),
            "source_checkpoint": str(Path(checkpoint_path).expanduser()),
            "model_state_dict": trainer.model.state_dict(),
            "finetune_epochs": epochs,
            "finetune_training": train_result.to_dict(),
            "pruning_plan": checkpoint.get("pruning_plan", {}),
        },
        post_finetune_checkpoint_path,
    )

    debug_payload = {
        "stage": "post_finetune",
        "structured": structured,
        "model_class_before": model.__class__.__name__,
        "model_class_after": trainer.model.__class__.__name__,
        "hidden_dims_source": source_hidden_dims,
        "hidden_dims_before_finetune": pre_finetune_hidden_dims,
        "hidden_dims_after_finetune": post_finetune_hidden_dims,
        "parameter_count_source": source_param_count,
        "parameter_count_before_finetune": pre_finetune_param_count,
        "parameter_count_after_finetune": post_finetune_param_count,
        "evaluated_parameter_count_post_finetune": post_eval_param_count,
        "loaded_checkpoint_path": str(Path(checkpoint_path).expanduser()),
        "saved_checkpoint_path": str(post_finetune_checkpoint_path),
    }
    _write_debug_artifact(output_dir, "finetune", debug_payload)
    diagnostics_path = _write_pruning_diagnostics(
        output_dir=output_dir,
        payload={
            "metrics": {
                "post_finetune": post_metrics,
            },
            "finetune": {
                "loaded_checkpoint_path": str(Path(checkpoint_path).expanduser()),
                "parameter_count_before_finetune": pre_finetune_param_count,
                "parameter_count_after_finetune": post_finetune_param_count,
            },
        },
        merge=True,
    )

    return FineTuneArtifacts(
        pre_finetune_checkpoint_path=pre_finetune_checkpoint_path,
        post_finetune_checkpoint_path=post_finetune_checkpoint_path,
        pre_finetune_metrics_path=pre_finetune_metrics_path,
        post_finetune_metrics_path=post_finetune_metrics_path,
        diagnostics_path=diagnostics_path,
    )


def _load_or_create_split(config_path: str, output_dir: Path, data: Any | None = None) -> Any:
    resolved = resolve_config(config_path)
    graph = data
    if graph is None:
        dataset = load_dataset(
            resolved.data.name,
            resolved.data.root,
            dblp_strategy=getattr(resolved.data, "dblp_strategy", "author_homogeneous"),
        )
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
        conv_last = model.convs[-1]
        if hasattr(conv0, "lin") and hasattr(conv_last, "lin"):
            return {
                "in_channels": int(conv0.lin.weight.shape[1]),
                "hidden_channels": int(conv0.lin.weight.shape[0]),
                "out_channels": int(conv_last.lin.weight.shape[0]),
                "num_layers": int(len(model.convs)),
                "dropout": float(getattr(model, "dropout", fallback.get("dropout", 0.0))),
            }
        if hasattr(conv0, "lin_l") and hasattr(conv_last, "lin_l"):
            return {
                "in_channels": int(conv0.lin_l.weight.shape[1]),
                "hidden_channels": int(conv0.lin_l.weight.shape[0]),
                "out_channels": int(conv_last.lin_l.weight.shape[0]),
                "num_layers": int(len(model.convs)),
                "dropout": float(getattr(model, "dropout", fallback.get("dropout", 0.0))),
            }
    return dict(fallback)


def _parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _hidden_dims(model: torch.nn.Module) -> list[int]:
    dims: list[int] = []
    if hasattr(model, "convs"):
        for conv in model.convs[:-1]:
            if hasattr(conv, "out_channels"):
                dims.append(int(conv.out_channels))
    return dims


def _write_debug_artifact(output_dir: Path, key: str, payload: Dict[str, Any]) -> None:
    path = output_dir / "pruning_debug.json"
    existing: Dict[str, Any] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
    existing[key] = payload
    with path.open("w", encoding="utf-8") as handle:
        json.dump(existing, handle, indent=2)


def _write_pruning_diagnostics(output_dir: Path, payload: Dict[str, Any], merge: bool = False) -> Path:
    path = output_dir / "pruning_diagnostics.json"
    existing: Dict[str, Any] = {}
    if merge and path.exists():
        with path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
    merged = _deep_merge(existing, payload)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(merged, handle, indent=2)
    return path


def _deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _file_size_bytes(path: Path) -> int:
    return int(path.stat().st_size) if path.exists() else 0


def _dropped_indices(total_channels: int, kept: list[int]) -> list[int]:
    kept_set = set(int(i) for i in kept)
    return [idx for idx in range(total_channels) if idx not in kept_set]

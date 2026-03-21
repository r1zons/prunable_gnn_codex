"""High-level dense training/evaluation workflows."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from gnn_pruning.config import dump_yaml, resolve_config
from gnn_pruning.config.schema import snapshot_path
from gnn_pruning.data import (
    generate_exact_ratio_split,
    load_dataset,
    load_split_indices,
    save_split_indices,
    to_index_tensors,
)
from gnn_pruning.evaluation import classification_metrics
from gnn_pruning.models import build_model
from gnn_pruning.training.checkpoints import load_checkpoint, save_checkpoint
from gnn_pruning.training.trainer import DenseTrainer
from gnn_pruning.utils import resolve_output_dir, set_seed


@dataclass
class TrainArtifacts:
    """Artifacts produced by dense training run."""

    resolved_config_path: Path
    split_path: Path
    checkpoint_path: Path
    metrics_path: Path


@dataclass
class EvalArtifacts:
    """Artifacts produced by dense evaluation run."""

    resolved_config_path: Path
    metrics_path: Path


def train_dense(config_path: str, resume: bool = True, output_dir_override: Optional[str] = None) -> TrainArtifacts:
    """Train dense model and persist checkpoint/metrics artifacts."""
    resolved = resolve_config(config_path)
    output_dir = _resolve_run_dir(resolved, resume=resume, output_dir_override=output_dir_override)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved.run.output_dir = str(output_dir)

    set_seed(resolved.run.seed)
    resolved_path = snapshot_path(output_dir)
    dump_yaml(resolved.to_dict(), resolved_path)

    dataset = load_dataset(resolved.data.name, resolved.data.root)
    data = dataset[0]

    split_path = output_dir / "splits.yaml"
    if split_path.exists():
        split = load_split_indices(split_path)
    else:
        split = generate_exact_ratio_split(
            num_nodes=int(data.num_nodes),
            seed=resolved.run.seed,
            train_ratio=resolved.data.train_ratio,
            val_ratio=resolved.data.val_ratio,
            test_ratio=resolved.data.test_ratio,
        )
        split_path = save_split_indices(split, output_dir)

    device = resolved.device.device
    indices = to_index_tensors(split, device=device)
    model = _build_model_from_data(resolved, data)

    checkpoint_path = output_dir / "dense_checkpoint.pt"
    resume_state: Optional[Dict[str, Any]] = None
    if resume and checkpoint_path.exists():
        resume_state = load_checkpoint(checkpoint_path, map_location=device)

    trainer = DenseTrainer(
        model=model,
        device=device,
        lr=resolved.training.lr,
        weight_decay=resolved.training.weight_decay,
        max_epochs=resolved.training.epochs,
        early_stopping_patience=resolved.training.early_stopping_patience,
    )
    train_result = trainer.fit(data=data, train_idx=indices["train"], val_idx=indices["val"], resume_state=resume_state)

    # evaluate on splits using best model state restored by trainer
    metrics = evaluate_dense(
        config_path,
        checkpoint_path=None,
        model_override=trainer.model,
        split_override=split,
        output_dir_override=str(output_dir),
    )
    metrics["training"] = train_result.to_dict()

    checkpoint_payload = {
        "epoch": int(train_result.epochs_ran - 1),
        "best_epoch": int(train_result.best_epoch),
        "best_val_loss": float(train_result.best_val_loss),
        "model_state_dict": trainer.model.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "model_name": resolved.model.name,
        "model_config": {
            "in_channels": _resolve_in_channels(resolved, data),
            "hidden_channels": resolved.model.hidden_channels,
            "out_channels": _resolve_out_channels(resolved, data),
            "num_layers": resolved.model.num_layers,
            "dropout": resolved.model.dropout,
        },
    }
    save_checkpoint(checkpoint_payload, checkpoint_path)

    metrics_path = output_dir / "metrics_train.json"
    _write_json(metrics, metrics_path)

    return TrainArtifacts(
        resolved_config_path=resolved_path,
        split_path=split_path,
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
    )


def evaluate_dense(
    config_path: str,
    checkpoint_path: Optional[Path] = None,
    model_override: Optional[torch.nn.Module] = None,
    split_override: Any = None,
    output_dir_override: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """Evaluate dense model and return split metrics."""
    resolved = resolve_config(config_path)
    output_dir = _resolve_run_dir(
        resolved,
        resume=True,
        output_dir_override=output_dir_override or (str(checkpoint_path.parent) if checkpoint_path else None),
    )
    resolved.run.output_dir = str(output_dir)

    set_seed(resolved.run.seed)

    dataset = load_dataset(resolved.data.name, resolved.data.root)
    data = dataset[0]
    device = torch.device(resolved.device.device)

    split = split_override
    if split is None:
        split_file = output_dir / "splits.yaml"
        if split_file.exists():
            split = load_split_indices(split_file)
        else:
            split = generate_exact_ratio_split(
                num_nodes=int(data.num_nodes),
                seed=resolved.run.seed,
                train_ratio=resolved.data.train_ratio,
                val_ratio=resolved.data.val_ratio,
                test_ratio=resolved.data.test_ratio,
            )

    index_tensors = to_index_tensors(split, device=device)

    if model_override is not None:
        model = model_override.to(device)
    else:
        model = _build_model_from_data(resolved, data).to(device)
        if checkpoint_path is None:
            checkpoint_path = output_dir / "dense_checkpoint.pt"
        state = load_checkpoint(checkpoint_path, map_location=device)
        model.load_state_dict(state["model_state_dict"])

    model.eval()
    data = data.to(device)
    with torch.no_grad():
        logits = model(data)
        pred = logits.argmax(dim=-1)

    y_true = data.y.detach().cpu().numpy()
    y_pred = pred.detach().cpu().numpy()

    result: Dict[str, Dict[str, float]] = {}
    for split_name, idx_tensor in index_tensors.items():
        idx = idx_tensor.detach().cpu().numpy()
        result[split_name] = classification_metrics(y_true[idx], y_pred[idx])

    return result


def evaluate_dense_and_save(
    config_path: str,
    checkpoint_path: Optional[str] = None,
    output_dir_override: Optional[str] = None,
) -> EvalArtifacts:
    """Evaluate dense checkpoint and save metrics/resolved config artifacts."""
    resolved = resolve_config(config_path)
    if checkpoint_path:
        output_dir = Path(checkpoint_path).expanduser().parent
    else:
        output_dir = _resolve_run_dir(resolved, resume=True, output_dir_override=output_dir_override)

    output_dir.mkdir(parents=True, exist_ok=True)
    resolved.run.output_dir = str(output_dir)

    resolved_path = snapshot_path(output_dir)
    dump_yaml(resolved.to_dict(), resolved_path)

    ckpt = Path(checkpoint_path).expanduser() if checkpoint_path else None
    metrics = evaluate_dense(config_path=config_path, checkpoint_path=ckpt, output_dir_override=str(output_dir))
    metrics_path = output_dir / "metrics_eval.json"
    _write_json(metrics, metrics_path)
    return EvalArtifacts(resolved_config_path=resolved_path, metrics_path=metrics_path)


def _build_model_from_data(resolved: Any, data: Any) -> torch.nn.Module:
    return build_model(
        resolved.model.name,
        in_channels=_resolve_in_channels(resolved, data),
        hidden_channels=resolved.model.hidden_channels,
        out_channels=_resolve_out_channels(resolved, data),
        num_layers=resolved.model.num_layers,
        dropout=resolved.model.dropout,
    )


def _resolve_in_channels(resolved: Any, data: Any) -> int:
    if resolved.model.in_channels is not None:
        return int(resolved.model.in_channels)
    return int(data.num_features)


def _resolve_out_channels(resolved: Any, data: Any) -> int:
    if resolved.model.out_channels is not None:
        return int(resolved.model.out_channels)
    return int(data.y.max().item()) + 1


def _write_json(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _resolve_run_dir(resolved: Any, resume: bool, output_dir_override: Optional[str]) -> Path:
    if output_dir_override:
        return Path(output_dir_override).expanduser()
    return resolve_output_dir(
        configured_output_dir=resolved.run.output_dir,
        experiment_name=resolved.run.experiment_name,
        dataset_name=resolved.data.name,
        model_name=resolved.model.name,
        seed=resolved.run.seed,
        resume=resume,
    )

"""High-level dense training/evaluation workflows."""

from __future__ import annotations

import hashlib
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
from gnn_pruning.evaluation import (
    classification_metrics,
    measure_inference_time,
    model_size_metrics,
    runtime_memory_metrics,
)
from gnn_pruning.models import build_model
from gnn_pruning.training.checkpoints import load_checkpoint, save_checkpoint
from gnn_pruning.training.trainer import DenseTrainer
from gnn_pruning.utils import ProgressReporter, resolve_output_dir, set_seed


@dataclass
class TrainArtifacts:
    """Artifacts produced by dense training run."""

    resolved_config_path: Path
    split_path: Path
    checkpoint_path: Path
    metrics_path: Path
    checkpoint_reused: bool = False
    split_hash: str = ""
    config_hash: str = ""


@dataclass
class EvalArtifacts:
    """Artifacts produced by dense evaluation run."""

    resolved_config_path: Path
    metrics_path: Path


def train_dense(
    config_path: str,
    resume: bool = True,
    output_dir_override: Optional[str] = None,
    progress_reporter: Optional[ProgressReporter] = None,
) -> TrainArtifacts:
    """Train dense model and persist checkpoint/metrics artifacts."""
    resolved = resolve_config(config_path)
    output_dir = _resolve_run_dir(resolved, resume=resume, output_dir_override=output_dir_override)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved.run.output_dir = str(output_dir)
    reporter = progress_reporter or ProgressReporter(enabled=False)

    set_seed(resolved.run.seed)
    resolved_path = snapshot_path(output_dir)
    dump_yaml(resolved.to_dict(), resolved_path)

    reporter.stage(1, 6, f"Loading dataset ({resolved.data.name})")
    dataset = _load_dataset_for_config(resolved)
    data = dataset[0]

    split_path = output_dir / "splits.yaml"
    reporter.stage(2, 6, "Creating/loading split")
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
    split_hash = _split_hash(split)

    device = resolved.device.device
    indices = to_index_tensors(split, device=device)
    reporter.stage(3, 6, f"Building model ({resolved.model.name})")
    model = _build_model_from_data(resolved, data)
    reporter.info(
        f"Architecture: model={resolved.model.name} num_layers={resolved.model.num_layers} "
        f"hidden_channels={resolved.model.hidden_channels} params={_parameter_count(model)}"
    )

    checkpoint_path = output_dir / "dense_checkpoint.pt"
    resume_state: Optional[Dict[str, Any]] = None
    checkpoint_reused = False
    run_signature = _build_run_signature(resolved=resolved, split_hash=split_hash)
    if resume and checkpoint_path.exists():
        loaded = load_checkpoint(checkpoint_path, map_location=device)
        compatible, reason = _checkpoint_is_compatible(loaded, run_signature)
        if compatible:
            resume_state = loaded
            checkpoint_reused = True
            reporter.info(f"Using dense checkpoint: {checkpoint_path}")
            reporter.info("Checkpoint compatibility validated")
        else:
            reporter.info(f"Checkpoint rejected due to config mismatch: {reason}")

    trainer = DenseTrainer(
        model=model,
        device=device,
        lr=resolved.training.lr,
        weight_decay=resolved.training.weight_decay,
        max_epochs=resolved.training.epochs,
        early_stopping_patience=resolved.training.early_stopping_patience,
    )
    reporter.stage(4, 6, "Training dense model")
    train_result = trainer.fit(
        data=data,
        train_idx=indices["train"],
        val_idx=indices["val"],
        resume_state=resume_state,
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

    # evaluate on splits using best model state restored by trainer
    reporter.stage(5, 6, "Evaluating dense model")
    metrics = evaluate_dense(
        config_path,
        checkpoint_path=None,
        model_override=trainer.model,
        split_override=split,
        output_dir_override=str(output_dir),
    )
    metrics["training"] = train_result.to_dict()
    reporter.phase_metrics("dense", metrics)

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
        "run_signature": run_signature,
    }
    save_checkpoint(checkpoint_payload, checkpoint_path)
    reporter.stage(6, 6, "Saving artifacts")

    metrics_path = output_dir / "metrics_train.json"
    _write_json(metrics, metrics_path)

    return TrainArtifacts(
        resolved_config_path=resolved_path,
        split_path=split_path,
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
        checkpoint_reused=checkpoint_reused,
        split_hash=split_hash,
        config_hash=str(run_signature["config_hash"]),
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

    dataset = _load_dataset_for_config(resolved)
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

    benchmark_metrics: Dict[str, float] = {}
    benchmark_metrics.update(
        measure_inference_time(
            model=model,
            data=data,
            device=device,
            warmup_passes=resolved.benchmark.inference_warmup_passes,
            timed_passes=resolved.benchmark.inference_timed_passes,
        )
    )
    benchmark_metrics.update(
        model_size_metrics(
            model=model,
            checkpoint_path=checkpoint_path,
        )
    )
    benchmark_metrics.update(
        runtime_memory_metrics(
            model=model,
            data=data,
            device=device,
        )
    )
    result["benchmark"] = benchmark_metrics

    return result


def evaluate_dense_and_save(
    config_path: str,
    checkpoint_path: Optional[str] = None,
    output_dir_override: Optional[str] = None,
    progress_reporter: Optional[ProgressReporter] = None,
) -> EvalArtifacts:
    """Evaluate dense checkpoint and save metrics/resolved config artifacts."""
    resolved = resolve_config(config_path)
    if checkpoint_path:
        output_dir = Path(checkpoint_path).expanduser().parent
    else:
        output_dir = _resolve_run_dir(resolved, resume=True, output_dir_override=output_dir_override)

    output_dir.mkdir(parents=True, exist_ok=True)
    resolved.run.output_dir = str(output_dir)
    reporter = progress_reporter or ProgressReporter(enabled=False)

    resolved_path = snapshot_path(output_dir)
    dump_yaml(resolved.to_dict(), resolved_path)

    ckpt = Path(checkpoint_path).expanduser() if checkpoint_path else None
    reporter.info("Evaluating dense checkpoint...")
    metrics = evaluate_dense(config_path=config_path, checkpoint_path=ckpt, output_dir_override=str(output_dir))
    reporter.phase_metrics("dense_eval", metrics)
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


def _split_hash(split: Any) -> str:
    payload = {
        "train": [int(v) for v in split.train],
        "val": [int(v) for v in split.val],
        "test": [int(v) for v in split.test],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _build_run_signature(resolved: Any, split_hash: str) -> Dict[str, Any]:
    return {
        "dataset": resolved.data.name,
        "model_name": resolved.model.name,
        "num_layers": int(resolved.model.num_layers),
        "hidden_channels": int(resolved.model.hidden_channels),
        "dropout": float(resolved.model.dropout),
        "seed": int(resolved.run.seed),
        "split_hash": split_hash,
        "config_hash": hashlib.sha256(json.dumps(resolved.to_dict(), sort_keys=True).encode("utf-8")).hexdigest(),
    }


def _checkpoint_is_compatible(payload: Dict[str, Any], expected: Dict[str, Any]) -> tuple[bool, str]:
    saved = payload.get("run_signature")
    if not isinstance(saved, dict):
        return False, "missing run_signature"
    for key in ["dataset", "model_name", "num_layers", "hidden_channels", "dropout", "seed", "split_hash", "config_hash"]:
        if saved.get(key) != expected.get(key):
            return False, f"{key} differs (saved={saved.get(key)} expected={expected.get(key)})"
    return True, ""


def _parameter_count(model: torch.nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


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


def _load_dataset_for_config(resolved: Any) -> Any:
    """Load dataset with backward-compatible call signature for monkeypatched tests."""
    if str(resolved.data.name).strip().lower() == "dblp":
        return load_dataset(
            resolved.data.name,
            resolved.data.root,
            getattr(resolved.data, "dblp_strategy", "author_homogeneous"),
        )
    return load_dataset(resolved.data.name, resolved.data.root)

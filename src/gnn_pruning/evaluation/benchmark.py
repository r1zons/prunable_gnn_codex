"""Benchmarking utilities for dense evaluation."""

from __future__ import annotations

import os
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Optional

import torch


def measure_inference_time(
    model: torch.nn.Module,
    data: Any,
    device: torch.device,
    warmup_passes: int,
    timed_passes: int,
) -> Dict[str, float]:
    """Measure inference latency with warmup and timed passes."""
    model.eval()
    data = data.to(device)

    for _ in range(warmup_passes):
        _run_forward(model, data, device)

    samples = []
    for _ in range(timed_passes):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        _run_forward(model, data, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        end = time.perf_counter()
        samples.append(end - start)

    return {
        "inference_time_mean_sec": float(mean(samples)),
        "inference_time_std_sec": float(pstdev(samples) if len(samples) > 1 else 0.0),
        "inference_timed_passes": float(timed_passes),
        "inference_warmup_passes": float(warmup_passes),
    }


def model_size_metrics(model: torch.nn.Module, checkpoint_path: Optional[Path] = None) -> Dict[str, float]:
    """Compute model size metrics from parameters/checkpoint."""
    param_count = sum(parameter.numel() for parameter in model.parameters())
    param_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    checkpoint_size = 0
    if checkpoint_path is not None and checkpoint_path.exists():
        checkpoint_size = os.path.getsize(checkpoint_path)

    return {
        "parameter_count": float(param_count),
        "parameter_bytes": float(param_bytes),
        "checkpoint_size_bytes": float(checkpoint_size),
    }


def runtime_memory_metrics(model: torch.nn.Module, data: Any, device: torch.device) -> Dict[str, float]:
    """Estimate runtime memory use on CPU/CUDA."""
    model.eval()
    data = data.to(device)

    cpu_param_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    cpu_input_bytes = _estimate_tensor_bytes(data)
    output = _run_forward(model, data, device)
    cpu_output_bytes = _estimate_tensor_bytes(output)

    result = {
        "cpu_memory_estimate_bytes": float(cpu_param_bytes + cpu_input_bytes + cpu_output_bytes),
        "cuda_peak_memory_bytes": 0.0,
    }

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        _run_forward(model, data, device)
        torch.cuda.synchronize(device)
        result["cuda_peak_memory_bytes"] = float(torch.cuda.max_memory_allocated(device))

    return result


def _run_forward(model: torch.nn.Module, data: Any, device: torch.device) -> Any:
    with torch.no_grad():
        return model(data.to(device))


def _estimate_tensor_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.numel() * value.element_size())

    total = 0
    if hasattr(value, "to_dict"):
        value = value.to_dict()

    if isinstance(value, dict):
        for item in value.values():
            total += _estimate_tensor_bytes(item)
    elif hasattr(value, "__dict__"):
        for item in value.__dict__.values():
            total += _estimate_tensor_bytes(item)

    return total

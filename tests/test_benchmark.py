"""Tests for benchmarking utility functions."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch_geometric.data import Data

from gnn_pruning.evaluation.benchmark import (
    measure_inference_time,
    model_size_metrics,
)
from gnn_pruning.training.checkpoints import save_checkpoint


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, data: Data) -> torch.Tensor:
        return self.linear(data.x)


def _dummy_data() -> Data:
    x = torch.randn((8, 4), dtype=torch.float32)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    return Data(x=x, edge_index=edge_index)


def test_timing_utility_returns_sane_values() -> None:
    model = TinyModel()
    data = _dummy_data()
    metrics = measure_inference_time(
        model=model,
        data=data,
        device=torch.device("cpu"),
        warmup_passes=2,
        timed_passes=5,
    )

    assert metrics["inference_time_mean_sec"] > 0
    assert metrics["inference_time_std_sec"] >= 0
    assert metrics["inference_timed_passes"] == 5


def test_parameter_counting_works() -> None:
    model = TinyModel()
    metrics = model_size_metrics(model)

    expected_params = sum(parameter.numel() for parameter in model.parameters())
    expected_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())

    assert metrics["parameter_count"] == float(expected_params)
    assert metrics["parameter_bytes"] == float(expected_bytes)


def test_checkpoint_size_metric_works(tmp_path: Path) -> None:
    model = TinyModel()
    checkpoint_path = save_checkpoint({"model_state_dict": model.state_dict()}, tmp_path / "dense_checkpoint.pt")

    metrics = model_size_metrics(model, checkpoint_path=checkpoint_path)

    assert metrics["checkpoint_size_bytes"] > 0

"""GCN node-classification model."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Union

import torch
from torch import Tensor, nn
from torch_geometric.nn import GCNConv

from .base import BaseNodeClassifier


class GCNNodeClassifier(BaseNodeClassifier):
    """Configurable multi-layer GCN for node classification."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: Union[int, Sequence[int]],
        out_channels: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__(dropout=dropout)
        if num_layers < 2:
            raise ValueError("GCN requires num_layers >= 2.")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.num_layers = int(num_layers)
        self.hidden_channel_dims = _normalize_hidden_channels(hidden_channels, self.num_layers)
        self.hidden_channels = int(self.hidden_channel_dims[0])

        dims: List[int] = [self.in_channels] + self.hidden_channel_dims + [self.out_channels]
        self.convs = nn.ModuleList([
            GCNConv(dims[i], dims[i + 1]) for i in range(self.num_layers)
        ])

    def forward(self, data: Any) -> Tensor:
        x, edge_index = data.x, data.edge_index
        for layer_idx, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            is_last = layer_idx == len(self.convs) - 1
            if not is_last:
                x = torch.relu(x)
                x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        return x

    def export_architecture_config(self) -> Dict[str, Any]:
        return {
            "name": "gcn",
            "in_channels": self.in_channels,
            "hidden_channels": _export_hidden_channels(self.hidden_channel_dims),
            "out_channels": self.out_channels,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
        }


def _normalize_hidden_channels(hidden_channels: Union[int, Sequence[int]], num_layers: int) -> List[int]:
    if isinstance(hidden_channels, int):
        widths = [int(hidden_channels)] * (num_layers - 1)
    else:
        widths = [int(width) for width in hidden_channels]
    if len(widths) != num_layers - 1:
        raise ValueError(f"Expected {num_layers - 1} hidden widths, got {len(widths)}.")
    if any(width <= 0 for width in widths):
        raise ValueError("All hidden channel widths must be positive.")
    return widths


def _export_hidden_channels(widths: Sequence[int]) -> Union[int, List[int]]:
    return int(widths[0]) if len(set(widths)) == 1 else [int(width) for width in widths]

"""GraphSAGE node-classification model."""

from __future__ import annotations

from typing import Any, Dict, List

import torch
from torch import Tensor, nn
from torch_geometric.nn import SAGEConv

from .base import BaseNodeClassifier


class GraphSAGENodeClassifier(BaseNodeClassifier):
    """Configurable multi-layer GraphSAGE for node classification."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__(dropout=dropout)
        if num_layers < 2:
            raise ValueError("GraphSAGE requires num_layers >= 2.")

        self.in_channels = int(in_channels)
        self.hidden_channels = int(hidden_channels)
        self.out_channels = int(out_channels)
        self.num_layers = int(num_layers)

        dims: List[int] = [self.in_channels] + [self.hidden_channels] * (self.num_layers - 1) + [self.out_channels]
        self.convs = nn.ModuleList([
            SAGEConv(dims[i], dims[i + 1]) for i in range(self.num_layers)
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
            "name": "graphsage",
            "in_channels": self.in_channels,
            "hidden_channels": self.hidden_channels,
            "out_channels": self.out_channels,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
        }

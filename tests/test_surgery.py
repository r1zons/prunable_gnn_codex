"""Tests for structural compression surgery utilities."""

from __future__ import annotations

import torch
from torch_geometric.data import Data

from gnn_pruning.models import GCNNodeClassifier, GraphSAGENodeClassifier
from gnn_pruning.surgery import structurally_prune_hidden_channels, validate_structural_compression


def _tiny_graph(num_nodes: int = 10, in_channels: int = 6) -> Data:
    x = torch.randn((num_nodes, in_channels), dtype=torch.float32)
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
        ],
        dtype=torch.long,
    )
    y = torch.randint(0, 3, (num_nodes,), dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y)


def _parameter_count(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def test_structural_compression_reduces_parameters_gcn() -> None:
    model = GCNNodeClassifier(in_channels=6, hidden_channels=8, out_channels=3, num_layers=2, dropout=0.0)
    data = _tiny_graph()

    compressed = structurally_prune_hidden_channels(model, layer_index=0, keep_indices=[0, 1, 2, 3])
    validate_structural_compression(model, compressed, data)

    assert _parameter_count(compressed) < _parameter_count(model)
    assert compressed.convs[0].out_channels == 4


def test_structural_compression_reduces_parameters_graphsage() -> None:
    model = GraphSAGENodeClassifier(in_channels=6, hidden_channels=10, out_channels=3, num_layers=2, dropout=0.0)
    data = _tiny_graph()

    compressed = structurally_prune_hidden_channels(model, layer_index=0, keep_indices=[0, 1, 2, 3, 4])
    validate_structural_compression(model, compressed, data)

    assert _parameter_count(compressed) < _parameter_count(model)
    assert compressed.convs[0].out_channels == 5


def test_compressed_model_forward_still_works() -> None:
    model = GCNNodeClassifier(in_channels=6, hidden_channels=8, out_channels=3, num_layers=2, dropout=0.0)
    data = _tiny_graph()

    compressed = structurally_prune_hidden_channels(model, layer_index=0, keep_indices=[0, 2, 4, 6])
    out = compressed(data)

    assert out.shape == (data.num_nodes, 3)
    assert compressed.convs[0].out_channels < model.convs[0].out_channels

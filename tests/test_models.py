"""Tests for baseline node-classification models."""

from __future__ import annotations

import torch
from torch_geometric.data import Data

from gnn_pruning.models import MODEL_REGISTRY, GCNNodeClassifier, GraphSAGENodeClassifier, build_model


def _tiny_graph(num_nodes: int = 4, in_channels: int = 3) -> Data:
    x = torch.randn((num_nodes, in_channels), dtype=torch.float32)
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 0, 2],
            [1, 0, 3, 2, 2, 0],
        ],
        dtype=torch.long,
    )
    return Data(x=x, edge_index=edge_index)


def test_model_registry_keys() -> None:
    assert "gcn" in MODEL_REGISTRY
    assert "graphsage" in MODEL_REGISTRY


def test_gcn_construction_and_forward_shape() -> None:
    model = GCNNodeClassifier(
        in_channels=3,
        hidden_channels=8,
        out_channels=2,
        num_layers=2,
        dropout=0.5,
    )
    data = _tiny_graph(num_nodes=5, in_channels=3)

    logits = model(data)

    assert logits.shape == (5, 2)
    assert model.export_architecture_config()["name"] == "gcn"


def test_graphsage_construction_and_forward_shape() -> None:
    model = GraphSAGENodeClassifier(
        in_channels=3,
        hidden_channels=8,
        out_channels=4,
        num_layers=3,
        dropout=0.2,
    )
    data = _tiny_graph(num_nodes=6, in_channels=3)

    logits = model(data)

    assert logits.shape == (6, 4)
    assert model.export_architecture_config()["name"] == "graphsage"


def test_predict_output_shape() -> None:
    model = build_model(
        "gcn",
        in_channels=3,
        hidden_channels=8,
        out_channels=3,
        num_layers=2,
        dropout=0.0,
    )
    data = _tiny_graph(num_nodes=7, in_channels=3)

    pred = model.predict(data)

    assert pred.shape == (7,)
    assert pred.dtype == torch.long

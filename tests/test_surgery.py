"""Tests for structural compression surgery utilities."""

from __future__ import annotations

import torch
from torch_geometric.data import Data

from gnn_pruning.models import GCNNodeClassifier, GraphSAGENodeClassifier
from gnn_pruning.surgery import (
    can_apply_structural_prune,
    structurally_prune_hidden_channels,
    structurally_prune_hidden_channels_local,
    validate_structural_compression,
)


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


def _hidden_widths(model) -> list[int]:
    return [int(conv.out_channels) for conv in model.convs[:-1]]


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


def test_cascade_behavior_is_preserved_for_multilayer_graphsage() -> None:
    model = GraphSAGENodeClassifier(in_channels=6, hidden_channels=8, out_channels=3, num_layers=3, dropout=0.0)
    compressed = structurally_prune_hidden_channels(model, layer_index=0, keep_indices=[0, 1, 2, 3])
    assert _hidden_widths(compressed) == [4, 4]


def test_local_nonmonotonic_graphsage_prune_is_safe() -> None:
    model = GraphSAGENodeClassifier(in_channels=6, hidden_channels=8, out_channels=3, num_layers=3, dropout=0.0)
    data = _tiny_graph()
    stage1 = structurally_prune_hidden_channels_local(model, layer_index=1, keep_indices=[0, 1, 2, 3])
    assert _hidden_widths(stage1) == [8, 4]

    stage2 = structurally_prune_hidden_channels_local(stage1, layer_index=0, keep_indices=[0, 1, 2, 3, 4, 5, 6])
    assert _hidden_widths(stage2) == [7, 4]
    out = stage2(data)
    assert out.shape == (data.num_nodes, 3)


def test_local_nonmonotonic_gcn_prune_is_safe() -> None:
    model = GCNNodeClassifier(in_channels=6, hidden_channels=8, out_channels=3, num_layers=3, dropout=0.0)
    data = _tiny_graph()
    stage1 = structurally_prune_hidden_channels_local(model, layer_index=1, keep_indices=[0, 1, 2, 3])
    assert _hidden_widths(stage1) == [8, 4]

    stage2 = structurally_prune_hidden_channels_local(stage1, layer_index=0, keep_indices=[0, 1, 2, 3, 4, 5, 6])
    assert _hidden_widths(stage2) == [7, 4]
    out = stage2(data)
    assert out.shape == (data.num_nodes, 3)


def test_feasibility_accepts_valid_local_nonmonotonic_case() -> None:
    model = GraphSAGENodeClassifier(in_channels=6, hidden_channels=8, out_channels=3, num_layers=3, dropout=0.0)
    stage1 = structurally_prune_hidden_channels_local(model, layer_index=1, keep_indices=[0, 1, 2, 3])
    result = can_apply_structural_prune(
        stage1,
        layer_index=0,
        keep_indices=[0, 1, 2, 3, 4, 5, 6],
        min_channels_per_layer=1,
        mode="local",
    )
    assert result.valid
    assert result.reason == "ok"
    assert result.hidden_widths_before == [8, 4]
    assert result.expected_hidden_widths_after == [7, 4]
    assert result.selected_layer_width == 8
    assert result.keep_count == 7


def test_feasibility_rejects_out_of_bounds_keep_indices() -> None:
    model = GraphSAGENodeClassifier(in_channels=6, hidden_channels=8, out_channels=3, num_layers=3, dropout=0.0)
    result = can_apply_structural_prune(
        model,
        layer_index=0,
        keep_indices=[0, 1, 8],
        min_channels_per_layer=1,
        mode="local",
    )
    assert not result.valid
    assert result.reason == "keep_indices_out_of_bounds"


def test_feasibility_rejects_min_channels_violation() -> None:
    model = GCNNodeClassifier(in_channels=6, hidden_channels=8, out_channels=3, num_layers=3, dropout=0.0)
    result = can_apply_structural_prune(
        model,
        layer_index=0,
        keep_indices=[0, 1, 2],
        min_channels_per_layer=4,
        mode="local",
    )
    assert not result.valid
    assert result.reason == "min_channels_violation"


def test_feasibility_reports_expected_widths_for_cascade_mode() -> None:
    model = GraphSAGENodeClassifier(in_channels=6, hidden_channels=8, out_channels=3, num_layers=3, dropout=0.0)
    result = can_apply_structural_prune(
        model,
        layer_index=0,
        keep_indices=[0, 1, 2, 3],
        min_channels_per_layer=1,
        mode="cascade",
    )
    assert result.valid
    assert result.expected_hidden_widths_after == [4, 4]

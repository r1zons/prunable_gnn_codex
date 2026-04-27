"""Validation tests for GraphSAGE config application during model build."""

from __future__ import annotations

from gnn_pruning.config import resolve_config
from gnn_pruning.models import build_model


def _parameter_count(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def test_flickr_graphsage_configs_change_parameter_count() -> None:
    config_paths = [
        "configs/experiments/flickr_graphsage_l2_h64.yaml",
        "configs/experiments/flickr_graphsage_l2_h128.yaml",
        "configs/experiments/flickr_graphsage_l3_h64.yaml",
        "configs/experiments/flickr_graphsage_l3_h128.yaml",
        "configs/experiments/flickr_graphsage_l4_h128.yaml",
    ]

    counts = []
    for config_path in config_paths:
        resolved = resolve_config(config_path)
        model = build_model(
            resolved.model.name,
            in_channels=16,
            hidden_channels=resolved.model.hidden_channels,
            out_channels=7,
            num_layers=resolved.model.num_layers,
            dropout=resolved.model.dropout,
        )
        counts.append(_parameter_count(model))

    assert len(set(counts)) == len(counts)


def test_graphsage_config_values_match_built_model() -> None:
    for config_path in [
        "configs/experiments/flickr_graphsage_l2_h64.yaml",
        "configs/experiments/flickr_graphsage_l3_h128.yaml",
        "configs/experiments/flickr_graphsage_l4_h128.yaml",
    ]:
        resolved = resolve_config(config_path)
        model = build_model(
            resolved.model.name,
            in_channels=16,
            hidden_channels=resolved.model.hidden_channels,
            out_channels=7,
            num_layers=resolved.model.num_layers,
            dropout=resolved.model.dropout,
        )

        assert model.num_layers == resolved.model.num_layers
        assert model.hidden_channels == resolved.model.hidden_channels
        assert model.dropout == resolved.model.dropout
        assert len(model.convs) == resolved.model.num_layers
        assert int(model.convs[0].lin_l.weight.shape[0]) == resolved.model.hidden_channels

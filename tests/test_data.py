"""Tests for dataset factory and split generation."""

from __future__ import annotations

from pathlib import Path

import pytest

from gnn_pruning.data import (
    generate_exact_ratio_split,
    get_dataset_loader,
    get_supported_datasets,
    save_split_indices,
)
from gnn_pruning.data.factory import load_dataset


def test_supported_dataset_names_are_configured() -> None:
    supported = set(get_supported_datasets())
    expected = {
        "cora",
        "citeseer",
        "pubmed",
        "texas",
        "cornell",
        "wisconsin",
        "actor",
        "amazon_computers",
        "flickr",
        "reddit",
        "dblp",
    }
    assert supported == expected

    for name in expected:
        assert callable(get_dataset_loader(name))


def test_dblp_loader_is_explicit_adapter_path() -> None:
    with pytest.raises(NotImplementedError, match="dedicated heterogeneous adapter"):
        load_dataset("dblp")


def test_split_correctness_disjoint_and_cover_all_nodes() -> None:
    split = generate_exact_ratio_split(num_nodes=101, seed=42)

    train_set = set(split.train)
    val_set = set(split.val)
    test_set = set(split.test)
    all_nodes = train_set | val_set | test_set

    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)
    assert len(all_nodes) == 101


def test_split_reproducibility_by_seed() -> None:
    split_a = generate_exact_ratio_split(num_nodes=57, seed=7)
    split_b = generate_exact_ratio_split(num_nodes=57, seed=7)
    split_c = generate_exact_ratio_split(num_nodes=57, seed=8)

    assert split_a == split_b
    assert split_a != split_c


def test_save_split_indices_artifact(tmp_path: Path) -> None:
    split = generate_exact_ratio_split(num_nodes=20, seed=1)
    artifact = save_split_indices(split, tmp_path)

    assert artifact.exists()
    content = artifact.read_text(encoding="utf-8")
    assert "train:" in content
    assert "val:" in content
    assert "test:" in content


def test_factory_routes_to_expected_pyg_classes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = []

    class _DummyDatasets:
        class Planetoid:
            def __init__(self, root: str, name: str) -> None:
                calls.append(("Planetoid", root, name))

        class WebKB:
            def __init__(self, root: str, name: str) -> None:
                calls.append(("WebKB", root, name))

        class Actor:
            def __init__(self, root: str) -> None:
                calls.append(("Actor", root))

        class Amazon:
            def __init__(self, root: str, name: str) -> None:
                calls.append(("Amazon", root, name))

        class Flickr:
            def __init__(self, root: str) -> None:
                calls.append(("Flickr", root))

        class Reddit:
            def __init__(self, root: str) -> None:
                calls.append(("Reddit", root))

    from gnn_pruning.data import factory

    monkeypatch.setattr(factory, "_require_pyg", lambda: _DummyDatasets)

    load_dataset("cora", tmp_path)
    load_dataset("texas", tmp_path)
    load_dataset("actor", tmp_path)
    load_dataset("amazon_computers", tmp_path)
    load_dataset("flickr", tmp_path)
    load_dataset("reddit", tmp_path)

    called_families = [entry[0] for entry in calls]
    assert called_families == ["Planetoid", "WebKB", "Actor", "Amazon", "Flickr", "Reddit"]

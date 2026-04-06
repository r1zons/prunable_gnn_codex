"""Tests for dataset factory and split generation."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from gnn_pruning.data import (
    generate_exact_ratio_split,
    get_dataset_loader,
    get_supported_datasets,
    save_split_indices,
)
from gnn_pruning.data import dblp_adapter
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


def test_dblp_loader_requires_explicit_supported_strategy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class _DummyDBLPDataset:
        def __init__(self, root: str) -> None:
            self.root = root

        def __getitem__(self, idx: int):
            _ = idx
            return _dummy_dblp_hetero()

    monkeypatch.setattr(dblp_adapter, "_require_pyg_datasets", lambda: type("D", (), {"DBLP": _DummyDBLPDataset}))
    monkeypatch.setattr(dblp_adapter, "_require_pyg_data", lambda: type("PD", (), {"Data": _DummyData}))

    adapted = load_dataset("dblp", root=tmp_path, dblp_strategy="author_homogeneous")
    graph = adapted[0]

    assert adapted.metadata["adapter_strategy"] == "author_homogeneous"
    assert hasattr(graph, "x")
    assert hasattr(graph, "y")
    assert hasattr(graph, "edge_index")
    assert graph.edge_index.shape[0] == 2


def test_dblp_loader_rejects_unknown_strategy() -> None:
    with pytest.raises(ValueError, match="Unsupported DBLP adaptation strategy"):
        dblp_adapter.load_dblp_dataset("./data", strategy="unknown_strategy")


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


class _DummyStore:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class _DummyHeteroData:
    def __init__(self) -> None:
        self.author_store = _DummyStore(
            x=torch.randn(4, 3),
            y=torch.tensor([0, 1, 0, 1], dtype=torch.long),
            num_nodes=4,
        )
        self.author_to_paper = _DummyStore(
            edge_index=torch.tensor(
                [
                    [0, 1, 1, 2, 3],
                    [0, 0, 1, 1, 2],
                ],
                dtype=torch.long,
            )
        )
        self.edge_types = [("author", "to", "paper")]

    def __getitem__(self, key):
        if key == "author":
            return self.author_store
        if key == ("author", "to", "paper"):
            return self.author_to_paper
        raise KeyError(key)


class _DummyData:
    def __init__(self, x, y, edge_index):
        self.x = x
        self.y = y
        self.edge_index = edge_index


def _dummy_dblp_hetero() -> _DummyHeteroData:
    return _DummyHeteroData()

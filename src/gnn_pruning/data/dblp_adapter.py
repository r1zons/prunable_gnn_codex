"""Explicit DBLP adaptation path for homogeneous baselines.

DBLP in PyG is heterogeneous. This adapter requires an explicit strategy and
never silently returns the raw heterogeneous object.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple, Union

import torch


SUPPORTED_DBLP_STRATEGIES = ("author_homogeneous",)


@dataclass
class AdaptedDBLPDataset:
    """Single-graph dataset wrapper for adapted DBLP views."""

    data: Any
    metadata: Dict[str, Any]

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> Any:
        if idx != 0:
            raise IndexError("Adapted DBLP dataset exposes a single graph at index 0.")
        return self.data


def load_dblp_dataset(
    root: Union[str, Path],
    strategy: str = "author_homogeneous",
) -> AdaptedDBLPDataset:
    """Load DBLP with an explicit heterogeneous-to-homogeneous adaptation strategy."""
    normalized = str(strategy).strip().lower()
    if normalized not in SUPPORTED_DBLP_STRATEGIES:
        raise ValueError(
            "Unsupported DBLP adaptation strategy "
            f"'{strategy}'. Supported strategies: {list(SUPPORTED_DBLP_STRATEGIES)}"
        )

    datasets_module = _require_pyg_datasets()
    raw_dataset = datasets_module.DBLP(root=str(Path(root).expanduser() / "dblp"))
    hetero_data = raw_dataset[0]

    if normalized == "author_homogeneous":
        graph = adapt_dblp_author_homogeneous(hetero_data)
        return AdaptedDBLPDataset(
            data=graph,
            metadata={
                "source_dataset": "dblp",
                "adapter_strategy": normalized,
                "node_view": "author",
                "edge_view": "coauthor_projection_via_shared_paper",
            },
        )

    raise RuntimeError(f"Unhandled DBLP strategy branch: {normalized}")


def adapt_dblp_author_homogeneous(hetero_data: Any) -> Any:
    """Project DBLP to an author-node homogeneous graph.

    Nodes:
    - author nodes only.
    Edges:
    - undirected co-author edges, where two authors connect if they share a paper.
    Labels/features:
    - copied from `hetero_data['author']`.
    """
    data_module = _require_pyg_data()
    author_store = hetero_data["author"]
    x = author_store.x
    y = author_store.y
    num_authors = int(author_store.num_nodes if getattr(author_store, "num_nodes", None) is not None else x.size(0))

    author_to_paper = _find_author_paper_edges(hetero_data)
    edge_index = _build_coauthor_edge_index(author_to_paper, num_authors=num_authors, device=x.device)

    graph = data_module.Data(x=x, y=y, edge_index=edge_index)
    graph.num_nodes = num_authors
    graph.adapter_strategy = "author_homogeneous"
    graph.source_dataset = "dblp"
    return graph


def _find_author_paper_edges(hetero_data: Any) -> torch.Tensor:
    """Find the author->paper edge index in DBLP hetero graph."""
    if not hasattr(hetero_data, "edge_types"):
        raise ValueError("Expected heterogeneous data with edge_types for DBLP adaptation.")
    for edge_type in hetero_data.edge_types:
        src_type, _, dst_type = edge_type
        if src_type == "author" and dst_type == "paper":
            edge_store = hetero_data[edge_type]
            return edge_store.edge_index
    raise ValueError("DBLP adaptation requires an ('author', *, 'paper') relation.")


def _build_coauthor_edge_index(author_paper_edge_index: torch.Tensor, num_authors: int, device: torch.device) -> torch.Tensor:
    """Build undirected co-author edge index from author-paper incidence edges."""
    authors = author_paper_edge_index[0].tolist()
    papers = author_paper_edge_index[1].tolist()

    paper_to_authors: Dict[int, set[int]] = {}
    for author_id, paper_id in zip(authors, papers):
        paper_to_authors.setdefault(int(paper_id), set()).add(int(author_id))

    undirected_edges: set[Tuple[int, int]] = set()
    for group in paper_to_authors.values():
        ids = sorted(group)
        if len(ids) < 2:
            continue
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                undirected_edges.add((a, b))
                undirected_edges.add((b, a))

    if not undirected_edges:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    src, dst = zip(*sorted(undirected_edges))
    edge_index = torch.tensor([list(src), list(dst)], dtype=torch.long, device=device)
    return edge_index


def _require_pyg_datasets() -> Any:
    try:
        from torch_geometric import datasets as pyg_datasets  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "torch_geometric is required to load DBLP. Install project dependencies first."
        ) from exc
    return pyg_datasets


def _require_pyg_data() -> Any:
    try:
        from torch_geometric import data as pyg_data  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "torch_geometric is required to adapt DBLP. Install project dependencies first."
        ) from exc
    return pyg_data

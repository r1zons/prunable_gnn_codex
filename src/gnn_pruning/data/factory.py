"""Dataset factory for PyG datasets used in experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Union

from .dblp_adapter import load_dblp_dataset

DatasetLoader = Callable[[Union[str, Path]], Any]


def get_supported_datasets() -> Dict[str, str]:
    """Return supported dataset names and their backend families."""
    return {
        "cora": "planetoid",
        "citeseer": "planetoid",
        "pubmed": "planetoid",
        "texas": "webkb",
        "cornell": "webkb",
        "wisconsin": "webkb",
        "actor": "actor",
        "amazon_computers": "amazon",
        "flickr": "flickr",
        "reddit": "reddit",
        "dblp": "dblp_adapter",
    }


def get_dataset_loader(name: str) -> DatasetLoader:
    """Resolve dataset name into a loader function."""
    key = _normalize(name)
    family = get_supported_datasets().get(key)
    if family is None:
        raise KeyError(f"Unsupported dataset: {name}")

    if family == "dblp_adapter":
        return load_dblp_dataset

    if family == "planetoid":
        return lambda root: _load_planetoid(root, key)
    if family == "webkb":
        return lambda root: _load_webkb(root, key)
    if family == "actor":
        return _load_actor
    if family == "amazon":
        return _load_amazon_computers
    if family == "flickr":
        return _load_flickr
    if family == "reddit":
        return _load_reddit

    raise KeyError(f"Unhandled dataset family for {name}: {family}")


def load_dataset(name: str, root: Union[str, Path] = "./data") -> Any:
    """Load dataset instance with caching enabled by default in PyG."""
    loader = get_dataset_loader(name)
    return loader(Path(root).expanduser())


def _load_planetoid(root: Union[str, Path], name: str) -> Any:
    module = _require_pyg()
    return module.Planetoid(root=str(Path(root) / "planetoid"), name=name.capitalize())


def _load_webkb(root: Union[str, Path], name: str) -> Any:
    module = _require_pyg()
    return module.WebKB(root=str(Path(root) / "webkb"), name=name.capitalize())


def _load_actor(root: Union[str, Path]) -> Any:
    module = _require_pyg()
    return module.Actor(root=str(Path(root) / "actor"))


def _load_amazon_computers(root: Union[str, Path]) -> Any:
    module = _require_pyg()
    return module.Amazon(root=str(Path(root) / "amazon"), name="Computers")


def _load_flickr(root: Union[str, Path]) -> Any:
    module = _require_pyg()
    return module.Flickr(root=str(Path(root) / "flickr"))


def _load_reddit(root: Union[str, Path]) -> Any:
    module = _require_pyg()
    return module.Reddit(root=str(Path(root) / "reddit"))


def _normalize(name: str) -> str:
    return name.strip().lower()


def _require_pyg() -> Any:
    try:
        from torch_geometric import datasets as pyg_datasets  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "torch_geometric is required to load datasets. Install project dependencies first."
        ) from exc
    return pyg_datasets

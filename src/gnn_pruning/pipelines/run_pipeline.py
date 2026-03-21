"""Pipeline orchestration scaffolding."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from gnn_pruning.config import dump_yaml, resolve_config
from gnn_pruning.config.schema import snapshot_path
from gnn_pruning.data import generate_exact_ratio_split, load_dataset, save_split_indices


@dataclass
class PipelineArtifacts:
    """Paths produced by one pipeline scaffold run."""

    config_snapshot: Path
    split_artifact: Path


def run_pipeline(config_path: str) -> PipelineArtifacts:
    """Load config and persist resolved config plus one reproducible split artifact."""
    resolved = resolve_config(config_path)

    config_out = snapshot_path(resolved.run.output_dir)
    dump_yaml(resolved.to_dict(), config_out)

    dataset = load_dataset(name=resolved.data.name, root=resolved.data.root)
    num_nodes = _infer_num_nodes(dataset)
    split = generate_exact_ratio_split(
        num_nodes=num_nodes,
        seed=resolved.run.seed,
        train_ratio=resolved.data.train_ratio,
        val_ratio=resolved.data.val_ratio,
        test_ratio=resolved.data.test_ratio,
    )
    split_out = save_split_indices(split, resolved.run.output_dir)

    return PipelineArtifacts(config_snapshot=config_out, split_artifact=split_out)


def _infer_num_nodes(dataset: object) -> int:
    """Infer node count from a PyG-like dataset object."""
    if hasattr(dataset, "__getitem__"):
        first = dataset[0]
        if hasattr(first, "num_nodes"):
            return int(first.num_nodes)

    data_obj = getattr(dataset, "data", None)
    if data_obj is not None and hasattr(data_obj, "num_nodes"):
        return int(data_obj.num_nodes)

    raise ValueError("Unable to infer num_nodes from dataset object.")

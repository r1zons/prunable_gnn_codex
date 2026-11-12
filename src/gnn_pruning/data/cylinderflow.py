"""Official CylinderFlow trajectory records converted to NumPy archives."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch


NORMAL, OUTFLOW, NODE_TYPES = 0, 5, 9


class Mesh(NamedTuple):
    """Time-independent directed edges, edge features and node types."""

    senders: torch.Tensor
    receivers: torch.Tensor
    edge_features: torch.Tensor
    node_type: torch.Tensor

    @property
    def active(self) -> torch.Tensor:
        """Nodes updated by the official CFD rollout and used in the loss."""
        return (self.node_type == NORMAL) | (self.node_type == OUTFLOW)

    def to(self, device: str | torch.device) -> "Mesh":
        """Return a mesh on the requested device."""
        return Mesh(*(tensor.to(device) for tensor in self))


def mesh_from_triangles(
    cells: torch.Tensor, mesh_pos: torch.Tensor, node_type: torch.Tensor
) -> Mesh:
    """Deduplicate undirected triangle sides, then emit both directions."""
    sides = torch.cat((cells[:, [0, 1]], cells[:, [1, 2]], cells[:, [2, 0]]))
    low, high = torch.sort(sides, dim=1).values.unique(dim=0).T
    senders = torch.cat((high, low))
    receivers = torch.cat((low, high))
    displacement = mesh_pos[senders] - mesh_pos[receivers]
    edge_features = torch.cat((displacement, displacement.norm(dim=1, keepdim=True)), dim=1)
    return Mesh(senders, receivers, edge_features, node_type)


@dataclass(frozen=True)
class Trajectory:
    """A single mesh with 600 consecutive two-component velocity fields."""

    split: str
    ordinal: int
    mesh: Mesh
    velocity: torch.Tensor

    @property
    def identity(self) -> str:
        """Identify the record by its official split and original ordinal."""
        return f"{self.split}:{self.ordinal}"


def load_trajectory(path: Path, *, split: str) -> Trajectory:
    """Read a converted official record and enforce the requested split."""
    with np.load(path, allow_pickle=False) as archive:
        if split not in {"train", "valid", "test"} or archive["split"].item() != split:
            raise ValueError(f"Trajectory is not in the requested {split} split")
        ordinal = archive["ordinal"].item()
        cells, pos, types, velocity = (archive[key] for key in
                                       ("cells", "mesh_pos", "node_type", "velocity"))
    if type(ordinal) is not int or ordinal < 0:
        raise ValueError("Invalid trajectory ordinal")
    if pos.ndim != 2 or pos.shape[1] != 2 or pos.shape[0] == 0:
        raise ValueError("Expected a nonempty [N, 2] mesh")
    n = pos.shape[0]
    if cells.ndim != 2 or cells.shape[1] != 3 or not cells.size:
        raise ValueError("Expected nonempty triangles")
    if types.shape != (n, 1) or velocity.shape != (600, n, 2):
        raise ValueError("Invalid node types or velocity shape")
    if cells.dtype != np.int32 or types.dtype != np.int32:
        raise ValueError("Expected int32 triangle indices and node types")
    if any(a.dtype != np.float32 or not np.isfinite(a).all() for a in (pos, velocity)):
        raise ValueError("Expected finite float32 positions and velocities")
    if cells.min() < 0 or cells.max() >= n or np.any(cells == cells[:, [1, 2, 0]]):
        raise ValueError("Invalid triangle indices")
    if types.min() < 0 or types.max() >= NODE_TYPES or not np.isin(types, [NORMAL, OUTFLOW]).any():
        raise ValueError("Invalid or inactive node types")
    mesh = mesh_from_triangles(torch.from_numpy(cells).long(), torch.from_numpy(pos),
                               torch.from_numpy(types[:, 0]).long())
    return Trajectory(split, ordinal, mesh, torch.from_numpy(velocity))

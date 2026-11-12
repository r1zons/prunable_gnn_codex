"""Two-component MeshGraphNet for the official fixed-mesh CylinderFlow task."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch import nn
from torch.nn.functional import one_hot

from gnn_pruning.data.cylinderflow import Mesh, NODE_TYPES, NORMAL
from gnn_pruning.training.checkpoints import save_checkpoint


LATENT = 128
PROCESSORS = 15
MIN_WIDTH = 4
CHECKPOINT_FORMAT = "meshgraphnet-cylinderflow-v1"


class Normalizer(nn.Module):
    """Accumulate reference per-feature moments in persistent, non-trainable buffers."""

    def __init__(self, size: int):
        super().__init__()
        self.register_buffer("count", torch.zeros(()))
        self.register_buffer("updates", torch.zeros((), dtype=torch.long))
        self.register_buffer("sum", torch.zeros(size))
        self.register_buffer("sum_sq", torch.zeros(size))

    def moments(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Use identity statistics before fitting and cap small standard deviations."""
        count = self.count.clamp_min(1)
        mean = self.sum / count
        variance = (self.sum_sq / count - mean.square()).clamp_min(0)
        std = torch.where(self.count > 0, variance.sqrt().clamp_min(1e-8), 1.0)
        return mean, std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.updates < 1_000_000:
            with torch.no_grad():
                self.sum.add_(x.sum(dim=0))
                self.sum_sq.add_(x.square().sum(dim=0))
                self.count.add_(x.shape[0])
                self.updates.add_(1)
        mean, std = self.moments()
        return (x - mean) / std

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Return a normalized prediction to physical units."""
        mean, std = self.moments()
        return x * std + mean


class MLP(nn.Sequential):
    """Two hidden Linear/ReLU pairs, a linear output and optional output LayerNorm."""

    def __init__(self, inputs: int, outputs: int, width: int = LATENT, *, layernorm: bool = True):
        modules = [nn.Linear(inputs, LATENT), nn.ReLU(), nn.Linear(LATENT, width),
                   nn.ReLU(), nn.Linear(width, outputs)]
        if layernorm:
            modules.append(nn.LayerNorm(outputs))
        super().__init__(*modules)


def sum_incoming(edges: torch.Tensor, receivers: torch.Tensor, nodes: int) -> torch.Tensor:
    """Sum each edge feature into its receiver's row."""
    return edges.new_zeros((nodes, edges.shape[1])).index_add_(0, receivers, edges)


class Processor(nn.Module):
    """Update edges, aggregate into nodes, then add both residuals."""

    def __init__(self, edge_width: int, node_width: int):
        super().__init__()
        self.edge = MLP(3 * LATENT, LATENT, edge_width)
        self.node = MLP(2 * LATENT, LATENT, node_width)

    def forward(self, nodes: torch.Tensor, edges: torch.Tensor, mesh: Mesh):
        delta_edges = self.edge(torch.cat((nodes[mesh.senders], nodes[mesh.receivers], edges), dim=1))
        incoming = sum_incoming(delta_edges, mesh.receivers, nodes.shape[0])
        delta_nodes = self.node(torch.cat((nodes, incoming), dim=1))
        return nodes + delta_nodes, edges + delta_edges


class MeshGraphNet(nn.Module):
    """Encode, process in 15 independent blocks, and predict two velocity increments."""

    def __init__(self, widths: Sequence[Sequence[int]] = ((LATENT, LATENT),) * PROCESSORS):
        super().__init__()
        if len(widths) != PROCESSORS or any(len(pair) != 2 for pair in widths):
            raise ValueError("Expected edge and node widths for all 15 processors")
        if any(type(w) is not int or not MIN_WIDTH <= w <= LATENT for pair in widths for w in pair):
            raise ValueError("Invalid second-hidden widths")
        self.node_norm = Normalizer(2 + NODE_TYPES)
        self.edge_norm = Normalizer(3)
        self.output_norm = Normalizer(2)
        self.node_encoder = MLP(2 + NODE_TYPES, LATENT)
        self.edge_encoder = MLP(3, LATENT)
        self.processors = nn.ModuleList(Processor(*pair) for pair in widths)
        self.decoder = MLP(LATENT, 2, layernorm=False)

    def widths(self) -> list[list[int]]:
        """Report current second-hidden widths in processor, edge/node order."""
        return [[block.edge[2].out_features, block.node[2].out_features]
                for block in self.processors]

    def forward(self, mesh: Mesh, velocity: torch.Tensor) -> torch.Tensor:
        """Return normalized increments before integration and boundary enforcement."""
        node_features = torch.cat((velocity, one_hot(mesh.node_type, NODE_TYPES).to(velocity)), dim=1)
        nodes = self.node_encoder(self.node_norm(node_features))
        edges = self.edge_encoder(self.edge_norm(mesh.edge_features))
        for block in self.processors:
            nodes, edges = block(nodes, edges, mesh)
        return self.decoder(nodes)

    def step(self, mesh: Mesh, velocity: torch.Tensor) -> torch.Tensor:
        """Integrate one predicted increment and leave fixed boundary nodes untouched."""
        predicted = velocity + self.output_norm.inverse(self(mesh, velocity))
        return torch.where(mesh.active[:, None], predicted, velocity)

    def training_loss(
        self, mesh: Mesh, current: torch.Tensor, target: torch.Tensor, *, noise_std: float
    ) -> torch.Tensor:
        """Compute the reference normalized squared-error loss with noise gamma equal to one."""
        if not self.training:
            raise ValueError("Training loss requires train mode")
        noise = torch.randn_like(current) * noise_std
        current = current + torch.where(mesh.node_type[:, None] == NORMAL, noise, 0.0)
        prediction = self(mesh, current)
        target_increment = self.output_norm(target - current)
        return (prediction - target_increment).square().sum(dim=1)[mesh.active].mean()


@torch.no_grad()
def rollout(model: MeshGraphNet, mesh: Mesh, initial: torch.Tensor, steps: int) -> torch.Tensor:
    """Return frames [initial, prediction_1, ..., prediction_steps] with feedback."""
    if model.training or steps < 1:
        raise ValueError("Rollout requires eval mode and a positive step count")
    frames = [initial]
    for _ in range(steps):
        frames.append(model.step(mesh, frames[-1]))
    return torch.stack(frames)


def save_mesh_checkpoint(model: MeshGraphNet, path: Path) -> Path:
    """Save physical widths, weights and normalization state."""
    return save_checkpoint({"format": CHECKPOINT_FORMAT, "widths": model.widths(),
                            "state_dict": model.state_dict()}, path)


def load_mesh_checkpoint(path: Path, device: str | torch.device = "cpu") -> MeshGraphNet:
    """Reconstruct the exact architecture, strictly load state, and enter eval mode."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["format"] != CHECKPOINT_FORMAT:
        raise ValueError("Unsupported MeshGraphNet checkpoint format")
    model = MeshGraphNet(payload["widths"])
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device).eval()

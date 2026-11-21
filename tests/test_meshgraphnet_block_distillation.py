"""Check the feature loss, frozen parameters and reference-input diagnostic."""

from collections import Counter
from pathlib import Path
import sys

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_cylinderflow_block_distillation import balanced_orders, train_block, frozen_hash, LABELS
from run_cylinderflow_channel_selection import score_rollout


def test_reference_inputs_differ_from_autoregressive():
    class Increment:
        def step(self, mesh, velocity):
            return velocity + mesh.active[:, None]

    mesh = type("Mesh", (), {"active": torch.tensor([True, False])})()
    frames = torch.tensor([[[0., 0.], [0., 0.]], [[2., 2.], [1., 1.]], [[4., 4.], [2., 2.]]])
    assert score_rollout(Increment(), mesh, frames)["curve"] == [1., 4.]
    assert score_rollout(Increment(), mesh, frames, reference_inputs=True)["curve"] == [1., 1.]


def test_balanced_orders():
    orders = balanced_orders()
    assert all(set(column) == set(LABELS) for column in zip(*orders))
    assert set(Counter(pair for order in orders for pair in zip(order, order[1:])).values()) == {1}


def test_only_retained_block_optimized(tmp_path):
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.zeros(()))

        def forward(self, nodes, edges, mesh):
            return nodes * self.scale, edges * self.scale

    model = nn.Module()
    model.processors = nn.ModuleList(Block() for _ in range(14))
    model.requires_grad_(False)
    model.processors[6].requires_grad_(True)
    unchanged = frozen_hash(model, 6)
    nodes, edges = torch.ones(2, 1), torch.ones(5, 1)
    item = (None, nodes, edges, nodes, edges)
    cfg = dict(seed=101, epochs=2, optimizer="Adam", learning_rate=.1,
               checkpoint_selection="lowest_heldout_feature_loss_earliest_tie_including_initial")
    state, epoch, hist = train_block(model, [item], [item], cfg, tmp_path)
    assert epoch == 2
    assert hist[0]["train"] == dict(node_mse=1., edge_mse=1., equal_weight_mse=1.)
    assert hist[-1]["heldout"]["equal_weight_mse"] < 1
    assert state["scale"] > 0
    assert frozen_hash(model, 6) == unchanged

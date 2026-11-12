"""Focused structural and dynamical checks without downloading a dataset."""

import numpy as np
import pytest
import torch

from gnn_pruning.data.cylinderflow import load_trajectory, mesh_from_triangles
from gnn_pruning.models.meshgraphnet import MeshGraphNet, load_mesh_checkpoint, rollout
from gnn_pruning.models.meshgraphnet import save_mesh_checkpoint, sum_incoming
from gnn_pruning.surgery.meshgraphnet import prune_processor_hidden


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(original)


@pytest.fixture
def square():
    cells = torch.tensor([[0, 1, 2], [1, 3, 2]])
    positions = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
    types = torch.tensor([0, 4, 5, 6])
    return cells, positions, types


def test_trajectory_identity_and_edges(tmp_path, square):
    cells, pos, types = square
    path = tmp_path / "trajectory.npz"
    np.savez(path, split="valid", ordinal=7, cells=cells.numpy().astype(np.int32),
             mesh_pos=pos.numpy(), node_type=types[:, None].numpy().astype(np.int32),
             velocity=np.zeros((600, 4, 2), dtype=np.float32))
    trajectory = load_trajectory(path, split="valid")
    assert trajectory.identity == "valid:7"
    with pytest.raises(ValueError, match="requested train"):
        load_trajectory(path, split="train")
    mesh = trajectory.mesh
    pairs = list(zip(mesh.senders.tolist(), mesh.receivers.tolist()))
    assert len(pairs) == len(set(pairs)) == 10
    assert set(pairs) == {(0, 1), (0, 2), (1, 0), (1, 2), (1, 3),
                          (2, 0), (2, 1), (2, 3), (3, 1), (3, 2)}
    edge = pairs.index((2, 1))
    torch.testing.assert_close(mesh.edge_features[edge], torch.tensor([-1., 1., 2.**0.5]))
    values = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
    torch.testing.assert_close(sum_incoming(values, torch.tensor([0, 2, 2]), 4),
                               torch.tensor([[1., 2.], [0., 0.], [8., 10.], [0., 0.]]))


def test_rollout_integrates_denormalized_feedback_and_keeps_boundaries(square, monkeypatch):
    mesh = mesh_from_triangles(*square)
    model = MeshGraphNet()
    model.output_norm(torch.tensor([[1., 0.], [3., 4.]]))
    model.eval()
    state = {k: v.clone() for k, v in model.state_dict().items()}
    monkeypatch.setattr(model, "forward", lambda _mesh, velocity: velocity)
    initial = torch.tensor([[1., 2.], [7., 8.], [2., 3.], [9., 10.]])
    expected = torch.stack((initial,
                            torch.tensor([[4., 8.], [7., 8.], [6., 11.], [9., 10.]]),
                            torch.tensor([[10., 26.], [7., 8.], [14., 35.], [9., 10.]])))
    torch.testing.assert_close(rollout(model, mesh, initial, 2), expected, rtol=0, atol=0)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["edge", "node"])
def test_physical_projection_and_strict_reload(kind, square, tmp_path):
    torch.manual_seed(101)
    mesh = mesh_from_triangles(*square)
    dense = MeshGraphNet()
    velocity = torch.randn(4, 2)
    dense.training_loss(mesh, velocity, velocity + 1, noise_std=0).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in dense.parameters())
    dense.eval()
    original = {name: value.clone() for name, value in dense.state_dict().items()}
    removed = {2, 4, 6, 8, 10, 12, 14, 16}
    keep = [channel for channel in range(128) if channel not in removed]
    pruned = prune_processor_hidden(dense, 4, kind, keep)
    layer = getattr(dense.processors[4], kind)
    pruned_layer = getattr(pruned.processors[4], kind)
    assert (pruned_layer[2].out_features, pruned_layer[4].in_features) == (120, 120)
    assert pruned_layer[-1].normalized_shape == (128,)
    torch.testing.assert_close(pruned_layer[2].weight, layer[2].weight[keep], rtol=0, atol=0)
    torch.testing.assert_close(pruned_layer[4].weight, layer[4].weight[:, keep], rtol=0, atol=0)
    mask = torch.tensor([i in keep for i in range(128)])
    with torch.no_grad():
        handle = layer[3].register_forward_hook(lambda _m, _x, activation: activation * mask)
        try:
            masked = dense(mesh, velocity)
        finally:
            handle.remove()
        torch.testing.assert_close(pruned(mesh, velocity), masked, rtol=1e-4, atol=1e-5)
    assert sum(p.numel() for p in dense.parameters()) == 2_332_930
    assert sum(p.numel() for p in pruned.parameters()) == 2_330_874
    assert len({block.edge[2].weight.data_ptr() for block in dense.processors}) == 15
    for name, value in dense.state_dict().items():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)
    for label, model in (("dense", dense), ("pruned", pruned)):
        path = save_mesh_checkpoint(model, tmp_path / f"{label}.pt")
        loaded = load_mesh_checkpoint(path)
        assert not loaded.training
        assert loaded.widths() == model.widths()
        for name, value in loaded.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)
        torch.testing.assert_close(rollout(loaded, mesh, velocity, 2),
                                   rollout(model, mesh, velocity, 2), rtol=0, atol=0)
    payload = torch.load(path, weights_only=True)
    del payload["state_dict"]["output_norm.count"]
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_mesh_checkpoint(path)

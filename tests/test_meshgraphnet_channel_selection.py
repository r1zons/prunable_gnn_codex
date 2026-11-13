"""Check the controlled comparison without opening the official data."""

import json
import sys
from pathlib import Path

import pytest
import torch
import yaml

from gnn_pruning.data.cylinderflow import mesh_from_triangles
from gnn_pruning.models.meshgraphnet import MeshGraphNet, load_mesh_checkpoint, save_mesh_checkpoint
from gnn_pruning.surgery.meshgraphnet import prune_processor_hidden_layers

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_cylinderflow_channel_selection as run


@pytest.fixture
def cfg():
    return yaml.safe_load(run.CONFIG.read_text())


@pytest.fixture
def example():
    cells = torch.tensor([[0, 1, 2], [1, 3, 2]])
    mesh = mesh_from_triangles(cells, torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.]]),
                               torch.tensor([0, 4, 5, 6]))
    frames = torch.zeros(6, 4, 2)
    return mesh, frames


def test_candidate_library_and_tie(cfg):
    ranks = [list(reversed(range(128)))] * 30
    parent = ((),) * 30
    for _ in range(15):
        draws = run.library(ranks, parent, cfg)
        assert draws == run.library(ranks, parent, cfg)
        assert len(draws) == len(set(draws)) == 8
        assert all(len(pair) == 2 and not set(pair) & set(old)
                   for action in draws for pair, old in zip(action, parent))
        allowed = [set([c for c in rank if c not in old][:8]) for rank, old in zip(ranks, parent)]
        assert all(set(pair) <= pool for action in draws[:5] for pair, pool in zip(action, allowed))
        parent = run.extend(parent, draws[4])
    assert len(parent[0]) == 30
    candidates = [(2,), (1,), (0,), (-1,)]
    assert run.choose(candidates, [1.0, 1.00001, 1.001, float("nan")], rtol=.0001) == 1
    assert run.choose(candidates, [1.0, 1.00001, 1.001, None]) == 0
    assert run.choose(candidates, [None] * 4) is None


def test_nested_physical_widths_and_reload(cfg, example, tmp_path):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(101)
        dense = MeshGraphNet().eval()
        frozen = run.tensors_hash(dense.state_dict())
        mesh, frames = example
        ranks = [list(range(128))] * 30
        parent = ((),) * 30
        sequential = dense
        for n in range(1, 16):
            action = run.library(ranks, parent, cfg)[-1]
            current_keeps = {}
            for layer, old, new in zip(run.LAYERS, parent, action):
                ids = [i for i in range(128) if i not in old]
                current_keeps[layer] = [k for k, original in enumerate(ids) if original not in new]
            sequential = prune_processor_hidden_layers(sequential, current_keeps)
            parent = run.extend(parent, action)
            if n not in cfg["snapshot_decisions"]:
                continue
            model, count = run.project(dense, parent)
            assert count == {2: 2302090, 4: 2271250, 8: 2209570, 15: 2101630}[n]
            assert model.widths() == [[128 - 2 * n] * 2] * 15
            assert run.tensors_hash(model.state_dict()) == run.tensors_hash(sequential.state_dict())
            assert run.parity(dense, model, list(map(list, parent)), mesh, frames[0]) < 1e-5
            path = save_mesh_checkpoint(model, tmp_path / "physical.pt")
            assert run.tensors_hash(load_mesh_checkpoint(path).state_dict()) == run.tensors_hash(model.state_dict())
        assert run.tensors_hash(dense.state_dict()) == frozen
    finally:
        torch.set_num_threads(previous_threads)


def test_rollout_feedback_failure_and_split(example):
    mesh, frames = example

    class Model:
        def step(self, mesh, velocity):
            return velocity + mesh.active[:, None]

    row = run.score_rollout(Model(), mesh, frames)
    assert row["curve"] == [1., 4., 9., 16., 25.]
    assert row["forward_calls"] == 5
    assert run.score_candidate(Model(), [(58, mesh, frames), (12, mesh, frames * 2)], 1)[0] == 1

    class Invalid:
        def step(self, mesh, velocity):
            return torch.where(mesh.active[:, None], float("nan"), velocity)

    score, rows = run.score_candidate(Invalid(), [(58, mesh, frames)], 5)
    assert score is None and rows[0]["first_failed_step"] == 1
    assert rows[0]["status"] == "nonfinite" and rows[0]["forward_calls"] == 5
    for split, ids in (("test", [58]), ("valid", run.BASELINE)):
        with pytest.raises(ValueError):
            run.load_windows(type("Records", (), {"split": split})(), ids, [200], 5, purpose="selection")
    splits = run.split_manifest()
    assert set(splits["selection"]).isdisjoint(splits["diagnostic"])


def test_resume_between_decisions(cfg, tmp_path):
    cfg["selection_horizons"] = {"magnitude": 0}
    cfg["snapshot_decisions"] = []
    ranks = [list(range(128))] * 30
    resumed, direct = tmp_path / "resumed", tmp_path / "direct"
    resumed.mkdir()
    direct.mkdir()
    gates = iter([True] * 5 + [False])
    assert run.search(None, ranks, [], resumed, cfg, lambda _: next(gates))[1] is False
    assert run.search(None, ranks, [], resumed, cfg, lambda _: True)[1] is True
    assert run.search(None, ranks, [], direct, cfg, lambda _: True)[1] is True
    for i in range(1, 16):
        records = [json.loads((p / "magnitude" / f"decision-{i:02}.json").read_text())
                   for p in (resumed, direct)]
        for row in records:
            row.pop("search_seconds")
        assert records[0] == records[1]
        assert records[0]["winner"] == 0
        assert records[0]["forward_calls"] == 0

"""Check the indexed record and training-worker primitives on CPU."""

import random
import struct

import numpy as np
import torch

from gnn_pruning.data import cylinderflow_records as records
from gnn_pruning.experiments.meshgraphnet_training import atomic_save, train_batch, validate


def _varint(number):
    if number < 128:
        return bytes([number])
    return bytes([(number & 127) | 128]) + _varint(number >> 7)


def _field(tag, content):
    return _varint(tag * 8 + 2) + _varint(len(content)) + content


def test_frame_reads_and_disjoint_batch(tmp_path, monkeypatch):
    velocity = np.arange(600 * 3 * 2, dtype="<f4").reshape(600, 3, 2)
    values = {
        "cells": np.array([[0, 1, 2]], dtype="<i4"),
        "mesh_pos": np.array([[0, 0], [0, 1], [1, 0]], dtype="<f4"),
        "node_type": np.array([0, 4, 5], dtype="<i4"),
        "velocity": velocity,
        "pressure": np.zeros((600, 3), dtype="<f4"),
    }
    features = b"".join(_field(1, _field(1, name.encode()) + _field(
        2, _field(1, _field(1, values[name].tobytes())))) for name in values)
    payload = _field(1, features)
    record = struct.pack("<QI", len(payload), 0) + payload + bytes(4)
    path = tmp_path / "train.tfrecord"
    path.write_bytes(record * 2)
    monkeypatch.setitem(records.OBJECTS, "train", (2 * len(record), 2, 123))
    indexed = records.CylinderFlowRecords(path, "train")
    torch.testing.assert_close(indexed.velocity(1, 55, 2), torch.from_numpy(velocity[55:57]))
    mesh, current, target = train_batch(indexed, random.Random(101), "cpu")
    assert mesh.node_type.tolist() == [0, 4, 5, 0, 4, 5]
    assert torch.equal(mesh.senders < 3, mesh.receivers < 3)
    torch.testing.assert_close(target - current, torch.full((6, 2), 6.0))

    class Persistence(torch.nn.Module):
        def step(self, mesh, current):
            return current

    result = validate(Persistence(), indexed, [0, 1], [100, 300], 100, "cpu", lambda: False)
    assert result["finite_windows"] == 4
    assert result["e1"] == 36 and result["e20"] == 14400
    assert result["e100"] == result["persistence_e100"] == 360000
    assert validate(Persistence(), indexed, [0], [100], 100, "cpu", lambda: True) is None


def test_atomic_resumable_payload(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    sampler = random.Random(101)
    path = tmp_path / "latest.pt"
    state = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
             "sampler": sampler.getstate(), "torch_rng": torch.get_rng_state(), "updates": 1}
    atomic_save(state, path)
    loaded = torch.load(path, weights_only=True)
    model.load_state_dict(loaded["state_dict"], strict=True)
    optimizer.load_state_dict(loaded["optimizer"])
    expected = sampler.random()
    sampler.setstate(loaded["sampler"])
    assert sampler.random() == expected and loaded["updates"] == 1
    assert not path.with_suffix(".partial").exists()

"""Export and audit the frozen baseline-validation CylinderFlow rollouts."""

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from gnn_pruning.config.loader import load_yaml
from gnn_pruning.data.cylinderflow import mesh_from_triangles
from gnn_pruning.data.cylinderflow_records import CylinderFlowRecords
from gnn_pruning.experiments.flickr_dense import sha, tensors_hash, write
from gnn_pruning.models.meshgraphnet import load_mesh_checkpoint, rollout


def evaluate(parent, output):
    """Use the exact job-1618479 checkpoint and windows, exporting physical fields."""
    started = time.perf_counter()
    run = parent / "continuation-02" / "run.1618479"
    path = run / "best.pt"
    checkpoint_hash = sha(path)
    assert checkpoint_hash == "d4a3360760f191d4b05ee80111a32067f33aea5609466b12328b4e802b2a1386"
    source = (parent / "source-commit.txt").read_text().strip()
    assert source == "78d46a96214bb319872902859ffef828c400393f"
    config = json.loads((run / "config.json").read_text())
    assert config == load_yaml(parent / "configs/experiments/meshgraphnet_cylinderflow_training.yaml")
    events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
    saved = next(row for row in events if row["event"] == "validation" and row["updates"] == 47000)
    assert events[-1]["best"] == {"updates": 47000, "e100": saved["e100"]}
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    assert checkpoint["updates"] == 47000 and checkpoint["widths"] == [[128, 128]] * 15
    assert all(checkpoint["validation"][k] == saved[k] for k in checkpoint["validation"])
    data = parent / "data"
    assert checkpoint["dataset_identity"] == (data / "VERIFIED.md5").read_text()
    metadata = json.loads((data / "meta.json").read_text())
    assert metadata["dt"] == 0.01
    records = CylinderFlowRecords(data / "valid.tfrecord", "valid")
    ids = list(range(len(records)))
    random.Random(config["validation_seed"]).shuffle(ids)
    ids = ids[:config["validation_trajectories"]]
    starts, steps = config["validation_starts"], config["validation_horizon"]
    assert ids == checkpoint["validation_ids"] == events[0]["validation_ids"]
    assert len(ids) == 16 and starts == [100, 300] and steps == 100
    fixed = (ids[0], starts[0])
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert "P100" in torch.cuda.get_device_name()
    model = load_mesh_checkpoint(path, "cuda:0")
    original = tensors_hash(model.state_dict())
    normalization = {}
    for name in ("node_norm", "edge_norm", "output_norm"):
        norm = getattr(model, name)
        mean, std = norm.moments()
        assert norm.updates.item() == 48000 and norm.count.item() > 0
        assert torch.isfinite(mean).all() and torch.isfinite(std).all() and (std > 0).all()
        normalization[name] = {"updates": norm.updates.item(), "count": norm.count.item(),
                               "mean": mean.tolist(), "std": std.tolist()}

    output.mkdir(exist_ok=False)
    windows = []
    with torch.inference_mode():
        for ordinal in ids:
            geometry = records.geometry(ordinal)
            mesh = mesh_from_triangles(*geometry).to("cuda:0")
            active = mesh.active
            for start in starts:
                reference = records.velocity(ordinal, start, steps + 1).cuda()
                prediction = rollout(model, mesh, reference[0], steps)
                assert prediction.shape == reference.shape and torch.isfinite(prediction).all()
                assert torch.equal(prediction[0], reference[0])
                assert torch.equal(prediction[:, ~active], reference[0, ~active].expand_as(prediction[:, ~active]))
                mse = torch.stack([(p[active] - r[active]).square().mean()
                                   for p, r in zip(prediction[1:], reference[1:])]).tolist()
                persistence = torch.stack([(reference[0, active] - r[active]).square().mean()
                                           for r in reference[1:]]).tolist()
                filename = f"valid-{ordinal:03d}-start-{start}.npz"
                np.savez(output / filename, cells=geometry[0].numpy(), xy=geometry[1].numpy(),
                         node_type=geometry[2].numpy(), active=active.cpu().numpy(),
                         reference=reference.cpu().numpy(), prediction=prediction.cpu().numpy(),
                         frames=np.arange(start, start + steps + 1))
                windows.append({"ordinal": ordinal, "start": start, "file": filename,
                                "model": mse, "persistence": persistence})
            print(f"Completed valid:{ordinal}", flush=True)
    assert original == tensors_hash(model.state_dict())
    aggregate = {method: np.asarray([row[method] for row in windows]).reshape(16, 2, steps)
                 .mean(axis=1).mean(axis=0).tolist() for method in ("model", "persistence")}
    deltas = {f"e{step}": aggregate["model"][step - 1] - saved[f"e{step}"] for step in (1, 20, 100)}
    deltas["persistence_e100"] = aggregate["persistence"][99] - saved["persistence_e100"]
    deltas["mean_mse"] = float(np.mean(aggregate["model"])) - saved["mean_mse"]
    write(output / "evaluation.json", {
        "checkpoint_sha256": checkpoint_hash, "tensor_sha256": original, "updates": 47000,
        "training_job": 1618479, "training_source": source, "config": config, "saved": saved,
        "dataset_identity": checkpoint["dataset_identity"], "normalizers": normalization,
        "ids": ids, "starts": starts, "fixed": fixed, "dt_seconds": metadata["dt"],
        "windows": windows, "aggregate": aggregate, "delta_vs_logged": deltas,
        "torch": str(torch.__version__), "gpu": torch.cuda.get_device_name(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "seconds": time.perf_counter() - started,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    evaluate(args.training_root, args.output)

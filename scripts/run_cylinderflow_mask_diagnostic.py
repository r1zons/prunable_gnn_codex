"""Run the frozen CylinderFlow mask diagnostic, with one atomic record per window."""

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch

from gnn_pruning.data.cylinderflow_records import CylinderFlowRecords
from gnn_pruning.experiments.flickr_dense import sha, tensors_hash
from gnn_pruning.models.meshgraphnet import load_mesh_checkpoint, save_mesh_checkpoint
from gnn_pruning.surgery.meshgraphnet import prune_processor_hidden

CHECKPOINT = "d4a3360760f191d4b05ee80111a32067f33aea5609466b12328b4e802b2a1386"
TENSORS = "ba1b9d7bc024205428be3ad6f60d23e9034e21f421d00ddb0c7c5b6a03332b08"
LAYERS = [(i, kind) for i in range(15) for kind in ("edge", "node")]
BASELINE = [31, 7, 40, 3, 48, 96, 67, 10, 65, 35, 66, 99, 89, 71, 76, 33]
STARTS = (100, 300)
ATOL, RTOL = 1e-5, 1e-4


def split_manifest():
    ids = list(range(100))
    random.Random(101).shuffle(ids)
    assert ids[:16] == BASELINE
    return dict(baseline=ids[:16], diagnostic=ids[16:24], selection=ids[24:32],
                starts=list(STARTS), horizon=100, split="valid")


def seal(path, value):
    """Publish a complete JSON record and require immutable metadata on resume."""
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Existing record differs: {path}")
        return
    tmp = path.with_suffix(".partial")
    with tmp.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def masks(model):
    """Create the fixed library from CPU float64 dense importance ranks."""
    rng = random.Random(101)
    ranks = []
    for block, kind in LAYERS:
        mlp = getattr(model.processors[block], kind)
        incoming = mlp[2].weight.detach().cpu().double()
        bias = mlp[2].bias.detach().cpu().double()
        outgoing = mlp[4].weight.detach().cpu().double()
        scores = (incoming.square().sum(1) + bias.square()).sqrt() * outgoing.norm(dim=0)
        assert scores.isfinite().all() and len(scores) == 128
        ranks.append(sorted(range(128), key=lambda channel: (scores[channel].item(), channel)))
    library, seen = [], set()
    for name, lo, hi in (("lower", 0, 43), ("middle", 43, 86), ("upper", 86, 128)):
        for number in range(1, 9):
            while True:
                removed = tuple(tuple(sorted(rng.sample(rank[lo:hi], 4))) for rank in ranks)
                if removed not in seen:
                    break
            seen.add(removed)
            library.append(dict(id=f"{name}-{number:02}", stratum=name,
                                removed=[list(ids) for ids in removed]))
    return dict(layers=[list(layer) for layer in LAYERS], ranks=ranks, masks=library)


def compress(dense, removed):
    """Physically project every processor MLP using the existing surgery."""
    model = dense
    for (block, kind), ids in zip(LAYERS, removed):
        model = prune_processor_hidden(model, block, kind,
                                       sorted(set(range(128)).difference(ids)))
    assert model.widths() == [[124, 124]] * 15
    assert sum(p.numel() for p in model.parameters()) == 2302090
    assert tensors_hash(dict(model.named_buffers())) == tensors_hash(dict(dense.named_buffers()))
    return model


@torch.inference_mode()
def parity(dense, model, removed, mesh, initial):
    """Check the physical projection against dense activation masking."""
    with ExitStack() as hooks:
        for (block, kind), ids in zip(LAYERS, removed):
            activation = getattr(dense.processors[block], kind)[3]
            keep = torch.ones(128, dtype=torch.bool, device=initial.device)
            keep[ids] = False
            handle = activation.register_forward_hook(lambda _m, _i, y, keep=keep: y * keep)
            hooks.callback(handle.remove)
        expected = dense(mesh, initial)
    actual = model(mesh, initial)
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
    return (actual - expected).abs().max().item()


@torch.inference_mode()
def window(model, mesh, frames):
    """Use the reference at step zero, then record errors and first nonfinite step."""
    velocity = frames[0]
    predictions = []
    failure = None
    for _ in range(100):
        try:
            velocity = model.step(mesh, velocity) if model is not None else velocity
            predictions.append(velocity)
        except RuntimeError as exc:
            failure = str(exc)
            break
    if not predictions:
        return dict(status="runtime_error", first_failed_step=1, curve=[], error=failure)
    fields = torch.stack(predictions)
    fixed = ~mesh.active
    assert torch.equal(fields[:, fixed], frames[0, fixed].expand_as(fields[:, fixed]))
    curve = (fields[:, mesh.active] - frames[1:1+len(fields), mesh.active]).square().mean(dim=(1, 2))
    good = (torch.isfinite(fields).flatten(1).all(1) & torch.isfinite(curve)).tolist()
    first = next((i for i, finite in enumerate(good) if not finite), len(good))
    status = "ok" if first == 100 else "nonfinite"
    if failure is not None and first == len(good):
        status = "runtime_error"
    return dict(status=status, first_failed_step=None if first == 100 else first + 1,
                curve=curve[:first].tolist(), error=failure)


def run(parent, output, receipt, seconds, budget_from):
    start_clock = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    gpu = torch.cuda.get_device_properties(0)
    assert "P100" in gpu.name and (gpu.major, gpu.minor) == (6, 0)
    path = parent / "continuation-02/run.1618479/best.pt"
    assert sha(path) == CHECKPOINT
    ck = torch.load(path, map_location="cpu", weights_only=True)
    assert ck["updates"] == 47000 and ck["validation_ids"] == BASELINE
    assert ck["validation"]["e100"] == 0.002244355198854464
    assert ck["widths"] == [[128, 128]] * 15
    assert tensors_hash(ck["state_dict"]) == TENSORS
    training_source = (parent / "source-commit.txt").read_text().strip()
    assert training_source == "78d46a96214bb319872902859ffef828c400393f"
    data = parent / "data"
    dataset = (data / "VERIFIED.md5").read_text()
    assert ck["dataset_identity"] == dataset
    assert "0a2f5f59112d6aeb7fb42ef974daca4d  valid.tfrecord" in dataset.splitlines()
    splits = split_manifest()
    assert len(set(splits["baseline"] + splits["diagnostic"] + splits["selection"])) == 32
    seal(output / "splits.json", splits)
    dense = load_mesh_checkpoint(path)
    norms = {}
    for name in ("node_norm", "edge_norm", "output_norm"):
        norm = getattr(dense, name)
        mean, std = norm.moments()
        assert norm.updates.item() == 48000 and norm.count.item() > 0
        assert mean.isfinite().all() and std.isfinite().all() and (std > 0).all()
        norms[name] = dict(count=norm.count.item(), updates=norm.updates.item(),
                           mean=mean.tolist(), std=std.tolist())
    assert sum(p.numel() for p in dense.parameters()) == 2332930
    library = masks(dense)
    seal(output / "masks.json", library)
    seal(output / "protocol.json", dict(training_source=training_source, checkpoint_sha256=CHECKPOINT,
        tensor_sha256=TENSORS, normalizers=norms, dataset=dataset,
        evaluation_source=receipt.read_text(), base_revision=Path("source-commit.txt").read_text().strip(),
        rtol=RTOL, atol=ATOL, rank_seed=101,
        thresholds=dict(atol=1e-9, rtol=1e-4), compression=30840/2332930))
    records = CylinderFlowRecords(data / "valid.tfrecord", "valid")
    cached = []
    for ordinal in splits["diagnostic"]:
        mesh = records.mesh(ordinal).to("cuda")
        for start in STARTS:
            cached.append((ordinal, start, mesh, records.velocity(ordinal, start, 101).cuda()))
    cached.sort(key=lambda row: (-row[2].senders.numel(), row[0], row[1]))
    largest = cached[0]
    dense.cuda()
    torch.cuda.reset_peak_memory_stats()
    attempt = output / f"attempt-{os.environ['LSB_JOBID']}.json"
    seal(attempt, dict(host=platform.node(), architecture=platform.machine(), torch=str(torch.__version__),
                       cuda=torch.version.cuda, python=platform.python_version(), numpy=np.__version__,
                       gpu=gpu.name, gpu_bytes=gpu.total_memory, cpu_affinity=sorted(os.sched_getaffinity(0))))
    completed = 0
    for entry in [dict(id="dense"), dict(id="persistence"), *library["masks"]]:
        name = entry["id"]
        folder = output / name
        folder.mkdir(exist_ok=True)
        remaining = [(o,s,m,f) for o,s,m,f in cached if not (folder/f"{o}-{s}.json").exists()]
        if not remaining:
            continue
        if time.time() - budget_from >= seconds:
            break
        began = time.monotonic()
        model = None if name == "persistence" else dense
        if "removed" in entry:
            try:
                model = compress(dense, entry["removed"])
                peak = parity(dense, model, entry["removed"], largest[2], largest[3][0])
                tmp = folder / "physical.partial"
                save_mesh_checkpoint(model, tmp)
                restored = load_mesh_checkpoint(tmp, "cuda")
                model_hash = tensors_hash(model.state_dict())
                assert model_hash == tensors_hash(restored.state_dict())
                assert restored.widths() == model.widths()
                seal(folder / "surgery.json", dict(parameters=2302090, widths=model.widths(),
                                                   tensor_sha256=model_hash))
                seal(folder / f"parity-{os.environ['LSB_JOBID']}.json", dict(masked_max_abs=peak))
                tmp.unlink()
                model = restored
            except (AssertionError, RuntimeError, ValueError) as exc:
                for ordinal, start, _, _ in remaining:
                    seal(folder / f"{ordinal}-{start}.json",
                         dict(status="surgery_failed", first_failed_step=None, curve=[],
                              error=f"{type(exc).__name__}: {exc}"))
                continue
        for ordinal, start, mesh, frames in remaining:
            if time.time() - budget_from >= seconds:
                break
            result = window(model, mesh, frames)
            seal(folder / f"{ordinal}-{start}.json", result)
            completed += 1
        assert tensors_hash(dense.state_dict()) == TENSORS
        print(name, "elapsed", round(time.monotonic()-began, 2), "new windows", completed, flush=True)
    expected = 26 * len(cached)
    total = sum(len(list((output / name).glob("[0-9]*.json")))
                for name in ["dense", "persistence", *(row["id"] for row in library["masks"])])
    failures = sum(json.loads(p.read_text())["status"] != "ok"
                   for p in output.glob("*/[0-9]*.json"))
    status = "partial" if total < expected else ("complete_with_failures" if failures else "complete")
    seal(output / f"finish-{os.environ['LSB_JOBID']}.json", dict(
        status=status, saved_windows=total, failed_windows=failures, expected_windows=expected,
        new_windows=completed, elapsed_seconds=time.monotonic()-start_clock,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(), dense_tensor_sha256=TENSORS))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--seconds", type=int, required=True)
    parser.add_argument("--started", type=float, required=True)
    args = parser.parse_args()
    run(args.parent, args.output, args.receipt, args.seconds, args.started)

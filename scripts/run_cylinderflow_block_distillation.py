"""Distill only processor 7 against a frozen MeshGraphNet teacher on POLUS."""

import argparse
import copy
import json
import os
from pathlib import Path
import platform
import random
import statistics
import time

import numpy as np
import torch
import yaml

from gnn_pruning.data.cylinderflow_records import CylinderFlowRecords
from gnn_pruning.experiments.flickr_dense import sha, tensors_hash
from gnn_pruning.models.meshgraphnet import load_mesh_checkpoint
from run_cylinderflow_channel_selection import score_rollout, VALID_SHA
from run_cylinderflow_linear_block_pilot import (
    BASELINE, CHECKPOINT, CHECKPOINT_PATH, TENSORS, benchmark, load_windows, scientific,
    seal, split_manifest, variant,
)
from run_cylinderflow_linear_exchange import LinearExchange


CONFIG = Path("configs/experiments/meshgraphnet_cylinderflow_distillation.yaml")
CALIBRATION_SHA = "029ec159fc50f718d6536d1d8441e58c7d3e89f8877cfd8cea99991ba0fbb809"
EXCHANGE_SHA = "a4fc282ec69b5cf67fbff76f0ff461cca4ce2ef48c2253b6aaa4405efe3ea718"
FIRST, LAST = 6, 7
LABELS = ("A", "D", "before", "after")


def read(path):
    return json.loads(path.read_text())


def frozen_hash(model, block):
    """Hash all state except the only trainable processor."""
    prefix = f"processors.{block}."
    return tensors_hash({k: v for k, v in model.state_dict().items() if not k.startswith(prefix)})


@torch.no_grad()
def capture(teacher, records, ids, frames, device, first_index=FIRST, last_index=LAST):
    """Capture complete teacher inputs and targets in detached tensors."""
    assert records.split == "train"
    states = {}

    def before(_module, args):
        states["input"] = args[:2]

    def after(_module, _args, output):
        states["target"] = output

    first = teacher.processors[first_index].register_forward_pre_hook(before)
    last = teacher.processors[last_index].register_forward_hook(after)
    samples = []
    try:
        for ordinal in ids:
            mesh = records.mesh(ordinal).to(device)
            for frame in frames:
                states.clear()
                teacher(mesh, records.velocity(ordinal, frame, 1)[0].to(device))
                tensors = (*states["input"], *states["target"])
                assert all(not tensor.requires_grad for tensor in tensors)
                samples.append((mesh, *tensors))
    finally:
        first.remove()
        last.remove()
    return samples


def losses(block, sample):
    """Give node and edge MSE the same weight irrespective of mesh size."""
    mesh, nodes, edges, target_nodes, target_edges = sample
    prediction_nodes, prediction_edges = block(nodes, edges, mesh)
    return (prediction_nodes - target_nodes).square().mean(), (prediction_edges - target_edges).square().mean()


@torch.no_grad()
def features(block, samples):
    """Average feature errors over frames, each trajectory contributing equally."""
    node, edge = torch.stack([torch.stack(losses(block, row)) for row in samples]).mean(0).tolist()
    assert np.isfinite((node, edge)).all()
    return dict(node_mse=node, edge_mse=edge, equal_weight_mse=(node + edge) / 2)


def train_block(after, train, heldout, cfg, out, on_epoch=None, block_index=FIRST):
    """Use held-out training trajectories to select a budgeted checkpoint."""
    block = after.processors[block_index]
    assert cfg["optimizer"] == "Adam"
    assert cfg["checkpoint_selection"] == "lowest_heldout_feature_loss_earliest_tie_including_initial"
    optimizer = torch.optim.Adam(block.parameters(), lr=cfg["learning_rate"])
    assert {id(p) for group in optimizer.param_groups for p in group["params"]} == {
        id(p) for p in after.parameters() if p.requires_grad}
    rng = random.Random(cfg["seed"])
    best_state, best_loss, best_epoch = None, float("inf"), None
    history = []
    for epoch in range(cfg["epochs"] + 1):
        if epoch:
            for index in rng.sample(range(len(train)), len(train)):
                optimizer.zero_grad(set_to_none=True)
                node, edge = losses(block, train[index])
                loss = (node + edge) * .5
                loss.backward()
                optimizer.step()
        metrics = dict(epoch=epoch, updates=epoch * len(train),
                       train=features(block, train), heldout=features(block, heldout))
        seal(out / f"epoch-{epoch:02d}.json", metrics)
        history.append(metrics)
        value = metrics["heldout"]["equal_weight_mse"]
        if value < best_loss:
            best_loss, best_epoch = value, epoch
            best_state = copy.deepcopy(block.state_dict())
        if on_epoch is not None:
            on_epoch(epoch, block, optimizer, rng)
        print("feature", epoch, metrics, flush=True)
    return best_state, best_epoch, history


def verify_processors(models, mesh, velocity):
    """Record actual processor calls on a real diagnostic mesh."""
    checks = {}
    for label, model in models.items():
        calls = []

        def called(module, _args, _output):
            calls.append(type(module).__name__)

        hooks = [b.register_forward_hook(called) for b in model.processors]
        try:
            with torch.inference_mode():
                prediction = model.step(mesh, velocity)
                assert torch.isfinite(prediction).all()
        finally:
            for hook in hooks:
                hook.remove()
        expected = ["Processor"] * (15 if label == "A" else 14)
        if label == "D":
            expected[FIRST] = "LinearExchange"
        assert calls == expected
        checks[label] = calls
    return checks


def balanced_orders():
    """Each variant appears once per position and precedes every other once."""
    return [tuple(LABELS[(i + j) % 4] for j in (0, 1, 3, 2)) for i in range(4)]


def run(parent, pilot, exchange, out):
    """Run fit, reproducibility checks, scientific forecasts and timed rollouts."""
    out.mkdir()
    cfg = yaml.safe_load(CONFIG.read_text())
    assert cfg["blocks_one_based"] == [7, 8] and cfg["horizon"] == 100
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    gpu = torch.cuda.get_device_properties(0)
    assert torch.cuda.device_count() == 1 and "P100" in gpu.name and (gpu.major, gpu.minor) == (6, 0)
    checkpoint = parent / CHECKPOINT_PATH
    assert sha(checkpoint) == CHECKPOINT
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["updates"] == 47000 and payload["validation_ids"] == BASELINE
    assert payload["dataset_identity"] == (parent / "data/VERIFIED.md5").read_text()
    del payload
    teacher = load_mesh_checkpoint(checkpoint, "cuda").requires_grad_(False)
    assert tensors_hash(teacher.state_dict()) == TENSORS
    calibration = pilot / "calibration.json"
    assert sha(calibration) == CALIBRATION_SHA
    source_samples = read(calibration)
    fit_ids = source_samples["trajectory_ids"]
    assert len(set(fit_ids)) == 8
    assert [(r["trajectory"], r["frame"]) for r in source_samples["samples"]] == [
        (i, f) for i in fit_ids for f in cfg["train_frames"]]
    assert read(exchange / "protocol.json")["job"] == "1623806"
    maps = exchange / "exchange.pt"
    assert sha(maps) == EXCHANGE_SHA
    graph_linear = LinearExchange().eval().cuda()
    graph_linear.load_state_dict(torch.load(maps, map_location="cpu", weights_only=True), strict=True)
    graph = variant(teacher, FIRST, LAST, graph_linear).requires_grad_(False)
    before = variant(teacher, LAST, LAST).requires_grad_(False)
    after = copy.deepcopy(before)
    after.processors[FIRST].requires_grad_(True)
    original = out / "before-model.pt"
    torch.save(before.state_dict(), original)
    original_hash = tensors_hash(before.state_dict())
    assert original_hash == tensors_hash(after.state_dict())
    fixed = frozen_hash(before, FIRST)
    before_block_hash = tensors_hash(before.processors[FIRST].state_dict())

    fit_start = time.perf_counter()
    records = CylinderFlowRecords(parent / "data/train.tfrecord", "train")
    heldout_ids = sorted(random.Random(cfg["heldout_seed"]).sample(
        [i for i in range(len(records)) if i not in fit_ids], cfg["heldout_trajectories"]))
    splits = split_manifest()
    seal(out / "protocol.json", dict(job=os.environ["LSB_JOBID"], config=cfg, splits=splits,
         train_ids=fit_ids, heldout_train_ids=heldout_ids, calibration_sha256=CALIBRATION_SHA,
         dense_sha256=CHECKPOINT, dense_tensor_sha256=TENSORS, exchange_sha256=EXCHANGE_SHA, valid_sha256=VALID_SHA,
         train_md5="046cf222b4a6632c9b3764522ece980b", source=Path("source-commit.txt").read_text().strip(),
         source_manifest_sha256=sha(Path("source.sha256")), gpu=gpu.name, host=platform.node(),
         torch=str(torch.__version__), cuda=torch.version.cuda, affinity=sorted(os.sched_getaffinity(0))))
    mesh = records.mesh(fit_ids[0]).to("cuda")
    velocity = records.velocity(fit_ids[0], cfg["train_frames"][0], 1)[0].cuda()
    with torch.inference_mode():
        assert torch.equal(before.step(mesh, velocity), after.step(mesh, velocity))
    train = capture(teacher, records, fit_ids, cfg["train_frames"], "cuda")
    heldout = capture(teacher, records, heldout_ids, cfg["train_frames"], "cuda")
    torch.cuda.synchronize()
    cache_seconds = time.perf_counter() - fit_start
    epochs = out / "features"
    epochs.mkdir()
    started = time.perf_counter()
    state, best, history = train_block(after, train, heldout, cfg, epochs)
    torch.cuda.synchronize()
    optimization_seconds = time.perf_counter() - started
    chosen = out / "distilled-block.pt"
    torch.save(state, chosen)
    after.processors[FIRST].load_state_dict(state, strict=True)
    with torch.inference_mode():
        prediction = after.step(mesh, velocity)
    restored = copy.deepcopy(before)
    restored.processors[FIRST].load_state_dict(torch.load(chosen, map_location="cpu", weights_only=True), strict=True)
    with torch.inference_mode():
        assert torch.equal(prediction, restored.step(mesh, velocity))
    assert frozen_hash(after, FIRST) == frozen_hash(restored, FIRST) == fixed
    assert tensors_hash(before.state_dict()) == original_hash
    assert tensors_hash(teacher.state_dict()) == TENSORS
    seal(out / "distillation.json", dict(chosen_epoch=best, chosen_updates=best * len(train),
         updates=cfg["epochs"] * len(train), cache_seconds=cache_seconds, optimization_seconds=optimization_seconds,
         total_seconds=time.perf_counter()-fit_start, before_features=history[0], after_features=history[best],
         last_features=history[-1], before_file_sha256=sha(original), block_file_sha256=sha(chosen),
         fixed_tensor_sha256=fixed, before_block_sha256=before_block_hash,
         after_block_sha256=tensors_hash(restored.processors[FIRST].state_dict()),
         initial_outputs_exact=True, reload_output_exact=True))
    print("distillation complete", best, flush=True)
    del records, train, heldout, after, state
    models = dict(zip(LABELS, (teacher, graph, before, restored)))
    windows = load_windows(CylinderFlowRecords(parent / "data/valid.tfrecord", "valid"),
                           splits["diagnostic"], splits["starts"], cfg["horizon"], purpose="diagnostic")
    checks = verify_processors(models, windows[0][2], windows[0][3][0])
    hashes = {k: tensors_hash(m.state_dict()) for k, m in models.items()}
    counts = {k: sum(p.numel() for p in m.parameters()) for k, m in models.items()}
    assert counts["before"] == counts["after"]
    seal(out / "gpu-checks.json", dict(processor_calls=checks, parameters=counts, state_sha256=hashes))
    for label, model in models.items():
        scientific(label, model, windows, out)
        if label in ("A", "D"):
            for ordinal, start, _, _ in windows:
                path = f"{ordinal}-{start}.json"
                current, old = (read(p / label / path) for p in (out, exchange))
                assert current["status"] == old["status"] == "ok"
                np.testing.assert_allclose(current["curve"], old["curve"],
                                           rtol=cfg["reference_curve_rtol"], atol=cfg["reference_curve_atol"])
    for label, model in models.items():
        folder = out / "reference_steps" / label
        folder.mkdir(parents=True)
        for ordinal, start, mesh, frames in windows:
            result = score_rollout(model, mesh, frames, reference_inputs=True)
            seal(folder / f"{ordinal}-{start}.json", result)
        print("reference steps", label, flush=True)
    timing = out / "timing"
    timing.mkdir()
    benchmark(models, windows, cfg, timing, orders=balanced_orders())
    assert hashes == {k: tensors_hash(m.state_dict()) for k, m in models.items()}
    seal(out / "finish.json", dict(windows=64, reference_step_windows=64, timing_samples=256, state_sha256=hashes))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "pilot", "exchange", "out"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))

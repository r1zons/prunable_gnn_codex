"""Bounded single-P100 CylinderFlow correctness check and technical timings."""

import argparse
import copy
import json
import math
import os
from pathlib import Path
import platform
import resource
import statistics
import sys
import time

import numpy as np
import torch
import torch_geometric

from gnn_pruning.config.loader import load_yaml
from gnn_pruning.data.cylinderflow import Mesh, load_trajectory
from gnn_pruning.evaluation.benchmark import model_size_metrics
from gnn_pruning.experiments.flickr_dense import sha, tensors_hash, write
from gnn_pruning.models.meshgraphnet import MeshGraphNet, load_mesh_checkpoint, rollout, save_mesh_checkpoint
from gnn_pruning.surgery.meshgraphnet import prune_processor_hidden


def compare(actual, reference, config):
    """Return measured absolute differences and enforce declared FP32 forward tolerance."""
    actual, reference = actual.detach().cpu(), reference.detach().cpu()
    torch.testing.assert_close(actual, reference, rtol=config["rtol"], atol=config["atol"])
    error = (actual - reference).abs().double()
    return {"max_abs": error.max().item(), "mean_abs": error.mean().item()}


def profile(call, warmups, repeats):
    """Measure synchronized GPU wall time and peak allocator memory."""
    for _ in range(warmups):
        call()
    torch.cuda.synchronize()
    baseline = {"resident_allocated_bytes": torch.cuda.memory_allocated(),
                "resident_reserved_bytes": torch.cuda.memory_reserved()}
    torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        call()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - start)
    return {**timing(samples), **baseline}


def timing(samples):
    """Summarize synchronized times and current peak CUDA allocator counters."""
    return {
        "n": len(samples), "median_sec": statistics.median(samples),
        "mean_sec": statistics.mean(samples), "std_sec": statistics.pstdev(samples),
        "min_sec": min(samples), "max_sec": max(samples), "samples_sec": samples,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def optimizer_timing(model, mesh, velocity, config):
    """Time Adam on two disjoint frame samples from the single staged train trajectory."""
    model.train()
    nodes = mesh.node_type.numel()
    mesh = Mesh(torch.cat((mesh.senders, mesh.senders + nodes)),
                torch.cat((mesh.receivers, mesh.receivers + nodes)),
                mesh.edge_features.repeat(2, 1), mesh.node_type.repeat(2))
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    def step(index):
        frame = 1 + (index * 2) % (velocity.shape[0] - 3)
        current = velocity[frame:frame + 2].flatten(0, 1)
        target = velocity[frame + 1:frame + 3].flatten(0, 1)
        optimizer.zero_grad(set_to_none=True)
        loss = model.training_loss(mesh, current, target, noise_std=config["noise_std"])
        loss.backward()
        optimizer.step()
        return loss.detach()

    for index in range(config["optimizer_warmups"]):
        step(index)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    samples, losses = [], []
    started = time.perf_counter()
    for index in range(config["optimizer_steps"]):
        start = time.perf_counter()
        losses.append(step(index + config["optimizer_warmups"]))
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - start)
        if time.perf_counter() - started >= config["optimizer_seconds_cap"]:
            break
    loss_values = torch.stack(losses)
    if not torch.isfinite(loss_values).all():
        raise ValueError("Nonfinite optimizer timing loss")
    return {**timing(samples), "warmups": config["optimizer_warmups"],
            "last_loss": loss_values[-1].item(), "batch": 2, "train_trajectory_count": 1}


def run(config_path: Path, inputs: Path, output: Path):
    """Verify real-data CPU/CUDA parity, structural surgery, reload and resource use."""
    config = load_yaml(config_path)
    torch.set_num_threads(config["threads"])
    torch.set_num_interop_threads(1)
    torch.manual_seed(config["seed"])
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected precisely one LSF-visible CUDA device")
    gpu = torch.cuda.get_device_properties(0)
    if (gpu.major, gpu.minor) != (6, 0) or "P100" not in gpu.name:
        raise RuntimeError(f"Expected a P100, found {gpu.name}")
    output.mkdir(parents=True, exist_ok=False)
    write(output / "config.json", config)
    manifest = json.loads((inputs / "inputs.json").read_text())
    trajectories = {}
    for split in ("train", "valid"):
        path = inputs / f"{split}-000000.npz"
        if sha(path) != manifest[split]["npz_sha256"]:
            raise ValueError(f"Unexpected {split} NPZ hash")
        sample = load_trajectory(path, split=split)
        if sample.identity != manifest[split]["record_id"]:
            raise ValueError(f"Unexpected {split} record")
        trajectories[split] = sample
    train, valid = trajectories["train"], trajectories["valid"]
    start = config["start_frame"]
    steps = config["rollout_steps"]
    mesh, initial = valid.mesh.to("cuda"), valid.velocity[start].cuda()
    train_mesh, train_velocity = train.mesh.to("cuda"), train.velocity.cuda()

    dense_cpu = MeshGraphNet().eval()
    original = tensors_hash(dense_cpu.state_dict())
    dense = copy.deepcopy(dense_cpu).cuda()
    print("Checking CPU/GPU forward and rollout", flush=True)
    with torch.no_grad():
        raw_cpu = dense_cpu(valid.mesh, valid.velocity[start])
        raw_gpu = dense(mesh, initial)
        differences = {"normalized_forward": compare(raw_gpu, raw_cpu, config),
                       "integrated_forward": compare(dense.step(mesh, initial),
                                                     dense_cpu.step(valid.mesh, valid.velocity[start]), config)}
        cpu_rollout = rollout(dense_cpu, valid.mesh, valid.velocity[start], steps)
        gpu_rollout = rollout(dense, mesh, initial, steps)
        drift = (gpu_rollout.cpu() - cpu_rollout).abs().flatten(1).amax(dim=1)
        differences["rollout_max_abs_by_step"] = drift.tolist()
        if not torch.isfinite(gpu_rollout).all():
            raise ValueError("Nonfinite GPU rollout")
        torch.testing.assert_close(gpu_rollout[:, ~mesh.active],
                                   initial[~mesh.active].expand(steps + 1, -1, -1), rtol=0, atol=0)

    print("Checking one train-only loss/backward", flush=True)
    training_copy = copy.deepcopy(dense).train()
    torch.cuda.reset_peak_memory_stats()
    train_loss = training_copy.training_loss(train_mesh, train_velocity[start],
                                            train_velocity[start + 1], noise_std=config["noise_std"])
    train_loss.backward()
    grads = [p.grad for p in training_copy.parameters()]
    if not math.isfinite(train_loss.item()) or not all(
        grad is not None and torch.isfinite(grad).all() for grad in grads
    ):
        raise ValueError("Nonfinite loss or incomplete gradients")
    backward = {"loss": train_loss.item(), "all_gradients_finite": True,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
    del grads, train_loss, training_copy

    print("Checking physical surgery and checkpoint reload", flush=True)
    removed = config["remove_channels"]
    block, kind = config["processor"], config["mlp"]
    width = getattr(dense.processors[block], kind)[2].out_features
    keep = list(range(width - removed))
    pruned = prune_processor_hidden(dense, block, kind, keep).eval()
    original_mlp = getattr(dense.processors[block], kind)
    compressed_mlp = getattr(pruned.processors[block], kind)
    for actual, expected in ((compressed_mlp[2].weight, original_mlp[2].weight[keep]),
                             (compressed_mlp[2].bias, original_mlp[2].bias[keep]),
                             (compressed_mlp[4].weight, original_mlp[4].weight[:, keep])):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    mask = torch.arange(width, device="cuda") < width - removed
    with torch.no_grad():
        handle = original_mlp[3].register_forward_hook(lambda _m, _x, value: value * mask)
        try:
            masked = dense(mesh, initial)
        finally:
            handle.remove()
        differences["physical_vs_masked"] = compare(pruned(mesh, initial), masked, config)
        differences["pruned_cpu_vs_gpu"] = compare(
            pruned(mesh, initial),
            prune_processor_hidden(dense_cpu, block, kind, keep)(valid.mesh, valid.velocity[start]), config,
        )
    measured = {}
    for label, model in (("dense", dense), ("pruned", pruned)):
        path = save_mesh_checkpoint(model, output / f"{label}.pt")
        restored = load_mesh_checkpoint(path, "cuda")
        identity = tensors_hash(model.state_dict())
        if restored.widths() != model.widths() or tensors_hash(restored.state_dict()) != identity:
            raise ValueError(f"Strict {label} reload changed state")
        with torch.no_grad():
            differences[f"{label}_reload"] = compare(restored(mesh, initial), model(mesh, initial), config)
        del restored
        torch.cuda.empty_cache()
        with torch.no_grad():
            forward = profile(lambda: model.step(mesh, initial), config["warmups"], config["repetitions"])
            trajectory = profile(lambda: rollout(model, mesh, initial, steps),
                                 config["warmups"], config["repetitions"])
        measured[label] = {**model_size_metrics(model, path), "sha256": sha(path),
                           "tensor_sha256": identity, "widths": model.widths(),
                           "forward": forward, "rollout": trajectory}
    if measured["dense"]["parameter_count"] != 2_332_930:
        raise ValueError("Unexpected dense parameter count")
    if measured["dense"]["parameter_count"] - measured["pruned"]["parameter_count"] != 257 * removed:
        raise ValueError("Unexpected physical parameter reduction")
    if tensors_hash(dense.state_dict()) != original or tensors_hash(dense_cpu.state_dict()) != original:
        raise ValueError("Dense state changed in eval or surgery")

    print("Timing the bounded train-only optimizer workload", flush=True)
    torch.cuda.empty_cache()
    optimizer = optimizer_timing(copy.deepcopy(dense), train_mesh, train_velocity, config)
    report = {
        "status": "PASS", "technical_only": True, "initialization": "fixed random",
        "job_id": os.environ.get("LSB_JOBID"), "host": platform.node(),
        "architecture": platform.machine(), "python": sys.version, "torch": str(torch.__version__),
        "cuda": torch.version.cuda, "pyg": torch_geometric.__version__, "numpy": np.__version__,
        "gpu": gpu.name, "gpu_total_bytes": gpu.total_memory, "gpu_capability": [gpu.major, gpu.minor],
        "cpu_affinity": sorted(os.sched_getaffinity(0)), "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(), "config": config,
        "inputs": {split: {**manifest[split], "nodes": sample.mesh.node_type.numel(),
                           "directed_edges": sample.mesh.senders.numel(),
                           "active_nodes": sample.mesh.active.sum().item()}
                   for split, sample in trajectories.items()},
        "differences": differences, "backward": backward, "models": measured,
        "retained_channel_ids": keep, "optimizer": optimizer,
        "peak_host_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    write(output / "results.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.inputs, args.output)

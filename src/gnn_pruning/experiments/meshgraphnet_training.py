"""Single-P100 training with bounded wall time and resumable CylinderFlow state."""

import argparse
import json
import math
import os
from pathlib import Path
import random
import signal
import time

import torch

from gnn_pruning.config.loader import load_yaml
from gnn_pruning.data.cylinderflow import Mesh
from gnn_pruning.data.cylinderflow_records import CylinderFlowRecords
from gnn_pruning.models.meshgraphnet import CHECKPOINT_FORMAT, MeshGraphNet


def atomic_save(payload, path: Path):
    """Replace the published checkpoint only after the new file is fully written."""
    temporary = path.with_suffix(".partial")
    with temporary.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def train_batch(records, sampler, device):
    """Sample two distinct official train trajectories and batch disjoint graphs."""
    first, second = sampler.sample(range(len(records)), 2)
    meshes, velocities = [], []
    for ordinal in (first, second):
        meshes.append(records.mesh(ordinal))
        velocities.append(records.velocity(ordinal, sampler.randrange(1, 599), 2))
    a, b = meshes
    offset = a.node_type.numel()
    mesh = Mesh(torch.cat((a.senders, b.senders + offset)),
                torch.cat((a.receivers, b.receivers + offset)),
                torch.cat((a.edge_features, b.edge_features)),
                torch.cat((a.node_type, b.node_type)))
    current, target = torch.cat(velocities, dim=1).to(device)
    return mesh.to(device), current, target


@torch.no_grad()
def validate(model, records, ids, starts, horizon, device, expired):
    """Compute per-window velocity MSE with autoregressive feedback and fixed boundaries."""
    model.eval()
    errors, persistence = [], []
    for ordinal in ids:
        mesh = records.mesh(ordinal).to(device)
        active = mesh.active
        for start in starts:
            if expired():
                return None
            frames = records.velocity(ordinal, start, horizon + 1).to(device)
            velocity = frames[0]
            window_errors = []
            for reference in frames[1:]:
                velocity = model.step(mesh, velocity)
                window_errors.append((velocity[active] - reference[active]).square().mean())
            curve = torch.stack(window_errors).cpu().double()
            errors.append(curve)
            persistence.append((frames[0, active] - frames[-1, active]).square().mean().item())
    if expired():
        return None
    curves = torch.stack(errors)
    stable = torch.isfinite(curves).all(dim=1)
    result = {"windows": len(errors), "finite_windows": stable.sum().item(),
              "persistence_e100": sum(persistence) / len(persistence)}
    result.update({"e1": None, "e20": None, "e100": None, "mean_mse": None})
    if stable.all():
        curve = curves.mean(dim=0)
        result.update(e1=curve[0].item(), e20=curve[19].item(), e100=curve[-1].item(),
                      mean_mse=curve.mean().item())
    return result


def run(config, data: Path, output: Path, elapsed_at_start: float, resume: Path = None):
    """Train until the update cap or the internal cutoff, and save the last safe state."""
    clock = time.monotonic
    started = clock() - max(0.0, elapsed_at_start)
    cutoff = started + config["wall_seconds"] - config["shutdown_reserve_seconds"]
    stop_signal = None

    def request_stop(signum, _frame):
        nonlocal stop_signal
        stop_signal = signal.Signals(signum).name

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, request_stop)

    torch.set_num_threads(config["cpu_threads"])
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected one LSF-visible CUDA device")
    gpu = torch.cuda.get_device_properties(0)
    if (gpu.major, gpu.minor) != (6, 0) or "P100" not in gpu.name:
        raise RuntimeError(f"Expected P100, found {gpu.name}")
    device = torch.device("cuda:0")
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    dataset_identity = (data / "VERIFIED.md5").read_text()
    train = CylinderFlowRecords(data / "train.tfrecord", "train")
    valid = CylinderFlowRecords(data / "valid.tfrecord", "valid")
    validation_ids = list(range(len(valid)))
    random.Random(config["validation_seed"]).shuffle(validation_ids)
    validation_ids = validation_ids[:config["validation_trajectories"]]
    random.seed(config["seed"])
    sampler = random.Random(config["seed"])
    torch.manual_seed(config["seed"])
    model = MeshGraphNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["adam_initial_lr"])
    updates = warmups = 0
    last_validation = best = None
    validation_budget = config["initial_validation_budget_seconds"]

    if resume is not None:
        state = torch.load(resume, map_location="cpu", weights_only=True)
        if (state["format"] != "meshgraphnet-training-v1" or state["config"] != config
                or state["dataset_identity"] != dataset_identity
                or state["validation_ids"] != validation_ids):
            raise ValueError("Incompatible resume state")
        model.load_state_dict(state["state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        updates, warmups = state["updates"], state["warmups"]
        last_validation, best = state["last_validation"], state["best"]
        sampler.setstate(state["sampler"])
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"])
        validation_budget = state["validation_budget"]

    last_save = clock()
    log_path = output / "events.jsonl"

    def log(event, **data):
        row = {"event": event, "updates": updates, "warmups": warmups,
               "elapsed_seconds": clock() - started, **data}
        message = json.dumps(row, allow_nan=False)
        with log_path.open("a") as stream:
            stream.write(message + "\n")
        print(message, flush=True)

    def save(reason):
        nonlocal last_save
        atomic_save({
            "format": "meshgraphnet-training-v1", "widths": model.widths(),
            "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
            "updates": updates, "warmups": warmups, "sampler": sampler.getstate(),
            "python_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(), "last_validation": last_validation,
            "best": best, "validation_budget": validation_budget,
            "config": config, "dataset_identity": dataset_identity,
            "validation_ids": validation_ids,
        }, output / "latest.pt")
        last_save = clock()
        log("checkpoint", reason=reason)

    def expired():
        return stop_signal is not None or clock() >= cutoff

    log("start", torch=str(torch.__version__), gpu=gpu.name, validation_ids=validation_ids,
        cpu_affinity=sorted(os.sched_getaffinity(0)))
    save("initial")
    reason = "update_limit"
    last_log = clock()
    while True:
        if expired():
            reason = stop_signal or "wall_time"
            break
        if clock() - last_save >= config["checkpoint_every_seconds"]:
            save("periodic")
        if warmups < config["normalization_batches"]:
            model.train()
            with torch.no_grad():
                loss = model.training_loss(*train_batch(train, sampler, device),
                                           noise_std=config["noise_std"])
            if not math.isfinite(loss.item()):
                save("nonfinite_warmup")
                raise RuntimeError("Nonfinite normalization loss")
            warmups += 1
            continue
        if updates % config["validation_every_updates"] == 0 and last_validation != updates:
            if clock() + validation_budget - last_save >= config["checkpoint_every_seconds"]:
                save("before_validation")
            if clock() + validation_budget >= cutoff:
                log("validation_skipped", reason="insufficient_wall_time",
                    budget_seconds=validation_budget, remaining_seconds=cutoff - clock())
                reason = "wall_time"
                break
            validation_start = clock()

            def validation_expired():
                if expired():
                    return True
                if clock() - last_save >= config["checkpoint_every_seconds"]:
                    save("during_validation")
                return expired()

            metrics = validate(model, valid, validation_ids, config["validation_starts"],
                               config["validation_horizon"], device, validation_expired)
            if metrics is None:
                log("validation_interrupted")
                reason = stop_signal or "wall_time"
                break
            elapsed = clock() - validation_start
            validation_budget = max(validation_budget, elapsed * 1.5 + 15)
            last_validation = updates
            log("validation", seconds=elapsed, **metrics)
            if metrics["e100"] is not None and (best is None or metrics["e100"] < best["e100"]):
                best = {"updates": updates, "e100": metrics["e100"]}
                atomic_save({"format": CHECKPOINT_FORMAT, "widths": model.widths(),
                             "state_dict": model.state_dict(), "validation": metrics,
                             "updates": updates, "validation_ids": validation_ids,
                             "dataset_identity": dataset_identity}, output / "best.pt")
            save("after_validation")
            continue
        if updates >= config["max_updates"]:
            break
        model.train()
        initial, floor = config["adam_initial_lr"], config["adam_lr_floor"]
        lr = floor + (initial - floor) * config["decay_rate"] ** (updates / config["decay_updates"])
        optimizer.param_groups[0]["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        loss = model.training_loss(*train_batch(train, sampler, device), noise_std=config["noise_std"])
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            log("nonfinite_loss")
            raise RuntimeError("Nonfinite optimizer loss; last valid checkpoint preserved")
        loss.backward()
        optimizer.step()
        updates += 1
        if updates % 100 == 0:
            torch.cuda.synchronize()
            now = clock()
            log("train", loss=loss_value, lr=lr, seconds_since_log=now - last_log,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved())
            last_log = now
    save(reason)
    log("finish", reason=reason, best=best)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--job-start", type=float, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    run(load_yaml(args.config), args.data, args.output, time.time() - args.job_start, args.resume)

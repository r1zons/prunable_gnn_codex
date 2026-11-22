"""Location sweep, independent GPU stage profile, then two-frame velocity correction."""
from contextlib import ExitStack
from pathlib import Path
import os
import random
import statistics
import sys
import time

import numpy as np
import torch
import yaml

import run_cylinderflow_block_distillation as dist
import run_cylinderflow_linear_block_pilot as pilot
import run_cylinderflow_linear_exchange as exchange
from gnn_pruning.data.cylinderflow_records import CylinderFlowRecords
from gnn_pruning.experiments.flickr_dense import sha, tensors_hash
from gnn_pruning.models.meshgraphnet import load_mesh_checkpoint
from run_cylinderflow_mask_diagnostic import seal, split_manifest

CONFIG = Path("configs/experiments/meshgraphnet_cylinderflow_positions.yaml")


def load_block(block, path):
    block.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return block


def make_models(root, train, dense, cfg, out):
    old = root / "meshgraphnet-linear-block-pilot-20261120/results"
    extended = root / "meshgraphnet-block-distillation-extended-20261121/results"
    graph = root / "meshgraphnet-linear-exchange-20261120/results/exchange.pt"
    assert sha(old / "calibration.json") == dist.CALIBRATION_SHA
    assert sha(old / "maps.pt") == "ce5498b9e5bfef954e56e5869280a2bd19778f0a53993ed0dfe233acc82d004d"
    assert sha(graph) == dist.EXCHANGE_SHA
    assert sha(extended / "best-block.pt") == "917ad37b2ce247f5cf3f985445d04099f17e8b1f7743edea675c55e7a071fbd0"
    reference = dist.read(old / "calibration.json")
    protocol = dist.read(extended / "protocol.json")
    ridge = yaml.safe_load(pilot.CONFIG.read_text())
    fit_cfg = dict(protocol["config"], epochs=cfg["epochs"])
    models = {"dense": dense}
    for left, right in cfg["pairs"]:
        first, last = left - 1, right - 1
        tag = f"{left}-{right}"
        retained = pilot.variant(dense, last, last).requires_grad_(False)
        student = pilot.variant(dense, last, last).requires_grad_(False)
        if (first, last) == (dist.FIRST, dist.LAST):
            affine = load_block(pilot.AffinePair(), old / "maps.pt")
            message = load_block(exchange.LinearExchange(), graph)
            load_block(student.processors[first], extended / "best-block.pt")
            best = dist.read(extended / "finish.json")["best_heldout_epoch"]
        else:
            affine, sample = pilot.calibrate(dense, train, ridge, first, last, "cuda")
            assert sample["samples"] == reference["samples"]
            message, _ = exchange.calibrate(dense, train, sample["samples"], first, last, ridge["ridge_relative"])
            directory = out / f"features-{tag}"
            directory.mkdir()
            fit, heldout = [dist.capture(dense, train, protocol[key], fit_cfg["train_frames"], "cuda", first, last)
                            for key in ("train_ids", "heldout_train_ids")]
            student.processors[first].requires_grad_(True)
            state, best, _ = dist.train_block(student, fit, heldout, fit_cfg, directory, block_index=first)
            student.processors[first].load_state_dict(state)
            torch.save(state, out / f"distilled-{tag}.pt")
            torch.save(affine.state_dict(), out / f"affine-{tag}.pt")
            torch.save(message.state_dict(), out / f"exchange-{tag}.pt")
            del fit, heldout, state
        models.update({tag+"/skip": pilot.variant(dense, first, last),
                       tag+"/affine": pilot.variant(dense, first, last, affine.cuda()),
                       tag+"/exchange": pilot.variant(dense, first, last, message.cuda()),
                       tag+"/remove-one": retained, tag+"/distilled": student.requires_grad_(False)})
        seal(out / f"fit-{tag}.json", dict(best_epoch=best, selection="heldout_feature_mse"))
        print("Models ready", tag, flush=True)
    return models, reference


def init_windows(valid, ids, starts, horizon):
    return [(idx, start, valid.mesh(idx).to("cuda"), valid.velocity(idx, start-1, horizon+2).cuda())
            for idx in ids for start in starts]


@torch.inference_mode()
def forecast(model, mesh, frames, horizon, correction=None, *, collect=False):
    previous, current = frames[0], frames[1]
    preds = []
    for _ in range(horizon):
        following = model.step(mesh, current)
        if correction is not None:
            scaled = correction[0] * (following - current)
            if len(correction) == 2:
                scaled = scaled + correction[1] * (current - previous)
            following = torch.where(mesh.active[:, None], current + scaled, current)
        previous, current = current, following
        if collect:
            preds.append(current)
    return torch.stack(preds) if collect else current


@torch.inference_mode()
def evaluate(models, corrections, windows, cfg, out):
    rows = []
    rng = random.Random(cfg["seed"])
    for idx, start, mesh, frames in windows:
        print("Evaluation", out.name, idx, start, flush=True)
        window = {}
        for key, model in models.items():
            corr = corrections.get(key)
            pred = forecast(model, mesh, frames, cfg["horizon"], corr, collect=True)
            assert torch.equal(pred[:, ~mesh.active], frames[1, ~mesh.active].expand_as(pred[:, ~mesh.active]))
            mse = (pred[:, mesh.active] - frames[2:, mesh.active]).square().mean((1, 2)).tolist()
            window[key] = dict(curve=[x if np.isfinite(x) else None for x in mse], seconds=[])
        # All quality passes warm the models. First start per trajectory gets dedicated timed repeats.
        if start == windows[0][1]:
            order = list(models)
            rng.shuffle(order)
            for rep in range(cfg["timing_repeats"]):
                if rep % 2 == 0 and rep:
                    rng.shuffle(order)
                for key in (order if rep % 2 == 0 else reversed(order)):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    forecast(models[key], mesh, frames, cfg["horizon"], corrections.get(key))
                    torch.cuda.synchronize()
                    window[key]["seconds"].append(time.perf_counter()-t0)
        seal(out / f"{idx}-{start}.json", window)
        rows.append(window)
    summary = {}
    for key in models:
        curves = [r[key]["curve"] for r in rows]
        complete = sum(None not in c for c in curves)
        latency = [x for r in rows for x in r[key]["seconds"]]
        summary[key] = dict(windows=complete, total=len(rows),
                            E1=statistics.mean(c[0] for c in curves) if complete == len(rows) else None,
                            E20=statistics.mean(c[-1] for c in curves) if complete == len(rows) else None,
                            mean20=statistics.mean(map(statistics.mean, curves)) if complete == len(rows) else None,
                            T20=statistics.median(latency), samples=len(latency))
    seal(out / "summary.json", summary)


@torch.inference_mode()
def profile(dense, windows, cfg, out):
    # Consecutive CUDA events on the normal execution stream; no synchronization between stages.
    profiles = []
    for idx, start, mesh, frames in windows:
        if start != windows[0][1]:
            continue
        pilot.steps(dense, mesh, frames[1], cfg["warmup_steps"])
        def event():
            stamp = torch.cuda.Event(enable_timing=True)
            stamp.record()
            return stamp
        marks = []
        def after(name):
            def hook(_module, _args, _output):
                marks.append((name, event()))
            return hook
        with ExitStack() as hooks:
            for name, module in [("node_input_and_encoder", dense.node_encoder), ("edge_input_and_encoder", dense.edge_encoder),
                                 *((f"block_{i}", b) for i, b in enumerate(dense.processors, 1)), ("decoder", dense.decoder)]:
                hooks.callback(module.register_forward_hook(after(name)).remove)
            velocity = frames[1]
            for _ in range(cfg["profile_steps"]):
                first = event()
                velocity = dense.step(mesh, velocity)
                marks.append(("integration_and_boundary", event()))
                torch.cuda.synchronize()
                row = {}
                for name, stop in marks:
                    row[name] = first.elapsed_time(stop)
                    first = stop
                profiles.append(dict(trajectory=idx, ms=row))
                marks.clear()
    seal(out / "profile.json", profiles)


@torch.inference_mode()
def coefficients(student, train, samples):
    x, y = [], []
    for item in samples:
        idx, t = item["trajectory"], item["frame"]
        mesh = train.mesh(idx).to("cuda")
        previous, current, next_field = train.velocity(idx, t-1, 3).cuda()
        pred = student.step(mesh, current)
        active = mesh.active
        x.append(torch.stack(((pred-current)[active], (current-previous)[active]), dim=-1).reshape(-1, 2).cpu().numpy())
        y.append((next_field-current)[active].flatten().cpu().numpy())
    X, Y = np.concatenate(x).astype("float64"), np.concatenate(y).astype("float64")
    return {key: np.linalg.lstsq(X[:, :count], Y, rcond=None)[0].tolist()
            for key, count in (("scale", 1), ("history", 2))}


def run(root, out):
    cfg = yaml.safe_load(CONFIG.read_text())
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    assert torch.cuda.device_count() == 1 and "P100" in torch.cuda.get_device_name(0)
    parent = root / "meshgraphnet-cylinderflow-training-20261113"
    checkpoint = parent / pilot.CHECKPOINT_PATH
    assert sha(checkpoint) == pilot.CHECKPOINT
    dense = load_mesh_checkpoint(checkpoint, "cuda").requires_grad_(False)
    assert tensors_hash(dense.state_dict()) == pilot.TENSORS
    out.mkdir()
    splits = split_manifest()
    remaining = sorted(set(range(100))-set(splits["baseline"]+splits["diagnostic"]+splits["selection"]))
    fresh = random.Random(cfg["seed"]).sample(remaining, cfg["confirmation_trajectories"])
    seal(out/"protocol.json", dict(config=cfg, splits=splits, confirmation=fresh, job=os.environ["LSB_JOBID"],
                                  checkpoint=pilot.CHECKPOINT, source=Path("source-commit.txt").read_text().strip()))
    train = CylinderFlowRecords(parent / "data/train.tfrecord", "train")
    valid = CylinderFlowRecords(parent / "data/valid.tfrecord", "valid")
    models, calibration = make_models(root, train, dense, cfg, out)
    windows = init_windows(valid, splits["diagnostic"], splits["starts"], cfg["horizon"])
    sweep = out/"locations"
    sweep.mkdir()
    evaluate(models, {}, windows, cfg, sweep)
    profile(dense, windows, cfg, out)
    print("Sweep and stage profile complete; starting history test", flush=True)
    student = models["7-8/distilled"]
    corrections = coefficients(student, train, calibration["samples"])
    seal(out/"coefficients.json", corrections)
    confirmation = init_windows(valid, fresh, splits["starts"], cfg["horizon"])
    hist = out/"history"
    hist.mkdir()
    evaluate(dict(dense=dense, student=student, scale=student, history=student), corrections, confirmation, cfg, hist)
    seal(out/"finish.json", dict(locations=len(models), history=4, diagnostic_windows=len(windows),
                                confirmation_windows=len(confirmation)))


if __name__ == "__main__":
    run(*map(Path, sys.argv[1:]))

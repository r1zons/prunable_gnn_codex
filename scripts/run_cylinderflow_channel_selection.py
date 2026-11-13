"""Run the frozen CylinderFlow channel-selection comparison on one P100."""

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
import socket
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from gnn_pruning.data.cylinderflow_records import CylinderFlowRecords
from gnn_pruning.experiments.flickr_dense import sha, tensors_hash
from gnn_pruning.models.meshgraphnet import LATENT, load_mesh_checkpoint, save_mesh_checkpoint
from gnn_pruning.surgery.meshgraphnet import prune_processor_hidden_layers
from run_cylinderflow_mask_diagnostic import (
    BASELINE, CHECKPOINT, LAYERS, TENSORS, masks, parity, seal, split_manifest,
)


DENSE_COUNT = 2_332_930
VALID_SHA = "79171fa282fcd4fd5ff93d13e04ade43e83c1b11a0b96eee0951e4e0ed37f3ca"
CHECKPOINT_PATH = "continuation-02/run.1618479/best.pt"
CONFIG = Path("configs/experiments/meshgraphnet_cylinderflow_channel_selection.yaml")


def library(ranks, parent, cfg):
    """Draw an ordered unique library using only the fixed ranks and parent IDs."""
    identity = json.dumps([cfg["seed"], parent], separators=(",", ":")).encode()
    rng = random.Random(int.from_bytes(hashlib.sha256(identity).digest(), "big"))
    available = [tuple(c for c in rank if c not in removed)
                 for rank, removed in zip(ranks, parent)]
    size = cfg["removals_per_decision"]
    candidates = [tuple(tuple(sorted(pool[:size])) for pool in available)]
    for group, pools in (
        (cfg["low_draws"], [pool[:cfg["low_pool"]] for pool in available]),
        (cfg["global_draws"], available),
    ):
        for _ in range(group):
            for _ in range(cfg["duplicate_retries"]):
                draw = tuple(tuple(sorted(rng.sample(pool, size))) for pool in pools)
                if draw not in candidates:
                    candidates.append(draw)
                    break
            else:
                raise RuntimeError("Could not draw a unique candidate")
    return candidates


def extend(parent, action):
    """Build the cumulative mask in original channel IDs."""
    return tuple(tuple(sorted(old + new)) for old, new in zip(parent, action))


def choose(candidates, scores, atol=0, rtol=0):
    """Break approximate ties relative to the exact minimum by original IDs."""
    valid = [(index, score) for index, score in enumerate(scores)
             if score is not None and math.isfinite(score)]
    if not valid:
        return None
    best = min(score for _, score in valid)
    return min((i for i, score in valid if score - best <= atol + rtol * max(abs(score), abs(best))),
               key=candidates.__getitem__)


def project(dense, removed):
    """Rebuild from original IDs, equivalent to sequentially slicing the parent."""
    if (len(removed) != len(LAYERS) or len({len(ids) for ids in removed}) != 1
            or any(len(set(ids)) != len(ids) or not set(ids) <= set(range(LATENT))
                   for ids in removed)):
        raise ValueError("Expected one uniformly sized mask per processor MLP")
    model = prune_processor_hidden_layers(dense, {
        layer: sorted(set(range(LATENT)).difference(ids)) for layer, ids in zip(LAYERS, removed)
    })
    count = sum(p.numel() for p in model.parameters())
    width = LATENT - len(removed[0])
    assert model.widths() == [[width, width]] * 15
    assert count == DENSE_COUNT - len(LAYERS) * (2 * LATENT + 1) * len(removed[0])
    return model, count


@torch.inference_mode()
def score_rollout(model, mesh, frames):
    """Record physical-unit MSE, actual forward calls and the first failed step."""
    velocity = frames[0]
    predictions = []
    error = None
    calls = 0
    for _ in range(len(frames) - 1):
        try:
            if model is not None:
                calls += 1
                velocity = model.step(mesh, velocity)
            predictions.append(velocity)
        except RuntimeError as exc:
            error = str(exc)
            break
    status = "runtime_error" if error else "ok"
    curve = []
    if predictions:
        fields = torch.stack(predictions)
        active = mesh.active
        assert torch.equal(fields[:, ~active], frames[0, ~active].expand_as(fields[:, ~active]))
        mse = (fields[:, active] - frames[1:1 + len(fields), active]).square().mean(dim=(1, 2))
        finite = torch.isfinite(fields).flatten(1).all(1) & torch.isfinite(mse)
        good = next((i for i, ok in enumerate(finite.tolist()) if not ok), len(fields))
        curve = mse[:good].tolist()
        if good < len(fields):
            status = "nonfinite"
    return dict(status=status, curve=curve, error=error, forward_calls=calls,
                first_failed_step=None if status == "ok" else len(curve) + 1)


def score_candidate(model, windows, horizon):
    """Evaluate only selection windows, giving trajectories equal weight."""
    rows = [dict(ordinal=ordinal, **score_rollout(model, mesh, frames[:horizon + 1]))
            for ordinal, mesh, frames in windows]
    score = None
    if all(row["status"] == "ok" for row in rows):
        score = statistics.mean(statistics.mean(row["curve"]) for row in rows)
    return score, rows


def load_windows(records, ids, starts, horizon, *, purpose):
    """Load specified validation windows onto the GPU, preserving original IDs."""
    splits = split_manifest()
    if records.split != "valid" or ids != splits[purpose]:
        raise ValueError("Only designated development validation trajectories may be used")
    return [(ordinal, start, records.mesh(ordinal).to("cuda"),
             records.velocity(ordinal, start, horizon + 1).cuda())
            for ordinal in ids for start in starts]


@torch.inference_mode()
def timing(model, mesh, initial, cfg):
    """Synchronize each single-step sample after warmup and record allocator peaks."""
    for _ in range(cfg["timing_warmup"]):
        model.step(mesh, initial)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    allocated_at_start = torch.cuda.memory_allocated()
    samples = []
    for _ in range(cfg["timing_repeats"]):
        start = time.perf_counter()
        model.step(mesh, initial)
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - start)
    return dict(seconds=samples, warmup_calls=cfg["timing_warmup"], timed_calls=len(samples),
                mean_seconds=statistics.mean(samples),
                median_seconds=statistics.median(samples), sd_seconds=statistics.stdev(samples),
                allocated_at_start_bytes=allocated_at_start,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved())


def snapshot(dense, removed, folder, sample, decision):
    """Verify a frozen structural checkpoint before publishing it."""
    folder.mkdir(exist_ok=True)
    path = folder / "physical.pt"
    model, count = project(dense, removed)
    identity = tensors_hash(model.state_dict())
    if not path.exists():
        tmp = folder / "physical.partial"
        save_mesh_checkpoint(model, tmp)
        restored = load_mesh_checkpoint(tmp, "cuda")
    else:
        restored = load_mesh_checkpoint(path, "cuda")
    assert restored.widths() == model.widths()
    assert tensors_hash(restored.state_dict()) == identity
    assert tensors_hash(dict(restored.named_buffers())) == tensors_hash(dict(dense.named_buffers()))
    _, mesh, frames = sample
    physical_error = parity(copy.deepcopy(dense).cuda(), restored,
                            [list(ids) for ids in removed], mesh, frames[0])
    if not path.exists():
        with tmp.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    return dict(decision=decision, removed=removed, parameters=count, widths=model.widths(),
                compression=1 - count / DENSE_COUNT, checkpoint_sha256=sha(path),
                tensor_sha256=identity, masked_max_abs=physical_error, verification_forward_calls=2)


def search(dense, ranks, selected, output, cfg, may_start):
    """Resume only from sealed decisions; keep diagnostic inputs outside search."""
    inverse = [{channel: index for index, channel in enumerate(rank)} for rank in ranks]
    strategies = {}
    for strategy, horizon in cfg["selection_horizons"].items():
        folder = output / strategy
        folder.mkdir(exist_ok=True)
        parent = ((),) * len(LAYERS)
        strategies[strategy] = []
        for decision in range(1, cfg["decisions"] + 1):
            path = folder / f"decision-{decision:02}.json"
            candidates = library(ranks, parent, cfg)
            if path.exists():
                row = json.loads(path.read_text())
                assert row["parent"] == [list(ids) for ids in parent]
                assert row["candidates"] == [[list(ids) for ids in action] for action in candidates]
                winner = row["winner"]
                assert winner == choose(candidates, row["scores"], **tie_kwargs(horizon, cfg))
            else:
                if not may_start("decision"):
                    return strategies, False
                start = time.monotonic()
                scores, evidence = [], []
                for action in candidates:
                    if horizon:
                        model, _ = project(dense, extend(parent, action))
                        model.cuda()
                        score, rows = score_candidate(model, selected, horizon)
                        del model
                    else:
                        score = sum(inv[channel] for inv, ids in zip(inverse, action) for channel in ids)
                        rows = []
                    scores.append(score)
                    evidence.append(rows)
                winner = choose(candidates, scores, **tie_kwargs(horizon, cfg))
                row = dict(parent=parent, candidates=candidates, scores=scores, windows=evidence,
                           winner=winner, search_seconds=time.monotonic() - start,
                           candidate_evaluations=len(candidates),
                           model_candidate_evaluations=len(candidates) if horizon else 0,
                           forward_calls=sum(win["forward_calls"] for rows in evidence for win in rows))
                seal(path, row)
                print(strategy, decision, "winner", winner, "calls", row["forward_calls"], flush=True)
            if winner is None:
                break
            parent = extend(parent, candidates[winner])
            if decision in cfg["snapshot_decisions"]:
                name = f"{strategy}-{len(parent[0]):02}"
                subfolder = output / name
                record = subfolder / "surgery.json"
                if record.exists():
                    info = json.loads(record.read_text())
                    assert info["removed"] == [list(ids) for ids in parent]
                    assert sha(subfolder / "physical.pt") == info["checkpoint_sha256"]
                else:
                    # A job can stop after the decision but before checkpoint publication.
                    if not may_start("snapshot"):
                        return strategies, False
                    info = snapshot(dense, parent, subfolder, selected[0], decision)
                    seal(record, info)
                strategies[strategy].append(name)
    return strategies, True


def tie_kwargs(horizon, cfg):
    return dict(atol=cfg["tie_atol"], rtol=cfg["tie_rtol"]) if horizon else {}


def evaluate(dense, strategies, records, ids, output, cfg, may_start):
    """Use diagnostic frames only after the entire search is sealed."""
    windows = load_windows(records, ids, cfg["diagnostic_starts"], cfg["diagnostic_horizon"],
                           purpose="diagnostic")
    windows.sort(key=lambda item: (-item[2].senders.numel(), item[0], item[1]))
    names = ["dense", "persistence", *(name for group in strategies.values() for name in group)]
    for name in names:
        folder = output / name
        folder.mkdir(exist_ok=True)
        pending = [row for row in windows if not (folder / f"{row[0]}-{row[1]}.json").exists()]
        time_path = folder / "timing.json"
        if not pending and (name == "persistence" or time_path.exists()):
            continue
        if not may_start("window"):
            return False
        torch.cuda.empty_cache()
        base_allocated = torch.cuda.memory_allocated()
        if name == "dense":
            model = copy.deepcopy(dense).cuda()
            expected_hash = TENSORS
        elif name == "persistence":
            model = None
        else:
            path = folder / "physical.pt"
            info = json.loads((folder / "surgery.json").read_text())
            assert sha(path) == info["checkpoint_sha256"]
            model = load_mesh_checkpoint(path, "cuda")
            expected_hash = info["tensor_sha256"]
            assert tensors_hash(model.state_dict()) == expected_hash
        if model is not None and not time_path.exists():
            ordinal, start, mesh, frames = windows[0]
            seal(time_path, dict(ordinal=ordinal, start=start, baseline_allocated_bytes=base_allocated,
                                 host=socket.gethostname(), **timing(model, mesh, frames[0], cfg)))
        for ordinal, start, mesh, frames in pending:
            if not may_start("window"):
                return False
            began = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            result = score_rollout(model, mesh, frames)
            seal(folder / f"{ordinal}-{start}.json", dict(
                elapsed_seconds=time.monotonic() - began,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(), **result))
        if model is not None:
            assert tensors_hash(model.state_dict()) == expected_hash
        del model
        print(f"{name}: {len(pending)} new diagnostic windows", flush=True)
    return True


def calibrate(dense, ranks, selected, cfg, output):
    """Measure one representative five-step candidate and one longest rollout."""
    action = library(ranks, ((),) * len(LAYERS), cfg)[0]
    started = time.monotonic()
    model, _ = project(dense, action)
    model.cuda()
    score, rows = score_candidate(model, selected, 5)
    search_s = time.monotonic() - started
    # Reconstruct independently to exercise the same path an interrupted decision uses.
    del model
    model, _ = project(dense, action)
    repeated_score, repeated_rows = score_candidate(model.cuda(), selected, 5)
    if score is None or (score, rows) != (repeated_score, repeated_rows):
        raise RuntimeError("CUDA selection scores failed finite/repeatability gate")
    del model
    _, mesh, frames = max(selected, key=lambda row: row[1].senders.numel())
    output.mkdir()
    probe = output
    started = time.monotonic()
    snap = snapshot(dense, action, probe, (None, mesh, frames), 1)
    snapshot_s = time.monotonic() - started
    restored = load_mesh_checkpoint(probe / "physical.pt", "cuda")
    started = time.monotonic()
    measured = timing(restored, mesh, frames[0], cfg)
    timing_s = time.monotonic() - started
    # Time an actual 100-step autoregressive rollout on the selection mesh.
    # The repeated reference here only exercises the timing path; the scores are discarded.
    reference = frames[0].expand(cfg["diagnostic_horizon"] + 1, -1, -1)
    started = time.monotonic()
    result = score_rollout(restored, mesh, reference)
    window_s = time.monotonic() - started
    if result["status"] != "ok":
        raise RuntimeError(f"GPU rollout timing probe failed: {result}")
    candidates = 1 + cfg["low_draws"] + cfg["global_draws"]
    snapshots = len(cfg["snapshot_decisions"]) * len(cfg["selection_horizons"])
    # The five-step candidate cost bounds the one-step candidate cost.
    forecast = 2 * (cfg["decisions"] * 2 * candidates * search_s +
                    snapshots * snapshot_s +
                    (snapshots + 2) * len(selected) * len(cfg["diagnostic_starts"]) * window_s +
                    (snapshots + 1) * timing_s) + 120
    return dict(candidate_seconds=search_s, snapshot_seconds=snapshot_s, timing_seconds=timing_s,
                window_seconds=window_s, forecast_seconds=forecast,
                edges=mesh.senders.numel(), candidate_forward_calls=sum(r["forward_calls"] for r in rows),
                repeatability="exact", timing=measured, **snap)


def run(args):
    """Execute a new or resumed attempt with immutable source and input metadata."""
    output = args.output
    output.mkdir(exist_ok=True)
    cfg = yaml.safe_load(CONFIG.read_text())
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    gpu = torch.cuda.get_device_properties(0)
    assert "P100" in gpu.name and (gpu.major, gpu.minor) == (6, 0)
    path = args.parent / CHECKPOINT_PATH
    assert sha(path) == CHECKPOINT
    ck = torch.load(path, map_location="cpu", weights_only=True)
    data = args.parent / "data"
    assert ck["updates"] == 47000 and ck["validation_ids"] == BASELINE
    assert ck["dataset_identity"] == (data / "VERIFIED.md5").read_text()
    assert tensors_hash(ck["state_dict"]) == TENSORS
    dense = load_mesh_checkpoint(path)
    del ck
    splits = split_manifest()
    assert len(set(splits["baseline"] + splits["diagnostic"] + splits["selection"])) == 32
    assert splits["starts"] == cfg["diagnostic_starts"]
    assert splits["horizon"] == cfg["diagnostic_horizon"]
    ranks = masks(dense)["ranks"]
    seal(output / "protocol.json", dict(config=cfg, splits=splits, ranks=ranks, layers=LAYERS,
         checkpoint_sha256=CHECKPOINT, tensor_sha256=TENSORS, valid_sha256=VALID_SHA,
         checkpoint_updates=47000, source=Path("source-commit.txt").read_text().strip(),
         source_manifest_sha256=sha(Path("source.sha256")), deterministic_algorithms=True,
         cublas_workspace=os.environ["CUBLAS_WORKSPACE_CONFIG"], gpu=gpu.name,
         torch=str(torch.__version__), cuda=torch.version.cuda, python=platform.python_version(),
         numpy=np.__version__, yaml=yaml.__version__))
    job = os.environ["LSB_JOBID"]
    seal(output / f"attempt-{job}.json", dict(host=socket.gethostname(), affinity=sorted(os.sched_getaffinity(0)),
                                              job=job, budget_seconds=args.seconds))
    records = CylinderFlowRecords(data / "valid.tfrecord", "valid")
    selected = [(o, mesh, frames) for o, _, mesh, frames in
                load_windows(records, splits["selection"], [cfg["selection_start"]],
                             max(cfg["selection_horizons"].values()), purpose="selection")]
    calibration = calibrate(dense, ranks, selected, cfg, output / f"probe-{job}")
    full = calibration["forecast_seconds"] <= (args.started + args.seconds - time.time()) * .75
    seal(output / f"calibration-{job}.json", dict(**calibration, forecast_fits=full))
    print(f"GPU calibration: {calibration['forecast_seconds']:.1f}s, fits={full}", flush=True)

    def may_start(unit):
        remaining = args.started + args.seconds - time.time()
        if not full:
            remaining -= args.seconds * .25
        needed = {"decision": calibration["candidate_seconds"] * 8,
                  "snapshot": calibration["snapshot_seconds"],
                  "window": calibration["window_seconds"] + calibration["timing_seconds"]}[unit]
        return remaining >= max(30, 3 * needed)

    strategies, selected_all = search(dense, ranks, selected, output, cfg, may_start)
    assert tensors_hash(dense.state_dict()) == TENSORS
    complete = False
    if selected_all:
        seal(output / "search-finished.json", strategies)
        del selected
        complete = evaluate(dense, strategies, records, splits["diagnostic"], output, cfg, may_start)
    assert tensors_hash(dense.state_dict()) == TENSORS
    stops = {}
    for strategy in cfg["selection_horizons"]:
        decisions = sorted((output / strategy).glob("decision-*.json"))
        last = json.loads(decisions[-1].read_text()) if decisions else None
        stops[strategy] = dict(decisions=len(decisions), reason=(
            "no_valid_candidate" if last is not None and last["winner"] is None else
            "decision_limit" if len(decisions) == cfg["decisions"] else "budget"))
    window_rows = [json.loads(path.read_text()) for path in output.glob("*/[0-9]*.json")]
    seal(output / f"finish-{job}.json", dict(search_finished=selected_all,
         available_snapshots=strategies, strategy_stops=stops, all_available_windows_finished=complete,
         all_requested_snapshots_available=sum(map(len, strategies.values())) == 12,
         expected_windows=14 * len(splits["diagnostic"]) * len(cfg["diagnostic_starts"]),
         saved_windows=len(window_rows), failed_windows=sum(row["status"] != "ok" for row in window_rows),
         elapsed_seconds=time.time() - args.started, dense_tensor_sha256=TENSORS))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seconds", type=int, required=True)
    parser.add_argument("--started", type=float, required=True)
    run(parser.parse_args())

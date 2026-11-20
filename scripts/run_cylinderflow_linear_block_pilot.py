"""Fit and evaluate a frozen CylinderFlow processor-pair replacement on POLUS."""

import argparse
import copy
from contextlib import ExitStack
import itertools
import os
from pathlib import Path
import platform
import random
import statistics
import time

import numpy as np
import torch
from torch import nn
import yaml

from gnn_pruning.data.cylinderflow_records import CylinderFlowRecords
from gnn_pruning.experiments.flickr_dense import sha, tensors_hash
from gnn_pruning.models.meshgraphnet import LATENT, load_mesh_checkpoint
from run_cylinderflow_channel_selection import CHECKPOINT_PATH, VALID_SHA, load_windows, score_rollout
from run_cylinderflow_mask_diagnostic import BASELINE, CHECKPOINT, TENSORS, seal, split_manifest


CONFIG = Path("configs/experiments/meshgraphnet_cylinderflow_linear_block_pilot.yaml")
DENSE_REFERENCE = 0.005443890033347998


class AffinePair(nn.Module):
    """Add independent affine estimates of the two-block residuals."""

    def __init__(self):
        super().__init__()
        self.node = nn.Linear(LATENT, LATENT)
        self.edge = nn.Linear(LATENT, LATENT)

    def forward(self, nodes, edges, _mesh):
        return nodes + self.node(nodes), edges + self.edge(edges)


def variant(dense, first, last, replacement=None):
    """Build a model whose processor list physically omits the selected pair."""
    model = copy.deepcopy(dense)
    original = list(model.processors)
    inserted = [] if replacement is None else [replacement]
    model.processors = nn.ModuleList(original[:first] + inserted + original[last + 1:])
    return model.eval()


def fit_ridge(inputs, targets, relative):
    """Fit a centered float64 ridge regression with an unpenalized intercept."""
    x, y = (np.concatenate(rows).astype(np.float64) for rows in (inputs, targets))
    mean_x, mean_y = x.mean(axis=0), y.mean(axis=0)
    x -= mean_x
    y -= mean_y
    gram = x.T @ x / len(x)
    penalty = relative * np.trace(gram) / gram.shape[0]
    if not np.isfinite(penalty) or penalty <= 0:
        raise ValueError("Invalid calibration variance")
    coef = np.linalg.solve(gram + penalty * np.eye(gram.shape[0]), x.T @ y / len(x))
    bias = mean_y - mean_x @ coef
    assert np.isfinite(coef).all() and np.isfinite(bias).all()
    return torch.from_numpy(coef.T.copy()).float(), torch.from_numpy(bias).float(), float(penalty)


@torch.no_grad()
def calibrate(dense, train, cfg, first, last, device):
    """Capture paired latent states from official training frames only."""
    assert train.split == "train"
    rng = random.Random(cfg["seed"])
    ids = sorted(rng.sample(range(len(train)), cfg["train_trajectories"]))
    data = {"node": ([], []), "edge": ([], [])}
    captured = {}
    sampling = []

    def before(_module, args):
        captured["before"] = args[:2]

    def after(_module, _args, result):
        captured["after"] = result

    with ExitStack() as hooks:
        hooks.callback(dense.processors[first].register_forward_pre_hook(before).remove)
        hooks.callback(dense.processors[last].register_forward_hook(after).remove)
        for ordinal in ids:
            mesh = train.mesh(ordinal).to(device)
            for frame in cfg["train_frames"]:
                captured.clear()
                dense(mesh, train.velocity(ordinal, frame, 1)[0].to(device))
                entry = dict(trajectory=ordinal, frame=frame)
                for branch, x, y in zip(data, captured["before"], captured["after"]):
                    n = min(x.shape[0], cfg[f"{branch}s_per_frame"])
                    ids_sample = sorted(rng.sample(range(x.shape[0]), n))
                    index = torch.tensor(ids_sample, device=device)
                    sampled = x.index_select(0, index)
                    data[branch][0].append(sampled.cpu().numpy())
                    data[branch][1].append((y.index_select(0, index) - sampled).cpu().numpy())
                    entry[f"{branch}_indices"] = ids_sample
                sampling.append(entry)

    replacement = AffinePair()
    fits = {}
    for name, (inputs, targets) in data.items():
        weight, bias, penalty = fit_ridge(inputs, targets, cfg["ridge_relative"])
        getattr(replacement, name).weight.copy_(weight)
        getattr(replacement, name).bias.copy_(bias)
        fits[name] = dict(rows=sum(map(len, inputs)), ridge_penalty=penalty)
    return replacement, dict(trajectory_ids=ids, samples=sampling, fits=fits)


@torch.inference_mode()
def check_variants(models, dense, mesh, initial, first, last):
    """Check execution counts, exact frozen tensors, and finite GPU output."""
    expected = {"A": [1] * 15, "B": [1] * 13, "C": [1] * 14}
    for label, model in models.items():
        calls = [0] * len(model.processors)

        def count(index):
            def hook(_module, _args, _result):
                calls[index] += 1
            return hook

        with ExitStack() as hooks:
            for i, block in enumerate(model.processors):
                hooks.callback(block.register_forward_hook(count(i)).remove)
            velocity = model.step(mesh, initial)
        assert calls == expected[label]
        assert torch.isfinite(velocity).all() and torch.equal(velocity[~mesh.active], initial[~mesh.active])
        source_blocks = list(range(15))
        if label != "A":
            del source_blocks[first:last + 1]
        for n, original in enumerate(source_blocks):
            pos = n + int(label == "C" and n >= first)
            assert tensors_hash(model.processors[pos].state_dict()) == tensors_hash(
                dense.processors[original].state_dict())
        for name in ("node_norm", "edge_norm", "output_norm", "node_encoder", "edge_encoder", "decoder"):
            assert tensors_hash(getattr(model, name).state_dict()) == tensors_hash(getattr(dense, name).state_dict())
    return expected


@torch.inference_mode()
def steps(model, mesh, velocity, count):
    """Execute autoregressive steps with only the final velocity kept."""
    for _ in range(count):
        velocity = model.step(mesh, velocity)
    return velocity


def benchmark(models, windows, cfg, output):
    """Time six orders per window, balancing all model positions on the same GPU."""
    orders = list(itertools.permutations(models))
    rng = random.Random(cfg["seed"])
    for ordinal, start, mesh, frames in windows:
        for model in models.values():
            steps(model, mesh, frames[0], cfg["timing_warmup_steps"])
        rng.shuffle(orders)
        samples = []
        for repeat, order in enumerate(orders):
            for label in order:
                torch.cuda.synchronize()
                began = time.perf_counter()
                prediction = steps(models[label], mesh, frames[0], cfg["horizon"])
                torch.cuda.synchronize()
                seconds = time.perf_counter() - began
                samples.append(dict(repeat=repeat, variant=label, seconds=seconds,
                                    final_finite=bool(torch.isfinite(prediction).all())))
        seal(output / f"{ordinal}-{start}.json", samples)
        print(f"timed {ordinal}/{start}", flush=True)


def scientific(label, model, windows, output):
    """Record detached scientific curves independently of timing."""
    folder = output / label
    folder.mkdir()
    rows = []
    for ordinal, start, mesh, frames in windows:
        item = score_rollout(model, mesh, frames)
        curve = item["curve"]
        item.update(E100=curve[-1] if len(curve) == 100 else None,
                    mean100=statistics.mean(curve) if len(curve) == 100 else None)
        seal(folder / f"{ordinal}-{start}.json", item)
        rows.append(item)
    print(f"evaluated {label}: {sum(r['status'] == 'ok' for r in rows)}/16", flush=True)
    return rows


def run(parent, out):
    """Run calibration, parity checks, scientific rollouts, and separate timing."""
    out.mkdir()
    cfg = yaml.safe_load(CONFIG.read_text())
    first, last = [i - 1 for i in cfg["blocks_one_based"]]
    assert (first, last) == (6, 7) and cfg["horizon"] == 100
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    gpu = torch.cuda.get_device_properties(0)
    assert torch.cuda.device_count() == 1 and "P100" in gpu.name and (gpu.major, gpu.minor) == (6, 0)
    path = parent / CHECKPOINT_PATH
    assert sha(path) == CHECKPOINT
    payload = torch.load(path, map_location="cpu", weights_only=True)
    data = parent / "data"
    assert payload["updates"] == 47000 and payload["validation_ids"] == BASELINE
    assert payload["dataset_identity"] == (data / "VERIFIED.md5").read_text()
    del payload
    dense = load_mesh_checkpoint(path, "cuda")
    assert tensors_hash(dense.state_dict()) == TENSORS
    splits = split_manifest()
    seal(out / "protocol.json", dict(config=cfg, splits=splits, job=os.environ["LSB_JOBID"],
         host=platform.node(), gpu=gpu.name, gpu_bytes=gpu.total_memory, driver_check="see LSF stdout",
         torch=str(torch.__version__), cuda=torch.version.cuda, python=platform.python_version(),
         numpy=np.__version__, affinity=sorted(os.sched_getaffinity(0)),
         checkpoint_sha256=CHECKPOINT, dense_tensor_sha256=TENSORS,
         valid_sha256=VALID_SHA, train_md5="046cf222b4a6632c9b3764522ece980b",
         source=Path("source-commit.txt").read_text().strip(), source_manifest_sha256=sha(Path("source.sha256"))))
    torch.cuda.synchronize()
    began = time.perf_counter()
    train = CylinderFlowRecords(data / "train.tfrecord", "train")
    fitted, calibration = calibrate(dense, train, cfg, first, last, "cuda")
    torch.cuda.synchronize()
    calibration["seconds"] = time.perf_counter() - began
    del train
    maps = out / "maps.pt"
    torch.save(fitted.state_dict(), maps)
    reloaded = AffinePair().eval()
    reloaded.load_state_dict(torch.load(maps, map_location="cpu", weights_only=True), strict=True)
    assert tensors_hash(fitted.state_dict()) == tensors_hash(reloaded.state_dict())
    fitted.cuda().eval()
    reloaded.cuda()
    probe = torch.arange(LATENT * 3, device="cuda").float().reshape(3, LATENT) / LATENT
    with torch.inference_mode():
        assert all(torch.equal(a, b) for a, b in zip(fitted(probe, probe, None), reloaded(probe, probe, None)))
    del fitted
    seal(out / "calibration.json", dict(**calibration, maps_sha256=sha(maps), reload_output_exact=True))
    print("calibration complete", calibration["seconds"], flush=True)
    models = {"A": dense, "B": variant(dense, first, last), "C": variant(dense, first, last, reloaded)}
    windows = load_windows(CylinderFlowRecords(data / "valid.tfrecord", "valid"),
                           splits["diagnostic"], splits["starts"], cfg["horizon"], purpose="diagnostic")
    seal(out / "gpu-checks.json", dict(calls=check_variants(models, dense, windows[0][2], windows[0][3][0], first, last),
         executed_original_blocks=dict(A=list(range(1, 16)), B=[1,2,3,4,5,6,9,10,11,12,13,14,15],
                                       C=[1,2,3,4,5,6,9,10,11,12,13,14,15]),
         parameter_counts={k: sum(p.numel() for p in m.parameters()) for k, m in models.items()}))
    initial_hashes = {k: tensors_hash(m.state_dict()) for k, m in models.items()}
    rows = scientific("A", dense, windows, out)
    complete = all(row["status"] == "ok" for row in rows)
    e100 = statistics.mean(row["E100"] for row in rows) if complete else None
    relative = e100 / DENSE_REFERENCE - 1 if complete else None
    passed = complete and abs(relative) <= cfg["dense_e100_relative_tolerance"]
    seal(out / "dense-gate.json", dict(reference=DENSE_REFERENCE, observed=e100, delta_relative=relative, passed=passed))
    if not passed:
        raise RuntimeError("Dense reproduction gate failed")
    for label in ("B", "C"):
        scientific(label, models[label], windows, out)
    timing = out / "timing"
    timing.mkdir()
    benchmark(models, windows, cfg, timing)
    assert initial_hashes == {k: tensors_hash(m.state_dict()) for k, m in models.items()}
    seal(out / "finish.json", dict(windows=48, timing_samples=288, state_sha256=initial_hashes))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.parent, args.output)

"""Fit a one-round affine graph block against the same targets as the pointwise pilot."""

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import yaml

from gnn_pruning.data.cylinderflow_records import CylinderFlowRecords
from gnn_pruning.experiments.flickr_dense import sha, tensors_hash
from gnn_pruning.models.meshgraphnet import LATENT, load_mesh_checkpoint, sum_incoming
from run_cylinderflow_linear_block_pilot import (
    AffinePair, CHECKPOINT, CHECKPOINT_PATH, CONFIG, TENSORS, benchmark, fit_ridge,
    load_windows, scientific, seal, split_manifest, variant,
)


class LinearExchange(nn.Module):
    """Approximate a pair of processors with one affine message and update."""

    def __init__(self):
        super().__init__()
        self.edge = nn.Linear(3 * LATENT, LATENT)
        self.node = nn.Linear(2 * LATENT, LATENT)

    def edge_inputs(self, nodes, edges, mesh):
        return torch.cat((nodes[mesh.senders], nodes[mesh.receivers], edges), dim=1)

    def node_inputs(self, nodes, messages, mesh):
        incoming = sum_incoming(messages, mesh.receivers, len(nodes))
        return torch.cat((nodes, incoming), dim=1)

    def forward(self, nodes, edges, mesh):
        messages = self.edge(self.edge_inputs(nodes, edges, mesh))
        return nodes + self.node(self.node_inputs(nodes, messages, mesh)), edges + messages


@torch.no_grad()
def calibrate(dense, train, samples, first, last, ridge):
    # Revisit each frame in the node pass so its inputs use the fitted edge map.
    assert train.split == "train"
    replacement = LinearExchange().cuda().eval()
    stats = {}
    for branch in ("edge", "node"):
        inputs, targets = [], []
        ordinal = None
        for sample in samples:
            if ordinal != sample["trajectory"]:
                ordinal = sample["trajectory"]
                mesh = train.mesh(ordinal).to("cuda")
            velocity = train.velocity(ordinal, sample["frame"], 1)[0].cuda()
            captured = {}

            def before(_module, args):
                captured["before"] = args[:2]

            def after(_module, _args, outputs):
                captured["after"] = outputs

            h1 = dense.processors[first].register_forward_pre_hook(before)
            h2 = dense.processors[last].register_forward_hook(after)
            try:
                dense(mesh, velocity)
            finally:
                h1.remove()
                h2.remove()
            nodes, edges = captured["before"]
            new_nodes, new_edges = captured["after"]
            x = replacement.edge_inputs(nodes, edges, mesh)
            if branch == "edge":
                target = new_edges - edges
            else:
                messages = replacement.edge(x)
                x = replacement.node_inputs(nodes, messages, mesh)
                target = new_nodes - nodes
            idx = torch.tensor(sample[branch + "_indices"], device="cuda")
            inputs.append(x[idx].cpu().numpy())
            targets.append(target[idx].cpu().numpy())

        weight, bias, penalty = fit_ridge(inputs, targets, ridge)
        layer = getattr(replacement, branch)
        layer.weight.copy_(weight)
        layer.bias.copy_(bias)
        stats[branch] = dict(rows=sum(map(len, inputs)), features=layer.in_features,
                             penalty=penalty)
    return replacement, stats


def run(parent, old, out):
    out.mkdir()
    cfg = yaml.safe_load(CONFIG.read_text())
    first, last = [i - 1 for i in cfg["blocks_one_based"]]
    assert (first, last) == (6, 7)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    assert torch.cuda.device_count() == 1 and "P100" in torch.cuda.get_device_name(0)
    assert sha(parent / CHECKPOINT_PATH) == CHECKPOINT
    assert sha(old / "maps.pt") == "ce5498b9e5bfef954e56e5869280a2bd19778f0a53993ed0dfe233acc82d004d"
    calibration_path = old / "calibration.json"
    assert sha(calibration_path) == "029ec159fc50f718d6536d1d8441e58c7d3e89f8877cfd8cea99991ba0fbb809"
    previous = json.loads(calibration_path.read_text())
    dense = load_mesh_checkpoint(parent / CHECKPOINT_PATH, "cuda")
    assert tensors_hash(dense.state_dict()) == TENSORS
    splits = split_manifest()
    seal(out / "protocol.json", dict(config=cfg, splits=splits, checkpoint=CHECKPOINT,
         job=os.environ["LSB_JOBID"], host=platform.node(), gpu=torch.cuda.get_device_name(0),
         source=Path("source-commit.txt").read_text().strip(), torch=str(torch.__version__),
         calibration_manifest=sha(calibration_path), pointwise_maps=sha(old / "maps.pt")))
    began = time.perf_counter()
    train = CylinderFlowRecords(parent / "data/train.tfrecord", "train")
    exchange, stats = calibrate(dense, train, previous["samples"], first, last, cfg["ridge_relative"])
    torch.cuda.synchronize()
    seconds = time.perf_counter() - began
    del train
    torch.save(exchange.state_dict(), out / "exchange.pt")
    reloaded = LinearExchange().cuda().eval()
    reloaded.load_state_dict(torch.load(out / "exchange.pt", map_location="cpu", weights_only=True))
    assert tensors_hash(reloaded.state_dict()) == tensors_hash(exchange.state_dict())
    seal(out / "calibration.json", dict(fits=stats, seconds=seconds,
         trajectories=previous["trajectory_ids"], maps_sha256=sha(out / "exchange.pt")))
    control = AffinePair().cuda().eval()
    control.load_state_dict(torch.load(old / "maps.pt", map_location="cpu", weights_only=True))
    models = {"A": dense, "C": variant(dense, first, last, control),
              "D": variant(dense, first, last, reloaded)}
    windows = load_windows(CylinderFlowRecords(parent / "data/valid.tfrecord", "valid"),
                           splits["diagnostic"], splits["starts"], cfg["horizon"], purpose="diagnostic")
    hashes = {k: tensors_hash(m.state_dict()) for k, m in models.items()}
    counts = {k: sum(p.numel() for p in m.parameters()) for k, m in models.items()}
    for label, model in models.items():
        scientific(label, model, windows, out)
        if label in ("A", "C"):
            for ordinal, start, _, _ in windows:
                filename = f"{ordinal}-{start}.json"
                archived = json.loads((old / label / filename).read_text())
                current = json.loads((out / label / filename).read_text())
                assert archived["status"] == current["status"] == "ok"
                np.testing.assert_allclose(current["curve"], archived["curve"], rtol=5e-3, atol=1e-8)
    timing = out / "timing"
    timing.mkdir()
    benchmark(models, windows, cfg, timing)
    assert hashes == {k: tensors_hash(m.state_dict()) for k, m in models.items()}
    seal(out / "finish.json", dict(windows=3*len(windows), timing_samples=18*len(windows),
                                   state_sha256=hashes, parameters=counts))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("parent", "old", "out"):
        parser.add_argument(arg, type=Path)
    run(**vars(parser.parse_args()))

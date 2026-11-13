"""Audit collected diagnostic evidence without model inference and export rank details."""

import argparse
import itertools
import json
from pathlib import Path
import random

import numpy as np
import torch
from scipy.stats import spearmanr

from gnn_pruning.experiments.flickr_dense import sha, tensors_hash
from gnn_pruning.models.meshgraphnet import load_mesh_checkpoint

from analyze_cylinderflow_mask_diagnostic import KEYS, comparisons, csv_write, direction, metrics
from run_cylinderflow_mask_diagnostic import CHECKPOINT, TENSORS


def audit(collected, output):
    """Verify inputs, reconstruct physical tensor hashes and audit the saved curves."""
    source = collected / "results"
    splits = json.loads((source / "splits.json").read_text())
    manifest = json.loads((source / "masks.json").read_text())
    protocol = json.loads((source / "protocol.json").read_text())
    finish = json.loads((source / "finish-1618575.json").read_text())
    assert (source / "splits.json").read_bytes() == (collected / "split-preflight.json").read_bytes()
    assert (source / "masks.json").read_bytes() == (collected / "masks-preflight.json").read_bytes()
    assert protocol["evaluation_source"] == (collected / "source.sha256").read_text()
    assert protocol["base_revision"] == (collected / "source-commit.txt").read_text().strip()
    assert splits["starts"] == [100, 300] and splits["split"] == "valid" and splits["horizon"] == 100
    order = list(range(100))
    random.Random(101).shuffle(order)
    assert splits["baseline"] + splits["diagnostic"] + splits["selection"] == order[:32]
    path = collected / "best.pt"
    assert sha(path) == protocol["checkpoint_sha256"] == CHECKPOINT
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    assert checkpoint["updates"] == 47000 and checkpoint["widths"] == [[128, 128]] * 15
    assert checkpoint["validation"]["e100"] == 0.002244355198854464
    assert checkpoint["dataset_identity"] == protocol["dataset"]
    assert checkpoint["validation_ids"] == splits["baseline"]
    model = load_mesh_checkpoint(path)
    state = model.state_dict()
    assert tensors_hash(state) == protocol["tensor_sha256"] == finish["dense_tensor_sha256"] == TENSORS
    for name, recorded in protocol["normalizers"].items():
        norm = getattr(model, name)
        mean, std = norm.moments()
        assert recorded["count"] == norm.count.item() and recorded["updates"] == norm.updates.item() == 48000
        np.testing.assert_allclose(mean.tolist(), recorded["mean"], rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(std.tolist(), recorded["std"], rtol=1e-6, atol=1e-8)
    assert set(protocol["normalizers"]) == {"node_norm", "edge_norm", "output_norm"}
    library = manifest["masks"]
    assert len(library) == len({tuple(tuple(channels) for channels in m["removed"]) for m in library}) == 24
    assert manifest["layers"] == [[i, name] for i in range(15) for name in ("edge", "node")]
    names = ["dense", "persistence", *(m["id"] for m in library)]
    ids, starts = splits["diagnostic"], splits["starts"]
    expected = {source / name / f"{i}-{s}.json" for name, i, s in itertools.product(names, ids, starts)}
    actual = set(source.glob("*/[0-9]*.json"))
    assert expected == actual and len(actual) == finish["saved_windows"] == 416
    values = np.empty((len(names), len(ids), len(starts), len(KEYS)))
    for n, i, s in itertools.product(range(len(names)), range(len(ids)), range(len(starts))):
        row = json.loads((source / names[n] / f"{ids[i]}-{starts[s]}.json").read_text())
        assert row["status"] == "ok" and row["first_failed_step"] is None
        assert len(row["curve"]) == 100 and np.isfinite(row["curve"]).all() and min(row["curve"]) >= 0
        values[n, i, s] = metrics(row["curve"])
    assert finish["failed_windows"] == 0 and finish["status"] == "complete"

    rng = random.Random(101)
    maxima = []
    for n, mask in enumerate(library):
        lo, hi, stratum = [(0, 43, "lower"), (43, 86, "middle"), (86, 128, "upper")][n // 8]
        assert mask["stratum"] == stratum and mask["id"] == f"{stratum}-{n%8+1:02}"
        assert len(mask["removed"]) == 30
        projected = state.copy()
        for (block, kind), removed, recorded_rank in zip(manifest["layers"], mask["removed"], manifest["ranks"]):
            p = f"processors.{block}.{kind}"
            weights = state[f"{p}.2.weight"].numpy().astype(np.float64)
            bias = state[f"{p}.2.bias"].numpy().astype(np.float64)
            outgoing = state[f"{p}.4.weight"].numpy().astype(np.float64)
            scores = np.sqrt(np.square(weights).sum(1) + bias**2) * np.linalg.norm(outgoing, axis=0)
            ranked = np.lexsort((np.arange(128), scores)).tolist()
            assert ranked == recorded_rank and removed == sorted(rng.sample(ranked[lo:hi], 4))
            keep = sorted(set(range(128)) - set(removed))
            projected[f"{p}.2.weight"] = state[f"{p}.2.weight"][keep]
            projected[f"{p}.2.bias"] = state[f"{p}.2.bias"][keep]
            projected[f"{p}.4.weight"] = state[f"{p}.4.weight"][:, keep]
        folder = source / mask["id"]
        surgery = json.loads((folder / "surgery.json").read_text())
        assert surgery["widths"] == [[124, 124]] * 15
        parameter_count = sum(projected[key].numel() for key, _ in model.named_parameters())
        assert parameter_count == surgery["parameters"] == 2302090
        assert tensors_hash(projected) == surgery["tensor_sha256"]
        error = json.loads((folder / "parity-1618575.json").read_text())["masked_max_abs"]
        assert np.isfinite(error) and error >= 0
        maxima.append(error)

    by_trajectory = values.mean(axis=2)
    averages = by_trajectory.mean(axis=1)
    masks = averages[2:]
    ranks = []
    for name, vals in zip(names[2:], masks):
        ranks.append(dict(mask=name, **{f"rank_{KEYS[k]}": int(1+sum(masks[:, k] < vals[k]))
                                        for k in (0, 5, 4)}))
    csv_write(output / "mask_ranks.csv", ranks)
    rows = []
    for j, ordinal in enumerate(ids):
        values_j = by_trajectory[2:, j]
        dense_j = by_trajectory[0, j]
        rows.append(dict(trajectory=ordinal, dense_E100=dense_j[4],
                         min_mask_E100=values_j[:, 4].min(), max_mask_E100=values_j[:, 4].max(),
                         improving_E100=int((direction(values_j[:, 4], dense_j[4]) < 0).sum()),
                         rho_E1=float(spearmanr(values_j[:, 0], values_j[:, 4]).statistic),
                         rho_mean5=float(spearmanr(values_j[:, 5], values_j[:, 4]).statistic)))
    csv_write(output / "trajectory_variation.csv", rows)
    omitted = []
    for j, ordinal in enumerate(ids):
        remaining = np.delete(by_trajectory, j, axis=1).mean(axis=1)
        stats = comparisons(remaining[2:], remaining[0])
        omitted.append(dict(omitted_trajectory=ordinal,
                            delta_spearman=float(stats["mean5"]["spearman"] - stats["E1"]["spearman"])))
    csv_write(output / "leave_one_trajectory_out.csv", omitted)

    # Sensitivity to the numerical decision band, with the original band unchanged.
    sensitivity = {}
    for factor in (1, 10, 100):
        sensitivity[factor] = {}
        i, j = np.triu_indices(len(masks), 1)
        for k in (0, 5):
            x, y = masks[i][:, [k, 4]], masks[j][:, [k, 4]]
            delta = x - y
            comparable = (np.abs(delta) > factor*(1e-9 + 1e-4*np.maximum(np.abs(x), np.abs(y)))).all(1)
            inverted = ((delta[:, 0]*delta[:, 1] < 0) & comparable).sum()
            sensitivity[factor][KEYS[k]] = [int(inverted), int(comparable.sum())]
    (output / "verification.json").write_text(json.dumps(dict(
        checkpoint_updates=47000, checkpoint_sha256=CHECKPOINT, dense_tensor_sha256=TENSORS,
        verified_window_records=len(actual), verified_physical_tensor_hashes=len(library),
        worker_max_masked_abs=max(maxima), tolerance_sensitivity=sensitivity), indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("collected", type=Path)
    parser.add_argument("analysis", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    audit(args.collected, args.analysis)

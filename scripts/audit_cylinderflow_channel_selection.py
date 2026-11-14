"""Verify job 1618600 from saved tensors and JSON, without model or data execution."""

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import statistics

import numpy as np
import torch
import yaml

from gnn_pruning.experiments.flickr_dense import sha, tensors_hash


METRICS = ("E1", "E5", "E20", "E50", "E100", "mean5", "mean20", "mean100")
LEVELS = {2: (4, 2302090), 4: (8, 2271250), 8: (16, 2209570), 15: (30, 2101630)}
DENSE = 2332930


def read(path):
    return json.loads(path.read_text())


def table(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def write_table(path, rows):
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def near(value, best):
    """Reuse the frozen numerical tolerance for descriptive E100 ties."""
    return abs(value - best) <= 1e-9 + 1e-4 * max(abs(value), abs(best))


def expected_library(ranks, parent):
    """Reconstruct the fixed library directly from the documented seed rule."""
    key = json.dumps([101, parent], separators=(",", ":")).encode()
    rng = random.Random(int(hashlib.sha256(key).hexdigest(), 16))
    remaining = [[c for c in rank if c not in gone] for rank, gone in zip(ranks, parent)]
    library = [[sorted(row[:2]) for row in remaining]]
    for pool, count in (([row[:8] for row in remaining], 4), (remaining, 3)):
        for _ in range(count):
            for _ in range(1000):
                action = [sorted(rng.sample(row, 2)) for row in pool]
                if action not in library:
                    library.append(action)
                    break
            else:
                raise AssertionError("Duplicate retry limit exceeded")
    return library


def snapshot(folder, decision, parent, names, state):
    """Verify exact tensor projection and physical shape against the frozen file."""
    n, count = LEVELS[decision]
    info = read(folder / "surgery.json")
    assert info["decision"] == decision and info["removed"] == parent
    assert sha(folder / "physical.pt") == info["checkpoint_sha256"]
    saved = torch.load(folder / "physical.pt", map_location="cpu", weights_only=True)
    assert saved["format"] == "meshgraphnet-cylinderflow-v1"
    expected = dict(state)
    assert all(len(row) == n for row in parent)
    for p, removed in zip(names, parent):
        keep = [channel for channel in range(128) if channel not in removed]
        expected[p + ".2.weight"] = state[p + ".2.weight"][keep]
        expected[p + ".2.bias"] = state[p + ".2.bias"][keep]
        expected[p + ".4.weight"] = state[p + ".4.weight"][:, keep]
    actual = saved["state_dict"]
    assert actual.keys() == expected.keys()
    assert all(torch.equal(actual[key], value) for key, value in expected.items())
    assert tensors_hash(actual) == info["tensor_sha256"]
    widths = [[actual[p + ".2.bias"].numel() for p in names[i:i + 2]] for i in range(0, 30, 2)]
    assert widths == saved["widths"] == info["widths"] == [[128 - n] * 2] * 15
    assert sum(t.numel() for k, t in actual.items() if k.endswith((".weight", ".bias"))) == count
    assert count == info["parameters"] and info["compression"] == 1 - count / DENSE
    return dict(parameters=count, width=128 - n, projection="exact",
                checkpoint_sha256=info["checkpoint_sha256"], tensor_sha256=info["tensor_sha256"])


def audit_decisions(source, protocol, dense_path):
    """Independently reconstruct every score, chosen action and saved mask."""
    assert sha(dense_path) == protocol["checkpoint_sha256"]
    ckpt = torch.load(dense_path, map_location="cpu", weights_only=True)
    splits = protocol["splits"]
    assert ckpt["updates"] == protocol["checkpoint_updates"] == 47000
    assert ckpt["validation_ids"] == splits["baseline"]
    state = ckpt["state_dict"]
    assert tensors_hash(state) == protocol["tensor_sha256"]
    assert ckpt["widths"] == [[128, 128]] * 15
    assert sum(t.numel() for k, t in state.items() if k.endswith((".weight", ".bias"))) == DENSE
    layers = [[i, kind] for i in range(15) for kind in ("edge", "node")]
    assert layers == protocol["layers"]
    names = [f"processors.{i}.{kind}" for i, kind in layers]
    ranks = []
    for p in names:
        w, b, v = [state[p + suffix].numpy().astype(np.float64)
                   for suffix in (".2.weight", ".2.bias", ".4.weight")]
        importance = np.sqrt((w * w).sum(axis=1) + b * b) * np.linalg.norm(v, axis=0)
        ranks.append(np.lexsort((np.arange(128), importance)).tolist())
    assert ranks == protocol["ranks"]
    rows, masks = [], {}
    for strategy, horizon in (("magnitude", 0), ("one_step", 1), ("five_step", 5)):
        parent = [[] for _ in names]
        for decision in range(1, 16):
            row = read(source / strategy / f"decision-{decision:02}.json")
            actions = expected_library(ranks, parent)
            assert row["parent"] == parent and row["candidates"] == actions
            assert len(row["windows"]) == row["candidate_evaluations"] == 8
            assert row["model_candidate_evaluations"] == (8 if horizon else 0)
            scores = []
            for action, windows in zip(actions, row["windows"]):
                if horizon:
                    assert [w["ordinal"] for w in windows] == splits["selection"]
                    for window in windows:
                        curve = window["curve"]
                        assert len(curve) == window["forward_calls"] == horizon
                        assert all(math.isfinite(x) and x >= 0 for x in curve)
                        assert window["status"] == "ok" and window["first_failed_step"] is None
                    scores.append(math.fsum(math.fsum(w["curve"]) / horizon for w in windows) / 8)
                else:
                    assert windows == []
                    scores.append(sum(rank.index(c) for rank, pair in zip(ranks, action) for c in pair))
            np.testing.assert_allclose(scores, row["scores"], rtol=1e-14, atol=0)
            best = min(scores)
            eligible = [i for i, score in enumerate(scores)
                        if (near(score, best) if horizon else score == best)]
            winner = min(eligible, key=lambda i: actions[i])
            assert winner == row["winner"] and row["forward_calls"] == 64 * horizon
            for old, pair in zip(parent, actions[winner]):
                assert not set(old).intersection(pair)
            parent = [sorted(old + new) for old, new in zip(parent, actions[winner])]
            assert all(len(set(layer)) == 2 * decision for layer in parent)
            checked = dict(parameters=None, width=None, projection=None,
                           checkpoint_sha256=None, tensor_sha256=None)
            if decision in LEVELS:
                n, _ = LEVELS[decision]
                name = f"{strategy}-{n:02}"
                checked = snapshot(source / name, decision, parent, names, state)
                masks[name] = parent
            rows.append(dict(strategy=strategy, decision=decision, eligible=len(eligible),
                             winner_zero_based=winner, best_score=best, winning_score=scores[winner],
                             search_seconds=row["search_seconds"], candidate_evaluations=8,
                             actual_forwards=row["forward_calls"], failed_candidates=0,
                             original_ids_verified=True, removed_per_mlp=2 * decision, **checked))
    return rows, masks


def audit_metrics(source, analysis, splits, masks):
    """Recompute window reductions, paired estimates, overlap and latency summaries."""
    ids, starts = splits["diagnostic"], splits["starts"]
    summary = table(analysis / "summary.csv")
    assert {r["model"] for r in summary} == {"dense", "persistence", *masks}
    values, raw = {}, {}
    for row in summary:
        name = row["model"]
        folder = source / name
        assert {p.name for p in folder.glob("[0-9]*.json")} == {
            f"{ordinal}-{start}.json" for ordinal in ids for start in starts}
        per_trajectory = []
        for ordinal in ids:
            windows = []
            for start in starts:
                w = read(folder / f"{ordinal}-{start}.json")
                assert w["status"] == "ok" and w["first_failed_step"] is None
                assert w["forward_calls"] == (0 if name == "persistence" else 100)
                curve = w["curve"]
                assert len(curve) == 100 and all(math.isfinite(x) and x >= 0 for x in curve)
                metrics = [curve[h - 1] for h in (1, 5, 20, 50, 100)]
                metrics.extend(math.fsum(curve[:h]) / h for h in (5, 20, 100))
                windows.append(metrics)
            per_trajectory.append(np.mean(windows, axis=0))
        raw[name] = np.array(per_trajectory)
        values[name] = raw[name].mean(axis=0)
        np.testing.assert_allclose(values[name], [float(row[key]) for key in METRICS], rtol=1e-14)
        assert (row["ok"], row["failed"], row["missing"]) == ("16", "0", "0")
        if name != "persistence":
            timing = read(folder / "timing.json")
            assert timing["warmup_calls"] == 5 and len(timing["seconds"]) == timing["timed_calls"] == 20
            assert (timing["ordinal"], timing["start"]) == (80, 100)
            assert timing["host"] == row["inference_host"]
            assert statistics.mean(timing["seconds"]) == float(row["inference_mean_seconds"])
            assert statistics.stdev(timing["seconds"]) == float(row["inference_sd_seconds"])
    trajectories = []
    dense = raw["dense"]
    for row in summary:
        name = row["model"]
        row["E100_degradation_from_dense_pct"] = 100 * (values[name][4] / values["dense"][4] - 1)
        for index, ordinal in enumerate(ids):
            result = raw[name][index]
            trajectories.append(dict(model=name, trajectory=ordinal, **dict(zip(METRICS, result)),
                                     delta_dense_E100=result[4] - dense[index, 4],
                                     E100_degradation_from_dense_pct=100 * (result[4] / dense[index, 4] - 1)))

    # One draw matrix preserves both windows and the same trajectory indices for all conditions.
    draws = np.random.Generator(np.random.PCG64(101)).integers(8, size=(10000, 8))
    contrasts = table(analysis / "contrasts.csv")
    paired_rows = []
    for row in contrasts:
        name, reference = row["model"], row["reference"]
        i = METRICS.index(row["metric"])
        left, right = raw[name][:, i], raw[reference][:, i]
        differences = left - right
        interval = np.percentile(np.mean(differences[draws], axis=1), [2.5, 97.5])
        np.testing.assert_allclose([differences.mean(), *interval],
                                   [float(row[k]) for k in ("delta", "lower", "upper")],
                                   rtol=1e-13, atol=1e-17)
        assert row["paired_trajectories"] == "8"
        if row["metric"] == "E100":
            outcomes = ["tie" if near(a, b) else "win" if a < b else "loss" for a, b in zip(left, right)]
            row.update(trajectory_wins=outcomes.count("win"), trajectory_ties=outcomes.count("tie"),
                       trajectory_losses=outcomes.count("loss"))
            for ordinal, a, b, delta, result in zip(ids, left, right, differences, outcomes):
                paired_rows.append(dict(model=name, reference=reference, trajectory=ordinal,
                                        E100=a, reference_E100=b, delta=delta, outcome=result))
        else:
            row.update(trajectory_wins=None, trajectory_ties=None, trajectory_losses=None)

    overlaps = table(analysis / "overlap.csv")
    for row in overlaps:
        left, right = [{(i, c) for i, layer in enumerate(masks[f"{name}-{int(row['level']):02}"])
                        for c in layer} for name in (row["a"], row["b"])]
        inter, union = len(left & right), len(left | right)
        assert int(row["removed_intersection"]) == inter
        assert float(row["removed_jaccard"]) == inter / union
        assert float(row["retained_jaccard"]) == (3840 - union) / (3840 - inter)
        assert int(row["identical_mlps"]) == 30 - len({i for i, _ in left ^ right})
    assert len(overlaps) == 12
    return summary, trajectories, contrasts, paired_rows, overlaps


def audit(collected, analysis, dense_path, output):
    """Cross-check provenance and export compact evidence tables."""
    source = collected / "results"
    protocol = read(source / "protocol.json")
    cfg = yaml.safe_load(Path("configs/experiments/meshgraphnet_cylinderflow_channel_selection.yaml").read_text())
    assert cfg == protocol["config"]
    assert protocol["source"] == (collected / "source-commit.txt").read_text().strip()
    assert protocol["source_manifest_sha256"] == sha(collected / "source.sha256")
    assert protocol["valid_sha256"] == "79171fa282fcd4fd5ff93d13e04ade43e83c1b11a0b96eee0951e4e0ed37f3ca"
    splits = protocol["splits"]
    order = list(range(100))
    random.Random(101).shuffle(order)
    assert splits["baseline"] + splits["diagnostic"] + splits["selection"] == order[:32]
    assert splits["starts"] == [100, 300] and splits["split"] == "valid" and cfg["selection_start"] == 200
    decisions, masks = audit_decisions(source, protocol, dense_path)
    finish = read(source / "finish-1618600.json")
    assert finish["dense_tensor_sha256"] == protocol["tensor_sha256"]
    assert finish["search_finished"] and finish["all_available_windows_finished"]
    assert finish["saved_windows"] == finish["expected_windows"] == 224
    assert finish["failed_windows"] == 0 and len(list(source.glob("*/[0-9]*.json"))) == 224
    assert sum(d["actual_forwards"] for d in decisions) == 5760
    summary, trajectories, contrasts, paired, overlaps = audit_metrics(source, analysis, splits, masks)
    for row in summary[2:]:
        strategy, level = row["model"].rsplit("-", 1)
        prefix = [d for d in decisions if d["strategy"] == strategy and d["removed_per_mlp"] <= int(level)]
        assert int(row["search_forwards"]) == sum(d["actual_forwards"] for d in prefix)
        assert int(row["candidates"]) == sum(d["candidate_evaluations"] for d in prefix)
        assert float(row["search_seconds"]) == sum(d["search_seconds"] for d in prefix)
    assert len(decisions) == 45 and len(masks) == 12 and len(trajectories) == 112
    output.mkdir()
    for name, rows in (("aggregate", summary), ("trajectories", trajectories), ("contrasts", contrasts),
                       ("trajectory_contrasts", paired), ("mask_overlap", overlaps),
                       ("decision_audit", decisions)):
        write_table(output / f"{name}.csv", rows)
    print("PASS: 45 decisions, 360 candidates, 12 exact tensor projections, 224 complete windows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("collected", "analysis", "dense_checkpoint", "output"):
        parser.add_argument(arg, type=Path)
    args = parser.parse_args()
    audit(args.collected, args.analysis, args.dense_checkpoint, args.output)

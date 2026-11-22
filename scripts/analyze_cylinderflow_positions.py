"""Validate paired windows, summarize position, stage and lagged-velocity experiments."""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median

import numpy as np


def read(path):
    return json.loads(path.read_text())


def export(folder, name, rows):
    with (folder / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(folder):
    windows = {p.stem: read(p) for p in sorted(folder.glob("[0-9]*-*.json"))}
    summary = read(folder / "summary.json")
    assert len(windows) == 16 and all(w.keys() == summary.keys() for w in windows.values())
    rows = []
    for model, official in summary.items():
        curves = [w[model]["curve"] for w in windows.values()]
        assert all(len(c) == 20 for c in curves)
        complete = sum(None not in c for c in curves)
        assert complete == official["windows"]
        for metric, step in (("E1", 0), ("E20", -1), ("mean20", None)):
            if complete == len(curves):
                value = mean(mean(c) if step is None else c[step] for c in curves)
                assert math.isclose(value, official[metric], rel_tol=1e-12)
        times = [t for w in windows.values() for t in w[model]["seconds"]]
        assert len(times) == official["samples"] == 24
        assert median(times) == official["T20"]
        # Each trajectory has its own mesh. Pair median times on the same mesh.
        paired = [median(w[model]["seconds"])/median(w["dense"]["seconds"])
                  for w in windows.values() if w[model]["seconds"]]
        rows.append(dict(model=model, **official, paired_time_pct_dense=100*median(paired),
                         paired_time_pct_dense_q1=100*np.quantile(paired, .25),
                         paired_time_pct_dense_q3=100*np.quantile(paired, .75)))
    return windows, rows


def analyze(artifact_root, output):
    out = artifact_root / "meshgraphnet-processor-positions-20261122/collected/results"
    assert read(out / "finish.json") == dict(locations=16, history=4, diagnostic_windows=16, confirmation_windows=16)
    protocol = read(out / "protocol.json")
    sweep, locations = summarize(out / "locations")
    history, corrections = summarize(out / "history")
    diagnostic = {int(k.split("-")[0]) for k in sweep}
    fresh = {int(k.split("-")[0]) for k in history}
    assert diagnostic == set(protocol["splits"]["diagnostic"]) and fresh == set(protocol["confirmation"])
    assert fresh.isdisjoint(diagnostic | set(protocol["splits"]["baseline"]) | set(protocol["splits"]["selection"]))
    old = {
        "dense": ("meshgraphnet-block-distillation-20261121", "A"),
        "7-8/skip": ("meshgraphnet-linear-block-pilot-20261120", "B"),
        "7-8/affine": ("meshgraphnet-linear-block-pilot-20261120", "C"),
        "7-8/exchange": ("meshgraphnet-linear-exchange-20261120", "D"),
        "7-8/remove-one": ("meshgraphnet-block-distillation-20261121", "before"),
        "7-8/distilled": ("meshgraphnet-block-distillation-extended-20261121", "rollout-125"),
    }
    for key, (campaign, model) in old.items():
        for window, w in sweep.items():
            archived = read(artifact_root / campaign / "collected/results" / model / (window+".json"))
            np.testing.assert_allclose(w[key]["curve"], archived["curve"][:20], rtol=.005, atol=1e-8)
    stages = read(out / "profile.json")
    assert len(stages) == 40 and {s["trajectory"] for s in stages} == diagnostic
    ms = {key: mean(r["ms"][key] for r in stages) for key in stages[0]["ms"]}
    total = sum(ms.values())
    stage_rows = [dict(stage=k, mean_ms=v, percent_of_profile=100*v/total) for k,v in ms.items()]
    output.mkdir(exist_ok=True, parents=True)
    export(output, "locations.csv", locations)
    export(output, "history.csv", corrections)
    export(output, "profile.csv", stage_rows)
    paired = []
    for trajectory in sorted(fresh):
        rows = [w for name,w in history.items() if int(name.split("-")[0]) == trajectory]
        paired.append(dict(trajectory=trajectory, **{k:mean(mean(r[k]["curve"]) for r in rows)
                                                       for k in rows[0]}))
    export(output, "history_by_trajectory.csv", paired)
    print("stage total and processor ms", total, sum(v for k,v in ms.items() if k.startswith("block_")))
    print("history improves over scale on", sum(p["history"] < p["scale"] for p in paired), "of", len(paired))
    for filename in ("locations.csv", "history.csv", "profile.csv"):
        print((output / filename).read_text())


if __name__ == "__main__":
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("artifact_root", type=Path)
    args.add_argument("output", type=Path)
    analyze(**vars(args.parse_args()))

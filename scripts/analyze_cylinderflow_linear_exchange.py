"""Summarize the frozen, paired A/C/D CylinderFlow experiment from POLUS."""

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
from statistics import mean, median

import numpy as np


def read(path):
    return json.loads(path.read_text())


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(source, previous, out):
    out.mkdir(parents=True, exist_ok=True)
    protocol = read(source / "protocol.json")
    calibration = read(source / "calibration.json")
    assert hashlib.sha256((source / "exchange.pt").read_bytes()).hexdigest() == calibration["maps_sha256"]
    assert calibration["trajectories"] == read(previous / "calibration.json")["trajectory_ids"]
    assert calibration["fits"]["edge"]["rows"] == 32768 and calibration["fits"]["node"]["rows"] == 8192
    ids = protocol["splits"]["diagnostic"]
    starts = protocol["splits"]["starts"]
    labels = ("A", "C", "D")
    finish = read(source / "finish.json")
    assert (len(ids), len(starts), finish["windows"], finish["timing_samples"]) == (8, 2, 48, 288)
    curves = {label: [] for label in labels}
    timing = {label: [] for label in labels}
    pairs = (("D", "C"), ("D", "A"), ("C", "A"))
    ratios = {pair: [] for pair in pairs}
    timedeltas = {pair: [] for pair in pairs}
    comparisons = {label: 0. for label in ("A", "C")}
    trajectory_rows, failures = [], []
    for ordinal in ids:
        one = {"trajectory": ordinal}
        for label in labels:
            saved = []
            for start in starts:
                name = f"{ordinal}-{start}.json"
                record = read(source / label / name)
                if record["status"] != "ok":
                    failures.append(dict(variant=label, trajectory=ordinal, start=start,
                                         step=record["first_failed_step"], status=record["status"]))
                    saved.append(np.full(100, np.nan))
                    continue
                curve = np.array(record["curve"])
                assert curve.shape == (100,) and np.isfinite(curve).all()
                assert np.isclose(curve.mean(), record["mean100"]) and curve[-1] == record["E100"]
                if label in comparisons:
                    old = np.array(read(previous / label / name)["curve"])
                    np.testing.assert_allclose(curve, old, rtol=.005, atol=1e-8)
                    comparisons[label] = max(comparisons[label], float(np.max(abs(curve-old))))
                saved.append(curve)
            average = np.mean(saved, axis=0)
            curves[label].append(average)
            one[f"{label}_E100"] = average[-1]
            one[f"{label}_mean100"] = average.mean()
        trajectory_rows.append(one)
        for start in starts:
            rows = read(source / "timing" / f"{ordinal}-{start}.json")
            assert len(rows) == 18
            orders = set()
            for repeat in range(6):
                group = rows[3*repeat:3*repeat+3]
                orders.add(tuple(item["variant"] for item in group))
                values = {}
                for item in group:
                    assert item["repeat"] == repeat and item["seconds"] > 0
                    values[item["variant"]] = item["seconds"]
                    timing[item["variant"]].append(item["seconds"])
                for left, right in pairs:
                    ratios[left,right].append(values[right] / values[left])
                    timedeltas[left,right].append(values[left] - values[right])
            assert orders == set(itertools.permutations(labels))
    summary = []
    for label in labels:
        curve = np.mean(curves[label], axis=0)
        summary.append(dict(model=label, params=finish["parameters"][label],
                            E1=curve[0], E5=curve[4], E20=curve[19], E50=curve[49],
                            E100=curve[-1], mean100=curve.mean(), T100_s=median(timing[label])))
    contrasts = []
    draws = np.random.default_rng(101).integers(0, len(ids), (10000,len(ids)))
    for left, right in pairs:
        delta = np.array(curves[left])[:, -1] - np.array(curves[right])[:, -1]
        low, high = np.quantile(delta[draws].mean(axis=1), [.025,.975])
        delta_mean = np.array(curves[left]).mean(axis=1) - np.array(curves[right]).mean(axis=1)
        mean_low, mean_high = np.quantile(delta_mean[draws].mean(axis=1), [.025,.975])
        speed = ratios[left,right]
        contrasts.append(dict(left=left, right=right, dE100=delta.mean(), ci_low=low,
                              ci_high=high, E100_wins=int((delta < 0).sum()), dMean100=delta_mean.mean(),
                              mean_ci_low=mean_low, mean_ci_high=mean_high,
                              mean100_wins=int((delta_mean < 0).sum()),
                              paired_speedup=median(speed), paired_seconds=median(timedeltas[left,right]),
                              faster_pairs=sum(s>1 for s in speed), pairs=len(speed)))
    write_csv(out / "summary.csv", summary)
    write_csv(out / "contrasts.csv", contrasts)
    write_csv(out / "trajectories.csv", trajectory_rows)
    write_csv(out / "curves.csv", [dict(step=i+1, **{k: mean(c[i] for c in curves[k]) for k in labels})
                                  for i in range(100)])
    from analyze_cylinderflow_mask_diagnostic import plt
    fig, ax = plt.subplots(figsize=(7.5, 4.3), layout="constrained")
    names = {"A": "A · Dense", "C": "C · Независимые карты", "D": "D · Обмен по рёбрам"}
    for label, color in (("A", "#535964"), ("C", "#b24a23"), ("D", "#006b79")):
        ax.plot(range(1, 101), np.mean(curves[label], axis=0), label=names[label],
                lw=2, color=color)
    ax.set(xlabel="Шаг прогноза", ylabel="MSE поля скорости", yscale="log", xlim=(1, 100))
    ax.grid(axis="y", alpha=.25)
    ax.legend()
    fig.savefig(out / "rollout-error.png", dpi=180)
    plt.close(fig)
    (out / "verification.json").write_text(json.dumps(dict(job=protocol["job"],
        archive_max_abs_curve_diff=comparisons, failures=failures,
        nonfinite_timings=sum(not item["final_finite"] for p in (source/"timing").glob("*.json")
                              for item in read(p))), indent=2)+"\n")
    print(summary, contrasts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("source", "previous", "out"):
        parser.add_argument(arg, type=Path)
    analyze(**vars(parser.parse_args()))

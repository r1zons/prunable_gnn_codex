"""Verify collected POLUS results and aggregate the paired CylinderFlow pilot."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import statistics

import numpy as np

from analyze_cylinderflow_mask_diagnostic import csv_write, plt


LABELS = ("A", "B", "C")
PAIRS = (("B", "A"), ("C", "A"), ("C", "B"))


def read(path):
    """Read a saved JSON record."""
    return json.loads(path.read_text())


def timing_summary(values):
    """Return median, quartiles and extrema of saved timing samples."""
    return dict(median=float(np.median(values)), q1=float(np.quantile(values, .25)),
                q3=float(np.quantile(values, .75)), low=float(min(values)), high=float(max(values)))


def analyze(source, archived, output):
    """Verify the dense reference and preserve pairing by trajectory, window and repeat."""
    protocol = read(source / "protocol.json")
    finish = read(source / "finish.json")
    calibration = read(source / "calibration.json")
    assert hashlib.sha256((source / "maps.pt").read_bytes()).hexdigest() == calibration["maps_sha256"]
    assert calibration["reload_output_exact"]
    assert len(calibration["trajectory_ids"]) == 8
    assert len(calibration["samples"]) == 32
    assert {name: fit["rows"] for name, fit in calibration["fits"].items()} == {"node": 8192, "edge": 32768}
    assert protocol["splits"]["split"] == "valid"
    ids = protocol["splits"]["diagnostic"]
    starts = protocol["splits"]["starts"]
    assert ids == [87, 19, 72, 15, 18, 80, 79, 88] and starts == [100, 300]
    assert read(source / "dense-gate.json")["passed"]
    checks = read(source / "gpu-checks.json")
    assert [sum(checks["calls"][label]) for label in LABELS] == [15, 13, 14]
    assert set(finish["state_sha256"]) == set(LABELS)
    assert finish["state_sha256"]["A"] == protocol["dense_tensor_sha256"]
    expected = {f"{ordinal}-{start}.json" for ordinal, start in itertools.product(ids, starts)}
    for label in (*LABELS, "timing"):
        assert {p.name for p in (source / label).glob("*.json")} == expected
    assert finish["windows"] == 48 and finish["timing_samples"] == 288
    timings = {label: [] for label in LABELS}
    speedups = {pair: [] for pair in PAIRS}
    curves = {label: {} for label in LABELS}
    raw_timing, windows, trajectories = [], [], []
    maximum_dense_difference = 0.
    for ordinal in ids:
        per_trajectory = dict(trajectory=ordinal)
        for label in LABELS:
            saved = []
            for start in starts:
                path = f"{ordinal}-{start}.json"
                row = read(source / label / path)
                curve = np.asarray(row["curve"])
                ok = row["status"] == "ok"
                assert ok == (len(curve) == row["forward_calls"] == 100 and row["first_failed_step"] is None)
                assert np.isfinite(curve).all() and (curve >= 0).all()
                if ok:
                    assert row["E100"] == curve[-1]
                    np.testing.assert_allclose(row["mean100"], curve.mean(), rtol=1e-14)
                else:
                    assert row["E100"] is None and row["mean100"] is None
                if label == "A":
                    previous = read(archived / "dense" / path)
                    assert previous["status"] == "ok" and ok
                    maximum_dense_difference = max(maximum_dense_difference,
                                                   float(np.max(np.abs(curve - previous["curve"]))))
                window = dict(variant=label, trajectory=ordinal, start=start, status=row["status"],
                              first_failed_step=row["first_failed_step"], error=row["error"],
                              E100=row["E100"], mean100=row["mean100"])
                windows.append(window)
                saved.append(curve if ok else np.full(100, np.nan))
            by_trajectory = np.mean(saved, axis=0)
            curves[label][ordinal] = by_trajectory
            per_trajectory[f"{label}_E100"] = by_trajectory[-1]
            per_trajectory[f"{label}_mean100"] = by_trajectory.mean()
        for left, right in PAIRS:
            per_trajectory[f"{left}_minus_{right}_E100"] = (
                per_trajectory[f"{left}_E100"] - per_trajectory[f"{right}_E100"])
        for start in starts:
            samples = read(source / "timing" / f"{ordinal}-{start}.json")
            assert len(samples) == 18
            groups = [samples[i:i + 3] for i in range(0, 18, 3)]
            assert {tuple(row["variant"] for row in group) for group in groups} == set(itertools.permutations(LABELS))
            for repeat, group in enumerate(groups):
                seconds = {}
                for position, row in enumerate(group):
                    assert row["repeat"] == repeat and np.isfinite(row["seconds"]) and row["seconds"] > 0
                    raw_timing.append(dict(trajectory=ordinal, start=start, position=position, **row))
                    seconds[row["variant"]] = row["seconds"]
                    timings[row["variant"]].append(row["seconds"])
                for left, right in PAIRS:
                    speedups[left, right].append(seconds[right] / seconds[left])
        for pair, ratios in speedups.items():
            per_trajectory[f"{pair[0]}_vs_{pair[1]}_T100_speedup"] = statistics.median(ratios[-12:])
        trajectories.append(per_trajectory)

    average_curves = {label: np.mean(list(items.values()), axis=0) for label, items in curves.items()}
    rng = np.random.default_rng(101)
    indices = rng.integers(len(ids), size=(10000, len(ids)))
    contrasts = []
    for left, right in PAIRS:
        differences = np.array([row[f"{left}_minus_{right}_E100"] for row in trajectories])
        complete = np.isfinite(differences).all()
        intervals = np.quantile(differences[indices].mean(axis=1), [.025, .975]) if complete else (None, None)
        complete_count = int(np.isfinite(differences).sum())
        paired_timing = timing_summary(speedups[left, right])
        contrasts.append(dict(variant=left, reference=right, paired_trajectories=complete_count,
                              E100_delta=float(differences.mean()) if complete else None,
                              lower=intervals[0], upper=intervals[1],
                              E100_wins=int((differences < 0).sum()),
                              E100_losses=int((differences > 0).sum()),
                              T100_faster_pairs=sum(ratio > 1 for ratio in speedups[left, right]),
                              **{f"speedup_{key}": value for key, value in paired_timing.items()}))
    rows = []
    for label, curve in average_curves.items():
        e100 = float(curve[-1]) if np.isfinite(curve[-1]) else None
        rows.append(dict(variant=label, T100_samples=len(timings[label]),
                         **{f"T100_s_{k}": v for k, v in timing_summary(timings[label]).items()},
                         paired_median_speedup=statistics.median(speedups[label, "A"]) if label != "A" else 1.,
                         E100=e100, E100_relative_to_A=e100 / average_curves["A"][-1] - 1 if e100 is not None else None,
                         mean100=float(curve.mean()) if e100 is not None else None,
                         failed_windows=sum(r["status"] != "ok" for r in windows if r["variant"] == label),
                         nonfinite_timing_runs=sum(not r["final_finite"] for r in raw_timing if r["variant"] == label)))

    output.mkdir()
    csv_write(output / "summary.csv", rows)
    csv_write(output / "trajectories.csv", trajectories)
    csv_write(output / "contrasts.csv", contrasts)
    csv_write(output / "windows.csv", windows)
    csv_write(output / "raw_timing.csv", raw_timing)
    figure, axes = plt.subplots(figsize=(7, 4.1), layout="constrained")
    for label, caption in zip(LABELS, ("A · Dense", "B · Skip", "C · Affine replacement")):
        axes.plot(range(1, 101), average_curves[label], label=caption)
    axes.set(xlabel="Autoregressive forecast step", ylabel="Velocity MSE", yscale="log", xlim=(1, 100))
    axes.grid(alpha=.2)
    axes.legend()
    figure.savefig(output / "rollout-error.png", dpi=180)
    plt.close(figure)
    verification = dict(job=protocol["job"], comparison_windows=48, timed_rollouts=288,
                        max_abs_curve_difference_dense_vs_archive=maximum_dense_difference,
                        calibration_seconds=calibration["seconds"], selected_train_ids=calibration["trajectory_ids"],
                        parameter_counts=checks["parameter_counts"], source=protocol["source"])
    (output / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "archived", "output"):
        parser.add_argument(name, type=Path)
    analyze(**vars(parser.parse_args()))

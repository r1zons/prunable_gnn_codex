"""Check archived controls and aggregate the POLUS block-distillation experiment."""

import argparse
from collections import Counter
import hashlib
import json
import itertools
import math
from pathlib import Path
import statistics

import numpy as np

from analyze_cylinderflow_linear_exchange import read, write_csv
from analyze_cylinderflow_mask_diagnostic import plt


LABELS = ("A", "D", "before", "after")
PAIRS = (("D", "A"), ("before", "A"), ("after", "A"), ("after", "before"), ("after", "D"))
MODES = ("rollout", "reference")


def analyze(source, archive, out):
    """Audit and summarize all saved windows, keeping both starts paired."""
    protocol = read(source / "protocol.json")
    distill = read(source / "distillation.json")
    cfg, splits = protocol["config"], protocol["splits"]
    ids, starts = splits["diagnostic"], splits["starts"]
    steps = cfg["horizon"]
    assert len(ids) == 8 and starts == [100, 300] and steps == 100
    assert splits["split"] == "valid" and len(protocol["train_ids"]) == 8
    assert len(protocol["heldout_train_ids"]) == 4
    assert set(protocol["train_ids"]).isdisjoint(protocol["heldout_train_ids"])
    assert distill["initial_outputs_exact"] and distill["reload_output_exact"]
    for file, key in (("before-model.pt", "before_file_sha256"), ("distilled-block.pt", "block_file_sha256")):
        assert hashlib.sha256((source / file).read_bytes()).hexdigest() == distill[key]
    feature_history = [read(source / "features" / f"epoch-{i:02}.json") for i in range(cfg["epochs"] + 1)]
    chosen = min(feature_history, key=lambda row: row["heldout"]["equal_weight_mse"])
    assert chosen == distill["after_features"] and chosen["epoch"] == distill["chosen_epoch"]
    assert distill["before_features"] == feature_history[0]
    checks = read(source / "gpu-checks.json")
    finish = read(source / "finish.json")
    assert finish["state_sha256"] == checks["state_sha256"]
    assert checks["state_sha256"]["A"] == protocol["dense_tensor_sha256"]
    assert [len(checks["processor_calls"][label]) for label in LABELS] == [15, 14, 14, 14]
    assert checks["processor_calls"]["before"] == checks["processor_calls"]["after"] == ["Processor"] * 14
    assert checks["parameters"]["before"] == checks["parameters"]["after"]
    assert (finish["windows"], finish["reference_step_windows"], finish["timing_samples"]) == (64, 64, 256)

    curves = {mode: {label: [] for label in LABELS} for mode in MODES}
    raw_times, paired_rows, trajectories, failures = [], [], [], []
    paired_times = {pair: [] for pair in PAIRS}
    timings = {label: [] for label in LABELS}
    difference = dict(A=0., D=0.)
    for ordinal in ids:
        row = dict(trajectory=ordinal)
        for mode, label in itertools.product(MODES, LABELS):
            folder = source if mode == "rollout" else source / "reference_steps"
            windows = []
            for start in starts:
                name = f"{ordinal}-{start}.json"
                path = folder / label / name
                result = read(path) if path.exists() else dict(status="missing", first_failed_step=None, curve=[])
                curve = np.array(result["curve"])
                assert np.isfinite(curve).all() and (curve >= 0).all()
                if result["status"] == "ok":
                    assert len(curve) == steps and result["first_failed_step"] is None
                    if mode == "rollout":
                        assert curve[-1] == result["E100"] and np.isclose(curve.mean(), result["mean100"])
                else:
                    failures.append(dict(mode=mode, variant=label, trajectory=ordinal, start=start,
                                         status=result["status"], first_failed_step=result["first_failed_step"]))
                    curve = np.full(steps, np.nan)
                if mode == "rollout" and label in difference:
                    archived = read(archive / label / name)["curve"]
                    np.testing.assert_allclose(curve, archived, rtol=cfg["reference_curve_rtol"],
                                               atol=cfg["reference_curve_atol"])
                    difference[label] = max(difference[label], float(np.max(abs(curve-archived))))
                windows.append(curve)
            mean = np.mean(windows, axis=0)
            curves[mode][label].append(mean)
            row[f"{label}_{mode}_mean100"] = mean.mean()
            row[f"{label}_{mode}_last"] = mean[-1]
        for mode in MODES:
            for key in ("mean100", "last"):
                row[f"after_minus_before_{mode}_{key}"] = row[f"after_{mode}_{key}"] - row[f"before_{mode}_{key}"]
        trajectories.append(row)
        for start in starts:
            samples = read(source / "timing" / f"{ordinal}-{start}.json")
            assert len(samples) == 16
            orders = []
            for repeat in range(4):
                group = samples[4*repeat:4*repeat+4]
                orders.append([r["variant"] for r in group])
                values = {}
                for position, item in enumerate(group):
                    assert item["repeat"] == repeat and math.isfinite(item["seconds"]) and item["seconds"] > 0
                    values[item["variant"]] = item["seconds"]
                    timings[item["variant"]].append(item["seconds"])
                    raw_times.append(dict(trajectory=ordinal, start=start, position=position, **item))
                for left, right in PAIRS:
                    ratio, delta = values[right] / values[left], values[left] - values[right]
                    paired_times[left, right].append((ratio, delta))
                    paired_rows.append(dict(trajectory=ordinal, start=start, repeat=repeat, variant=left,
                                            reference=right, speedup=ratio, delta_seconds=delta))
            assert all(set(order) == set(LABELS) for order in orders)
            assert all(set(column) == set(LABELS) for column in zip(*orders))
            assert set(Counter(pair for order in orders for pair in zip(order, order[1:])).values()) == {1}

    means = {mode: {k: np.mean(v, axis=0) for k, v in data.items()} for mode, data in curves.items()}
    summary, contrasts = [], []
    for label in LABELS:
        t = timings[label]
        q1, q3 = np.quantile(t, [.25, .75])
        r = means["rollout"][label]
        summary.append(dict(variant=label, parameters=checks["parameters"][label],
                            T100_s=statistics.median(t), T100_q1=q1, T100_q3=q3,
                            T100_min=min(t), T100_max=max(t),
                            paired_speedup_vs_dense=1. if label == "A" else np.median(np.array(paired_times[label, "A"])[:, 0]),
                            E1=r[0], E100=r[-1], E100_relative_vs_dense=r[-1]/means["rollout"]["A"][-1]-1,
                            mean100=r.mean(), reference_step_mean100=means["reference"][label].mean()))
    rng = np.random.default_rng(101)
    resample = rng.integers(0, len(ids), (10000, len(ids)))
    for left, right in PAIRS:
        ratio, timing_delta = np.array(paired_times[left, right]).T
        stats = dict(variant=left, reference=right, paired_speedup=np.median(ratio),
                     speedup_q1=np.quantile(ratio,.25), speedup_q3=np.quantile(ratio,.75),
                     delta_seconds=np.median(timing_delta), faster_pairs=int((ratio>1).sum()))
        for field in ("rollout_last", "rollout_mean100", "reference_mean100"):
            delta = np.array([r[f"{left}_{field}"] - r[f"{right}_{field}"] for r in trajectories])
            finite = np.isfinite(delta)
            low, high = np.quantile(delta[resample].mean(axis=1), [.025, .975]) if finite.all() else (None, None)
            stats.update({f"{field}_delta": delta.mean() if finite.all() else None,
                          f"{field}_ci_low": low, f"{field}_ci_high": high,
                          f"{field}_wins": int((delta < 0).sum()), f"{field}_paired_n": int(finite.sum())})
        contrasts.append(stats)
    out.mkdir()
    for name, rows in (("summary", summary), ("contrasts", contrasts), ("trajectories", trajectories),
                       ("raw-timing", raw_times), ("paired-timing", paired_rows), ("feature-losses", [
                           dict(epoch=r["epoch"], updates=r["updates"], **{
                               f"{group}_{key}": value for group in ("train", "heldout")
                               for key, value in r[group].items()}) for r in feature_history
                       ])):
        write_csv(out / f"{name}.csv", rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained", sharex=True)
    for mode, ax, title in zip(MODES, axes, ("Автономный прогноз", "Эталон на входе каждого шага")):
        for label, caption in zip(LABELS, ("Dense", "Линейный обмен D", "До обучения", "После обучения")):
            ax.plot(range(1, steps+1), means[mode][label], label=caption)
        ax.set(title=title, xlabel="Шаг", ylabel="MSE скорости", yscale="log")
        ax.grid(alpha=.2)
    axes[0].legend()
    fig.savefig(out / "rollout-error.png", dpi=175)
    plt.close(fig)
    (out / "verification.json").write_text(json.dumps(dict(
        job=protocol["job"], dense_and_D_max_abs_curve_delta=difference, failures=failures,
        nonfinite_timing=sum(not r["final_finite"] for r in raw_times),
        source=protocol["source"]), indent=2)+"\n")
    print("Audited 128 windows, 256 timings and held-out checkpoint selection.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("source", "archive", "out"):
        parser.add_argument(arg, type=Path)
    analyze(**vars(parser.parse_args()))

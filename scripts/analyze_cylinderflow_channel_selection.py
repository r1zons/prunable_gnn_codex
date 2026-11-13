"""Summarize the development comparison from sealed records, without model access."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from analyze_cylinderflow_mask_diagnostic import KEYS, csv_write, metrics, nullable, plt


def read(path):
    return json.loads(path.read_text())


def analyze(source, output):
    protocol = read(source / "protocol.json")
    cfg, splits = protocol["config"], protocol["splits"]
    windows = list(itertools.product(splits["diagnostic"], splits["starts"]))
    strategies = list(cfg["selection_horizons"])
    levels = [d * cfg["removals_per_decision"] for d in cfg["snapshot_decisions"]]
    entries = ["dense", "persistence", *(f"{s}-{n:02}" for n in levels for s in strategies)]
    values, rows, summary, overlaps, contrasts = {}, [], [], [], []
    for name in entries:
        folder = source / name
        for ordinal, start in windows:
            path = folder / f"{ordinal}-{start}.json"
            record = read(path) if path.exists() else dict(status="missing", curve=[])
            curve = record["curve"]
            assert len(curve) <= cfg["diagnostic_horizon"] and np.isfinite(curve).all()
            assert (record["status"] == "ok") == (len(curve) == cfg["diagnostic_horizon"])
            value = metrics(curve)
            key = ordinal, start
            values[name, key] = value
            rows.append(dict(model=name, ordinal=ordinal, start=start, status=record["status"],
                             first_failed_step=record.get("first_failed_step"),
                             forward_calls=record.get("forward_calls"),
                             seconds=record.get("elapsed_seconds"),
                             peak_allocated_bytes=record.get("peak_allocated_bytes"),
                             peak_reserved_bytes=record.get("peak_reserved_bytes"), **nullable(value),
                             **{f"delta_dense_{k}": v for k, v in
                                nullable(value - values["dense", key]).items()}))
        surgery = folder / "surgery.json"
        info = read(surgery) if surgery.exists() else {}
        timing_path = folder / "timing.json"
        timing = read(timing_path) if timing_path.exists() else {}
        n = info.get("decision", 0)
        decisions = [read(source / name.rsplit("-", 1)[0] / f"decision-{i:02}.json")
                     for i in range(1, n + 1)]
        group = [r for r in rows if r["model"] == name]
        summary.append(dict(model=name, parameters=info.get("parameters", 2332930 if name == "dense" else None),
                            compression=info.get("compression", 0 if name == "dense" else None),
                            widths=json.dumps(info.get("widths", [[128, 128]] * 15 if name == "dense" else None)),
                            search_seconds=sum(d["search_seconds"] for d in decisions) if n else None,
                            candidates=sum(d["candidate_evaluations"] for d in decisions) if n else None,
                            model_candidates=sum(d["model_candidate_evaluations"] for d in decisions) if n else None,
                            search_forwards=sum(d["forward_calls"] for d in decisions) if n else None,
                            diagnostic_forwards=sum(r["forward_calls"] or 0 for r in group),
                            inference_mean_seconds=timing.get("mean_seconds"),
                            inference_sd_seconds=timing.get("sd_seconds"),
                            inference_host=timing.get("host"), timing_peak_allocated_bytes=timing.get("peak_allocated_bytes"),
                            timing_peak_reserved_bytes=timing.get("peak_reserved_bytes"),
                            rollout_peak_allocated_bytes=max((r["peak_allocated_bytes"] or 0 for r in group), default=0) or None,
                            rollout_peak_reserved_bytes=max((r["peak_reserved_bytes"] or 0 for r in group), default=0) or None,
                            ok=sum(r["status"] == "ok" for r in group),
                            failed=sum(r["status"] not in {"ok", "missing"} for r in group),
                            missing=sum(r["status"] == "missing" for r in group),
                            **nullable(np.stack([values[name, w] for w in windows]).mean(0))))

    sample = np.random.default_rng(cfg["seed"]).integers(
        len(splits["diagnostic"]), size=(cfg["bootstrap_repeats"], len(splits["diagnostic"])))
    by_trajectory = {
        name: np.stack([values[name, w] for w in windows])
        .reshape(len(splits["diagnostic"]), len(splits["starts"]), len(KEYS)).mean(1)
        for name in entries
    }

    def compare(name, reference, index):
        delta = (by_trajectory[name] - by_trajectory[reference])[:, index]
        complete = np.isfinite(delta).all()
        interval = np.quantile(delta[sample].mean(1), [.025, .975]) if complete else [None, None]
        contrasts.append(dict(model=name, reference=reference, metric=KEYS[index],
                              delta=float(delta.mean()) if complete else None,
                              lower=interval[0], upper=interval[1],
                              paired_trajectories=int(np.isfinite(delta).sum())))

    for name in entries[1:]:
        for i in range(len(KEYS)):
            compare(name, "dense", i)
        if name != "persistence":
            compare(name, "persistence", KEYS.index("E100"))
    for level in levels:
        compare(f"five_step-{level:02}", f"one_step-{level:02}", KEYS.index("E100"))
        for a, b in itertools.combinations(strategies, 2):
            labels = [f"{s}-{level:02}" for s in (a, b)]
            files = [source / name / "surgery.json" for name in labels]
            if not all(p.exists() for p in files):
                overlaps.append(dict(level=level, a=a, b=b, removed_intersection=None,
                                     removed_jaccard=None, retained_jaccard=None, identical_mlps=None))
                continue
            left, right = [list(map(set, read(p)["removed"])) for p in files]
            inter = sum(len(x & y) for x, y in zip(left, right))
            union = sum(len(x | y) for x, y in zip(left, right))
            total = len(left) * 128
            overlaps.append(dict(level=level, a=a, b=b, removed_intersection=inter,
                                 removed_jaccard=inter / union,
                                 retained_jaccard=(total - union) / (total - inter),
                                 identical_mlps=sum(x == y for x, y in zip(left, right))))

    output.mkdir()
    for filename, records in [("windows", rows), ("summary", summary),
                              ("contrasts", contrasts), ("overlap", overlaps)]:
        csv_write(output / f"{filename}.csv", records)
    fig, ax = plt.subplots()
    for name, color in [("dense", "0.3"), ("persistence", "0.6")]:
        ax.axhline(by_trajectory[name][:, 4].mean(), label=name, color=color, linestyle="--")
    for strategy in strategies:
        ax.plot(levels, [by_trajectory[f"{strategy}-{n:02}"][:, 4].mean() for n in levels],
                marker="o", label=strategy)
    ax.set(xlabel="Removed channels per processor MLP", ylabel="E100")
    ax.set_xticks(levels)
    ax.legend()
    fig.savefig(output / "e100.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    lines = ["# CylinderFlow channel selection", "",
             "Development comparison on a reused diagnostic subset and one frozen checkpoint.",
             "Intervals are exploratory paired percentile bootstrap intervals over whole trajectories.",
             "Both windows and all variants share the same trajectory resamples.",
             "An aggregate is withheld if any required window is missing for that metric.",
             "Search cost is cumulative to each snapshot. GPU timing uses the largest diagnostic mesh.",
             "", "| Model | E100 | Δ E100 vs dense | 95% interval | OK / failed / missing |",
             "|---|---:|---:|---:|---:|"]
    def format_value(value):
        return "—" if value is None else f"{value:.6g}"

    for row in summary:
        delta = next((c for c in contrasts if c["model"] == row["model"]
                      and c["reference"] == "dense" and c["metric"] == "E100"), {})
        lines.append(f"| {row['model']} | {format_value(row['E100'])} | {format_value(delta.get('delta'))} | "
                     f"{format_value(delta.get('lower'))} … {format_value(delta.get('upper'))} | "
                     f"{row['ok']} / {row['failed']} / {row['missing']} |")
    lines += ["", "| Removed channels per MLP | Δ E100 five-step vs one-step | 95% interval |",
              "|---|---:|---:|"]
    for level in levels:
        delta = next(c for c in contrasts if c["model"] == f"five_step-{level:02}"
                     and c["reference"] == f"one_step-{level:02}")
        lines.append(f"| {level} | {format_value(delta['delta'])} | "
                     f"{format_value(delta['lower'])} … {format_value(delta['upper'])} |")
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    analyze(args.source, args.output)

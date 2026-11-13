"""Analyze fixed mask rankings after collecting the POLUS window records."""

import argparse
import csv
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import rankdata


KEYS = ("E1", "E5", "E20", "E50", "E100", "mean5", "mean20", "mean100")
SHORT = (0, 5)
LABELS = ("E1", "Средняя MSE, шаги 1–5")
ATOL, RTOL = 1e-9, 1e-4


def metrics(curve):
    result = []
    for h in (1, 5, 20, 50, 100):
        result.append(curve[h-1] if len(curve) >= h else np.nan)
    for h in (5, 20, 100):
        result.append(np.mean(curve[:h]) if len(curve) >= h else np.nan)
    return np.asarray(result)


def direction(a, b):
    tolerance = ATOL + RTOL * np.maximum(np.abs(a), np.abs(b))
    return np.where(np.abs(a-b) > tolerance, np.sign(a-b), 0)


def comparisons(values, baseline):
    """Return exact-tie Spearman, tolerance-aware inversions and paired sign flips."""
    result = {}
    for label, short in zip(("E1", "mean5"), SHORT):
        x, y = values[..., short], values[..., 4]
        ranks = [rankdata(z, axis=-1) for z in (x, y)]
        a, b = (rank - rank.mean(axis=-1, keepdims=True) for rank in ranks)
        denom = np.sqrt((a*a).sum(axis=-1) * (b*b).sum(axis=-1))
        rho = np.divide((a*b).sum(axis=-1), denom, out=np.full_like(denom, np.nan),
                        where=denom > 0)
        pairs = np.triu_indices(x.shape[-1], 1)
        dx = direction(x[..., pairs[0]], x[..., pairs[1]])
        dy = direction(y[..., pairs[0]], y[..., pairs[1]])
        comparable = (dx != 0) & (dy != 0)
        inverted = (dx*dy < 0).sum(axis=-1)
        n = comparable.sum(axis=-1)
        short_sign = direction(x, baseline[..., short, None])
        long_sign = direction(y, baseline[..., 4, None])
        result[label] = dict(spearman=rho, inversions=inverted, comparable_pairs=n,
                             inversion_fraction=np.divide(inverted, n, out=np.full_like(rho, np.nan), where=n>0),
                             short_better_long_worse=((short_sign<0)&(long_sign>0)).sum(axis=-1),
                             short_worse_long_better=((short_sign>0)&(long_sign<0)).sum(axis=-1))
    return result


def csv_write(path, rows):
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def nullable(values):
    return {key: float(v) if np.isfinite(v) else None for key, v in zip(KEYS, values)}


def analyze(source, out):
    protocol = json.loads((source / "protocol.json").read_text())
    splits = json.loads((source / "splits.json").read_text())
    library = json.loads((source / "masks.json").read_text())["masks"]
    entries = [dict(id="dense", stratum="dense"), dict(id="persistence", stratum="persistence"), *library]
    ids, starts = splits["diagnostic"], splits["starts"]
    shape = (len(entries), len(ids), len(starts))
    curves = np.full((*shape, 100), np.nan)
    per_window, per_trajectory, summary = [], [], []
    for i, entry in enumerate(entries):
        name = entry["id"]
        for j, k in itertools.product(range(len(ids)), range(len(starts))):
            path = source / name / f"{ids[j]}-{starts[k]}.json"
            row = json.loads(path.read_text()) if path.exists() else dict(status="missing", curve=[])
            curve = row["curve"]
            assert len(curve) <= 100 and np.isfinite(curve).all()
            assert (row["status"] == "ok") == (len(curve) == 100)
            curves[i,j,k,:len(curve)] = curve
            val = metrics(curve)
            base = metrics(curves[0,j,k])
            per_window.append(dict(mask=name, stratum=entry["stratum"], trajectory=ids[j],
                                   start=starts[k], status=row["status"], first_failed_step=row.get("first_failed_step"),
                                   **nullable(val), **{"delta_"+key: v for key,v in nullable(val-base).items()}))
    values = np.asarray([metrics(curve) for curve in curves.reshape(-1,100)]).reshape(*shape,len(KEYS))
    by_trajectory = values.mean(axis=2)
    aggregates = by_trajectory.mean(axis=1)
    for i, entry in enumerate(entries):
        name = entry["id"]
        for j, ordinal in enumerate(ids):
            val = by_trajectory[i,j]
            per_trajectory.append(dict(mask=name, stratum=entry["stratum"], trajectory=ordinal,
                                       **nullable(val), **{"delta_"+key: v for key,v in nullable(val-by_trajectory[0,j]).items()}))
        surgery = source / name / "surgery.json"
        verified = json.loads(surgery.read_text()) if surgery.exists() else {}
        parity = [json.loads(p.read_text())["masked_max_abs"] for p in
                  (source / name).glob("parity-*.json")]
        val = aggregates[i]
        status = [r["status"] for r in per_window if r["mask"] == name]
        summary.append(dict(mask=name, stratum=entry["stratum"], parameters=2332930 if name=="dense" else verified.get("parameters"),
                            compression=0 if name=="dense" else protocol["compression"] if verified else None,
                            removed=json.dumps(entry.get("removed", [])), masked_max_abs=max(parity, default=None),
                            ok=status.count("ok"), missing=status.count("missing"), failed=len(status)-status.count("ok")-status.count("missing"),
                            **nullable(val), **{"delta_"+key: v for key,v in nullable(val-aggregates[0]).items()}))
    out.mkdir(exist_ok=False)
    for name, rows in [("windows",per_window), ("trajectories",per_trajectory), ("masks",summary)]:
        csv_write(out/f"{name}.csv", rows)

    complete = np.isfinite(aggregates).all(1)
    rng = np.random.default_rng(101)
    sample = rng.integers(len(ids), size=(10000, len(ids)))
    boot = by_trajectory[:,sample,:].mean(2).transpose(1,0,2)
    results = {}
    for stratum in ("all","lower","middle","upper"):
        group = [i for i,e in enumerate(entries) if i>=2 and (stratum=="all" or e["stratum"]==stratum)]
        usable = [i for i in group if complete[i]]
        results[stratum] = record = dict(included=[entries[i]["id"] for i in usable],
                                         incomplete=[entries[i]["id"] for i in group if not complete[i]])
        if len(usable) < 2 or not complete[0]:
            continue
        point = comparisons(aggregates[usable], aggregates[0])
        dist = comparisons(boot[:,usable], boot[:,0])
        for key, item in point.items():
            record[key] = {k: float(v) if np.isfinite(v) else None for k,v in item.items()}
            for stat in ("spearman", "inversion_fraction", "inversions",
                         "short_better_long_worse", "short_worse_long_better"):
                finite = dist[key][stat][np.isfinite(dist[key][stat])]
                record[key][stat+"_ci95"] = np.quantile(finite,[.025,.975]).tolist() if len(finite) else None
                record[key][stat+"_valid_bootstraps"] = len(finite)
            record[key]["sign_flip_masks"] = [
                entries[i]["id"] for i in usable if
                direction(aggregates[i,0 if key=="E1" else 5], aggregates[0,0 if key=="E1" else 5])
                * direction(aggregates[i,4], aggregates[0,4]) < 0]
        difference = dist["mean5"]["spearman"] - dist["E1"]["spearman"]
        finite = difference[np.isfinite(difference)]
        record["delta_spearman_ci95"] = np.quantile(finite,[.025,.975]).tolist() if len(finite) else None
    (out/"rankings.json").write_text(json.dumps(results,indent=2,allow_nan=False)+"\n")

    def save(fig, name):
        for ext in ("png","pdf"):
            fig.savefig(out/f"{name}.{ext}",dpi=160,bbox_inches="tight")
        plt.close(fig)

    representatives = []
    if complete[0]:
        delta = aggregates-aggregates[0]
        for short, label in zip(SHORT,LABELS):
            fig, ax = plt.subplots()
            for stratum in ("lower","middle","upper"):
                group = [i for i,e in enumerate(entries) if complete[i] and e["stratum"]==stratum]
                ax.scatter(delta[group,short],delta[group,4],label=stratum)
            ax.axhline(0,color="0.7")
            ax.axvline(0,color="0.7")
            ax.set(xlabel=f"Δ {label}",ylabel="Δ E100")
            ax.legend(title="Страта")
            save(fig, f"scatter-{KEYS[short]}")
        fig, ax = plt.subplots()
        ax.plot(range(1,101),curves[0].mean(axis=(0,1)),label="dense")
        for stratum in ("lower","middle","upper"):
            group = [i for i,e in enumerate(entries) if complete[i] and e["stratum"]==stratum]
            if group:
                middle = np.median(aggregates[group,4])
                i = min(group,key=lambda i:(abs(aggregates[i,4]-middle),entries[i]["id"]))
                representatives.append(entries[i]["id"])
                ax.plot(range(1,101),curves[i].mean(axis=(0,1)),label=entries[i]["id"])
        ax.set(xlabel="Шаг прогноза",ylabel="MSE")
        ax.legend()
        save(fig,"horizons")
    full = results["all"]
    if full["incomplete"] or not complete[0]:
        conclusion = "Incomplete library or dense baseline. Review missing or failed windows before interpreting the ranking."
    elif full["E1"]["inversions"] or full["mean5"]["inversions"]:
        conclusion = ("The fixed masks include ranking inversions above the numerical tolerance. "
                      "Review their frequency, size and uncertainty before deciding on a strategy comparison.")
    else:
        conclusion = "No tolerance-aware inversion was observed in this library. The evidence for a strategy comparison at this compression is limited."
    lines = ["# CylinderFlow mask diagnostic", "", conclusion, "",
             "Ranks use exact-tie midranks. Inversions and sign flips use atol 1e-9 plus rtol 1e-4 times the larger absolute error.",
             "Intervals are percentile bootstrap intervals over 10,000 paired samples of whole trajectories, seed 101.",
             "Each sample keeps both windows and every mask. Findings are conditional on the fixed checkpoint and mask library.",
             "Representatives are nearest the stratum median E100, with mask ID breaking ties: "+", ".join(representatives)+".",
             "", "| Group | E1 / E100 Spearman | mean5 / E100 Spearman | Inversions / comparable pairs (E1, mean5) | Complete masks |",
             "|---|---:|---:|---:|---:|"]
    for key,g in results.items():
        if "E1" in g:
            a,b=g["E1"],g["mean5"]
            lines.append(f"| {key} | {a['spearman']} | {b['spearman']} | "
                         f"{a['inversions']}/{a['comparable_pairs']}, {b['inversions']}/{b['comparable_pairs']} "
                         f"| {len(g['included'])} |")
    (out/"REPORT.md").write_text("\n".join(lines)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    analyze(args.input,args.output)

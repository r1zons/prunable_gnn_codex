"""Summarize Flickr pruning study CSV."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


def _f(v: str) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def main() -> int:
    csv_path = Path("runs/flickr_pruning_study/flickr_results.csv")
    if not csv_path.exists():
        print(f"missing: {csv_path}")
        return 1

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    print("== Dense baseline by architecture ==")
    for row in rows:
        if row.get("phase") == "dense":
            print(row.get("num_layers"), row.get("hidden_channels"), row.get("test_accuracy"), row.get("parameter_count"))

    def show_phase(phase: str) -> None:
        print(f"\n== {phase} accuracy by method/sparsity ==")
        for row in rows:
            if row.get("phase") == phase:
                print(row.get("method"), row.get("sparsity"), row.get("test_accuracy"), row.get("test_macro_f1"))

    show_phase("post_prune")
    show_phase("post_finetune")

    print("\n== recovery gap (post_finetune - post_prune) ==")
    idx = {(r.get("run_dir"), r.get("method"), r.get("sparsity"), r.get("phase")): r for r in rows}
    keys = {(r.get("run_dir"), r.get("method"), r.get("sparsity")) for r in rows}
    for k in sorted(keys):
        pp = idx.get((k[0], k[1], k[2], "post_prune"))
        pf = idx.get((k[0], k[1], k[2], "post_finetune"))
        if pp and pf:
            print(k[1], k[2], _f(pf.get("test_accuracy", "nan")) - _f(pp.get("test_accuracy", "nan")))

    print("\n== compression ratio (pruned params / dense params) ==")
    dense_by_arch = {}
    for r in rows:
        if r.get("phase") == "dense":
            dense_by_arch[(r.get("num_layers"), r.get("hidden_channels"))] = _f(r.get("parameter_count", "nan"))
    for r in rows:
        if r.get("phase") in {"post_prune", "post_finetune"}:
            d = dense_by_arch.get((r.get("num_layers"), r.get("hidden_channels")))
            p = _f(r.get("parameter_count", "nan"))
            if d and d == d and p == p:
                print(r.get("phase"), r.get("method"), r.get("sparsity"), p / d)

    print("\n== speedup ratio (dense/pruned inference mean ms) ==")
    dense_t_by_arch = {}
    for r in rows:
        if r.get("phase") == "dense":
            dense_t_by_arch[(r.get("num_layers"), r.get("hidden_channels"))] = _f(r.get("inference_time_mean_ms", "nan"))
    for r in rows:
        if r.get("phase") in {"post_prune", "post_finetune"}:
            d = dense_t_by_arch.get((r.get("num_layers"), r.get("hidden_channels")))
            p = _f(r.get("inference_time_mean_ms", "nan"))
            if d and d == d and p == p and p > 0:
                print(r.get("phase"), r.get("method"), r.get("sparsity"), d / p)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

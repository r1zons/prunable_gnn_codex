"""Compare pruning behavior across datasets from a CSV file."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="runs/presentation/presentation_results.csv")
    args = parser.parse_args()

    rows = []
    with Path(args.csv).expanduser().open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    grouped = defaultdict(list)
    for row in rows:
        key = (row["dataset"] if "dataset" in row else Path(row["run_dir"]).parts[-3], row["model"], row["method"], row["sparsity"], row["phase"])
        grouped[key].append(float(row["test_accuracy"]))

    print("dataset,model,method,sparsity,phase,mean_test_accuracy")
    for key in sorted(grouped):
        vals = grouped[key]
        mean_val = sum(vals) / len(vals)
        print(f"{key[0]},{key[1]},{key[2]},{key[3]},{key[4]},{mean_val:.6f}")

    print("\npost_prune vs post_finetune gaps")
    print("dataset,model,method,sparsity,gap")
    phase_index = {(r.get("dataset", Path(r["run_dir"]).parts[-3]), r["model"], r["method"], r["sparsity"], r["phase"]): float(r["test_accuracy"]) for r in rows}
    keys = {(k[0], k[1], k[2], k[3]) for k in phase_index}
    for dataset, model, method, sparsity in sorted(keys):
        p = phase_index.get((dataset, model, method, sparsity, "post_prune"))
        f = phase_index.get((dataset, model, method, sparsity, "post_finetune"))
        if p is None or f is None:
            continue
        print(f"{dataset},{model},{method},{sparsity},{(f-p):.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

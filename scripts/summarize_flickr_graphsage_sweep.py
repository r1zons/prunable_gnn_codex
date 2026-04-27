"""Summarize Flickr GraphSAGE sweep results."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize Flickr GraphSAGE sweep CSV.")
    parser.add_argument("--csv", default="runs/flickr_graphsage_sweep/sweep_results.csv")
    args = parser.parse_args()

    csv_path = Path(args.csv).expanduser()
    if not csv_path.exists():
        print(f"[summarize_flickr_graphsage_sweep] missing CSV: {csv_path}")
        return 1

    rows = _read_rows(csv_path)
    _print_grouped(rows)
    _print_best(rows)
    _print_tradeoff(rows)
    return 0


def _read_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _print_grouped(rows: List[Dict[str, Any]]) -> None:
    grouped: Dict[Tuple[str, str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("num_layers", "")),
            str(row.get("hidden_channels", "")),
            str(row.get("method", "")),
            str(row.get("sparsity", "")),
            str(row.get("phase", "")),
        )
        grouped[key].append(row)

    print("=== Flickr GraphSAGE Sweep (grouped) ===")
    for key, members in sorted(grouped.items()):
        acc = _safe_mean([_to_float(member.get("test_accuracy")) for member in members])
        f1 = _safe_mean([_to_float(member.get("test_macro_f1")) for member in members])
        print(
            f"layers={key[0]} hidden={key[1]} method={key[2]} sparsity={key[3]} phase={key[4]} "
            f"acc={acc:.4f} f1={f1:.4f}"
        )


def _print_best(rows: List[Dict[str, Any]]) -> None:
    post = [row for row in rows if row.get("phase") == "post_finetune"]
    if not post:
        print("No post_finetune rows found.")
        return
    best = max(post, key=lambda row: _to_float(row.get("test_accuracy")))
    print("\n=== Best post_finetune accuracy ===")
    print(best)


def _print_tradeoff(rows: List[Dict[str, Any]]) -> None:
    candidates = [row for row in rows if row.get("phase") == "post_finetune" and row.get("method") != "dense"]
    if not candidates:
        print("No pruning candidates for tradeoff.")
        return
    best = max(
        candidates,
        key=lambda row: (
            _to_float(row.get("test_accuracy")),
            -_to_float(row.get("checkpoint_size_bytes")),
            -_to_float(row.get("inference_time_mean_sec")),
        ),
    )
    print("\n=== Best compression/speed tradeoff (heuristic) ===")
    print(best)


def _safe_mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _to_float(value: Any) -> float:
    if value in ("", None):
        return 0.0
    return float(value)


if __name__ == "__main__":
    raise SystemExit(main())

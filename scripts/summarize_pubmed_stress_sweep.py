"""Summarize PubMed stress sweep results from suite_runs CSV."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize PubMed stress sweep metrics.")
    parser.add_argument("--csv", default="runs/pubmed_stress_sweep/suite_runs.csv")
    args = parser.parse_args()

    csv_path = Path(args.csv).expanduser()
    if not csv_path.exists():
        print(f"[summarize_pubmed_stress_sweep] missing CSV: {csv_path}")
        return 1

    rows = _read_rows(csv_path)
    if not rows:
        print(f"[summarize_pubmed_stress_sweep] empty CSV: {csv_path}")
        return 1

    dense_map = _dense_baseline(rows)
    dense_val_map = _dense_baseline(rows, metric="val_accuracy")
    _print_dense_baseline(dense_map)
    _print_post_prune_table(rows)
    _print_accuracy_drop(rows, dense_map)
    _print_achieved_sparsity(rows)
    _print_param_reduction(rows, dense_map)
    _print_inference_time(rows)
    _print_highlights(rows, dense_val_map)
    return 0


def _read_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _arch_key(row: Dict[str, Any]) -> Tuple[str, str, str]:
    return (
        str(row.get("model", "")),
        str(row.get("num_layers", "")),
        str(row.get("hidden_channels", "")),
    )


def _dense_baseline(
    rows: List[Dict[str, Any]],
    metric: str = "test_accuracy",
) -> Dict[Tuple[str, str, str], float]:
    dense = {}
    for row in rows:
        if str(row.get("phase", "")) != "dense":
            continue
        dense[_arch_key(row)] = _to_float(row.get(metric))
    return dense


def _print_dense_baseline(dense_map: Dict[Tuple[str, str, str], float]) -> None:
    print("=== Dense baseline by architecture ===")
    for key in sorted(dense_map):
        model, layers, hidden = key
        print(f"{model:10s} l{layers}_h{hidden:>3s} | dense_acc={dense_map[key]:.4f}")


def _post_prune_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [row for row in rows if str(row.get("phase", "")) == "post_prune"]


def _print_post_prune_table(rows: List[Dict[str, Any]]) -> None:
    print("\n=== Post-prune accuracy (arch/method/sparsity) ===")
    for row in sorted(
        _post_prune_rows(rows),
        key=lambda r: (_arch_key(r), str(r.get("pruning_method", "")), _to_float(r.get("requested_sparsity"))),
    ):
        model, layers, hidden = _arch_key(row)
        method = str(row.get("pruning_method", ""))
        sparsity = _to_float(row.get("requested_sparsity"))
        acc = _to_float(row.get("test_accuracy"))
        f1 = _to_float(row.get("test_macro_f1"))
        print(f"{model:10s} l{layers}_h{hidden:>3s} | {method:18s} s={sparsity:.2f} | acc={acc:.4f} f1={f1:.4f}")


def _print_accuracy_drop(rows: List[Dict[str, Any]], dense_map: Dict[Tuple[str, str, str], float]) -> None:
    print("\n=== Accuracy drop (dense - post_prune) ===")
    for row in sorted(
        _post_prune_rows(rows),
        key=lambda r: (_arch_key(r), str(r.get("pruning_method", "")), _to_float(r.get("requested_sparsity"))),
    ):
        key = _arch_key(row)
        dense_acc = dense_map.get(key, 0.0)
        post_acc = _to_float(row.get("test_accuracy"))
        drop = dense_acc - post_acc
        print(
            f"{key[0]:10s} l{key[1]}_h{key[2]:>3s} | {str(row.get('pruning_method', '')):18s} "
            f"s={_to_float(row.get('requested_sparsity')):.2f} | drop={drop:.4f}"
        )


def _print_achieved_sparsity(rows: List[Dict[str, Any]]) -> None:
    print("\n=== Achieved sparsity ===")
    for row in sorted(
        _post_prune_rows(rows),
        key=lambda r: (_arch_key(r), str(r.get("pruning_method", "")), _to_float(r.get("requested_sparsity"))),
    ):
        print(
            f"{_arch_key(row)[0]:10s} l{_arch_key(row)[1]}_h{_arch_key(row)[2]:>3s} | "
            f"{str(row.get('pruning_method', '')):18s} req={_to_float(row.get('requested_sparsity')):.2f} "
            f"ach={_to_float(row.get('achieved_sparsity')):.4f}"
        )


def _print_param_reduction(rows: List[Dict[str, Any]], dense_map: Dict[Tuple[str, str, str], float]) -> None:
    print("\n=== Parameter count reduction ===")
    dense_params: Dict[Tuple[str, str, str], float] = {}
    for row in rows:
        if str(row.get("phase", "")) != "dense":
            continue
        dense_params[_arch_key(row)] = _to_float(row.get("parameter_count"))

    for row in sorted(
        _post_prune_rows(rows),
        key=lambda r: (_arch_key(r), str(r.get("pruning_method", "")), _to_float(r.get("requested_sparsity"))),
    ):
        key = _arch_key(row)
        before = dense_params.get(key, 0.0)
        after = _to_float(row.get("parameter_count"))
        reduction = before - after
        ratio = (reduction / before) if before > 0 else 0.0
        _ = dense_map  # keep signature symmetry; dense_map used by callers
        print(f"{key[0]:10s} l{key[1]}_h{key[2]:>3s} | {str(row.get('pruning_method', '')):18s} | -{reduction:.0f} ({ratio:.2%})")


def _print_inference_time(rows: List[Dict[str, Any]]) -> None:
    print("\n=== Inference time (post_prune) ===")
    for row in sorted(
        _post_prune_rows(rows),
        key=lambda r: (_arch_key(r), str(r.get("pruning_method", "")), _to_float(r.get("requested_sparsity"))),
    ):
        ms = _to_float(row.get("inference_time_mean_ms"))
        sec = ms / 1000.0
        print(
            f"{_arch_key(row)[0]:10s} l{_arch_key(row)[1]}_h{_arch_key(row)[2]:>3s} | "
            f"{str(row.get('pruning_method', '')):18s} s={_to_float(row.get('requested_sparsity')):.2f} "
            f"| {ms:.3f} ms ({sec:.6f} sec)"
        )


def _print_highlights(rows: List[Dict[str, Any]], dense_map: Dict[Tuple[str, str, str], float]) -> None:
    print("\n=== Validation-based candidate highlights ===")
    grouped: Dict[Tuple[str, str, str], Dict[float, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in _post_prune_rows(rows):
        key = _arch_key(row)
        sparsity = round(_to_float(row.get("requested_sparsity")), 2)
        dense_acc = dense_map.get(key, 0.0)
        drop = dense_acc - _to_float(row.get("val_accuracy"))
        grouped[key][sparsity].append(drop)

    printed = 0
    for key in sorted(grouped):
        dense_acc = dense_map.get(key, 0.0)
        drop_07 = _mean(grouped[key].get(0.7, []))
        drop_09 = _mean(grouped[key].get(0.9, []))
        if dense_acc >= 0.65 and 0.01 <= drop_07 <= 0.15 and drop_09 >= 0.05:
            print(
                f"{key[0]:10s} l{key[1]}_h{key[2]:>3s} | dense={dense_acc:.4f} "
                f"drop@0.7={drop_07:.4f} drop@0.9={drop_09:.4f}"
            )
            printed += 1
    if printed == 0:
        print("No rows matched the highlight heuristic yet.")


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _to_float(value: Any) -> float:
    if value in ("", None):
        return 0.0
    return float(value)


if __name__ == "__main__":
    raise SystemExit(main())

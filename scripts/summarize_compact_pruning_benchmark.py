"""Summarize compact pruning benchmark suite outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Mapping, Tuple


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize compact pruning benchmark results.")
    parser.add_argument("--csv", default="runs/compact_pruning_benchmark/suite_runs.csv")
    args = parser.parse_args()

    csv_path = Path(args.csv).expanduser()
    if not csv_path.exists():
        print(f"[summarize_compact_pruning_benchmark] missing CSV: {csv_path}")
        return 1

    rows = _read_rows(csv_path)
    if not rows:
        print(f"[summarize_compact_pruning_benchmark] empty CSV: {csv_path}")
        return 1

    enriched = _enrich_qlearning_diagnostics(rows)
    dense_baselines = _dense_accuracy_by_run(enriched)
    dense_val_baselines = _dense_metric_by_run(enriched, "val_accuracy")
    _print_dense_baseline(enriched)
    _print_post_prune_table(enriched, dense_baselines)
    _print_tradeoff_highlights(enriched, dense_val_baselines)
    _print_qlearning_vs_static_near_target(enriched, dense_baselines)
    _print_seed_stats(enriched, dense_baselines)
    _print_warnings(enriched)
    return 0


def _read_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _parse_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _run_key(row: Mapping[str, Any]) -> Tuple[str, str, str, str, str]:
    return (
        str(row.get("experiment_name", "")),
        str(row.get("dataset", "")),
        str(row.get("model", "")),
        str(row.get("num_layers", "")),
        str(row.get("hidden_channels", "")),
    )


def _run_instance_key(row: Mapping[str, Any]) -> Tuple[str, str, str, str, str, str]:
    return _run_key(row) + (str(row.get("run_seed", "")),)


def _enrich_qlearning_diagnostics(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cache: Dict[Path, Dict[str, Any]] = {}
    for row in rows:
        row["qlearning_final_hidden_widths"] = ""
        row["qlearning_target_gap"] = ""
        row["qlearning_stop_reason"] = ""
        method = str(row.get("pruning_method", ""))
        if method != "q_learning_tabular":
            continue
        run_dir = str(row.get("run_dir", "")).strip()
        if not run_dir:
            continue
        diag_path = Path(run_dir) / "action_space_diagnostics.json"
        if diag_path not in cache:
            if diag_path.exists():
                with diag_path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                cache[diag_path] = payload if isinstance(payload, dict) else {}
            else:
                cache[diag_path] = {}
        deployment = cache[diag_path].get("deployment", {}) if isinstance(cache[diag_path], dict) else {}
        if isinstance(deployment, dict):
            widths = deployment.get("final_hidden_widths", "")
            if isinstance(widths, list):
                row["qlearning_final_hidden_widths"] = str([int(v) for v in widths])
            row["qlearning_target_gap"] = deployment.get("final_target_gap", "")
            row["qlearning_stop_reason"] = deployment.get("final_stop_reason", row.get("stop_reason", ""))
    return rows


def _dense_accuracy_by_run(rows: Iterable[Mapping[str, Any]]) -> Dict[Tuple[str, str, str, str, str, str], float]:
    return _dense_metric_by_run(rows, "test_accuracy")


def _dense_metric_by_run(
    rows: Iterable[Mapping[str, Any]],
    metric: str,
) -> Dict[Tuple[str, str, str, str, str, str], float]:
    dense: Dict[Tuple[str, str, str, str, str, str], float] = {}
    for row in rows:
        if str(row.get("phase", "")) != "dense":
            continue
        value = _parse_float(row.get(metric))
        if value is not None:
            dense[_run_instance_key(row)] = value
    return dense


def _print_dense_baseline(rows: List[Mapping[str, Any]]) -> None:
    grouped: Dict[Tuple[str, str, str, str], List[float]] = defaultdict(list)
    for row in rows:
        if str(row.get("phase", "")) != "dense":
            continue
        acc = _parse_float(row.get("test_accuracy"))
        if acc is None:
            continue
        grouped[
            (
                str(row.get("dataset", "")),
                str(row.get("model", "")),
                str(row.get("num_layers", "")),
                str(row.get("hidden_channels", "")),
            )
        ].append(acc)
    print("=== Dense Baseline By Architecture ===")
    for key in sorted(grouped):
        values = grouped[key]
        std = stdev(values) if len(values) > 1 else 0.0
        print(
            f"{key[0]:8s} | {key[1]:10s} l{key[2]} h{key[3]:>3s} | "
            f"acc_mean={mean(values):.4f} acc_std={std:.4f} n={len(values)}"
        )


def _qlearning_accdrop(row: Mapping[str, Any]) -> float | None:
    name = str(row.get("experiment_name", ""))
    marker = "_accdrop"
    if marker not in name:
        return None
    suffix = name.split(marker, 1)[1]
    digits = "".join(ch for ch in suffix if ch.isdigit())
    if len(digits) != 3:
        return None
    return float(int(digits)) / 100.0


def _print_post_prune_table(
    rows: List[Mapping[str, Any]],
    dense_baselines: Mapping[Tuple[str, str, str, str, str, str], float],
) -> None:
    print("\n=== Post-Prune Summary (mean over seeds/runs) ===")
    grouped: Dict[Tuple[str, ...], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row.get("phase", "")) != "post_prune":
            continue
        grouped[
            (
                str(row.get("dataset", "")),
                str(row.get("model", "")),
                str(row.get("num_layers", "")),
                str(row.get("hidden_channels", "")),
                str(row.get("pruning_method", "")),
                str(row.get("requested_sparsity", "")),
                str(_qlearning_accdrop(row) if _qlearning_accdrop(row) is not None else ""),
            )
        ].append(row)

    for key in sorted(grouped):
        dataset, model, layers, hidden, method, req_s, accdrop = key
        members = grouped[key]
        achieved, accs, f1s, params, times, drops, gaps = [], [], [], [], [], [], []
        stop_reasons = Counter()
        widths = Counter()
        for row in members:
            run_dense = dense_baselines.get(_run_instance_key(row))
            acc = _parse_float(row.get("test_accuracy"))
            if acc is not None:
                accs.append(acc)
                if run_dense is not None:
                    drops.append(run_dense - acc)
            val = _parse_float(row.get("achieved_sparsity"))
            if val is not None:
                achieved.append(val)
            val = _parse_float(row.get("test_macro_f1"))
            if val is not None:
                f1s.append(val)
            val = _parse_float(row.get("parameter_count"))
            if val is not None:
                params.append(val)
            val = _parse_float(row.get("inference_time_mean_ms"))
            if val is not None:
                times.append(val)
            val = _parse_float(row.get("qlearning_target_gap"))
            if val is not None:
                gaps.append(val)
            reason = str(row.get("qlearning_stop_reason", row.get("stop_reason", ""))).strip()
            if reason:
                stop_reasons[reason] += 1
            width = str(row.get("qlearning_final_hidden_widths", "")).strip()
            if width:
                widths[width] += 1
        stop = stop_reasons.most_common(1)[0][0] if stop_reasons else ""
        width = widths.most_common(1)[0][0] if widths else ""
        print(
            f"{dataset:8s} {model:10s} l{layers} h{hidden:>3s} | {method:18s} "
            f"req={float(req_s):.2f} "
            f"accdrop={accdrop or '-':>5s} | "
            f"ach={_fmt_mean(achieved)} acc={_fmt_mean(accs)} drop={_fmt_mean(drops)} "
            f"f1={_fmt_mean(f1s)} params={_fmt_mean(params)} "
            f"t_ms={_fmt_mean(times)} q_gap={_fmt_mean(gaps)} "
            f"q_stop={stop or '-'} q_widths={width or '-'} n={len(members)}"
        )


def _print_tradeoff_highlights(
    rows: List[Mapping[str, Any]],
    dense_baselines: Mapping[Tuple[str, str, str, str, str, str], float],
) -> None:
    print("\n=== Validation-Selected Accuracy/Sparsity Trade-Off By Dataset/Model ===")
    grouped: Dict[Tuple[str, str, str, str], List[Tuple[float, float, Mapping[str, Any]]]] = defaultdict(list)
    for row in rows:
        if str(row.get("phase", "")) != "post_prune":
            continue
        achieved = _parse_float(row.get("achieved_sparsity"))
        acc = _parse_float(row.get("val_accuracy"))
        if achieved is None or acc is None:
            continue
        dense = dense_baselines.get(_run_instance_key(row))
        drop = (dense - acc) if dense is not None else 0.0
        grouped[
            (
                str(row.get("dataset", "")),
                str(row.get("model", "")),
                str(row.get("num_layers", "")),
                str(row.get("hidden_channels", "")),
            )
        ].append((drop, -achieved, row))

    for key in sorted(grouped):
        best = min(grouped[key], key=lambda item: (item[0], item[1]))
        row = best[2]
        print(
            f"{key[0]:8s} {key[1]:10s} l{key[2]} h{key[3]:>3s} | "
            f"{str(row.get('pruning_method', '')):18s} req={row.get('requested_sparsity', '')} "
            f"ach={row.get('achieved_sparsity', '')} val_acc={row.get('val_accuracy', '')} "
            f"val_drop={best[0]:.4f} test_acc={row.get('test_accuracy', '')}"
        )


def _print_qlearning_vs_static_near_target(
    rows: List[Mapping[str, Any]],
    dense_baselines: Mapping[Tuple[str, str, str, str, str, str], float],
) -> None:
    print("\n=== Post-Hoc Test Comparison Near Achieved Sparsity ~0.7 (Descriptive Only) ===")
    near_low, near_high = 0.65, 0.75
    grouped: Dict[Tuple[str, str, str, str], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if str(row.get("phase", "")) != "post_prune":
            continue
        achieved = _parse_float(row.get("achieved_sparsity"))
        acc = _parse_float(row.get("test_accuracy"))
        dense = dense_baselines.get(_run_instance_key(row))
        if achieved is None or acc is None or dense is None:
            continue
        if not (near_low <= achieved <= near_high):
            continue
        label = "q_learning_tabular" if str(row.get("pruning_method", "")) == "q_learning_tabular" else "static"
        grouped[
            (
                str(row.get("dataset", "")),
                str(row.get("model", "")),
                str(row.get("num_layers", "")),
                str(row.get("hidden_channels", "")),
            )
        ][label].append(dense - acc)

    for key in sorted(grouped):
        static_drop = grouped[key].get("static", [])
        q_drop = grouped[key].get("q_learning_tabular", [])
        if not static_drop and not q_drop:
            continue
        print(
            f"{key[0]:8s} {key[1]:10s} l{key[2]} h{key[3]:>3s} | "
            f"static_drop={_fmt_mean(static_drop)} q_drop={_fmt_mean(q_drop)}"
        )


def _print_seed_stats(
    rows: List[Mapping[str, Any]],
    dense_baselines: Mapping[Tuple[str, str, str, str, str, str], float],
) -> None:
    print("\n=== Seed Mean/Std (post_prune) ===")
    grouped: Dict[Tuple[str, ...], List[float]] = defaultdict(list)
    for row in rows:
        if str(row.get("phase", "")) != "post_prune":
            continue
        dense = dense_baselines.get(_run_instance_key(row))
        acc = _parse_float(row.get("test_accuracy"))
        if dense is None or acc is None:
            continue
        grouped[
            (
                str(row.get("dataset", "")),
                str(row.get("model", "")),
                str(row.get("num_layers", "")),
                str(row.get("hidden_channels", "")),
                str(row.get("pruning_method", "")),
                str(row.get("requested_sparsity", "")),
                str(_qlearning_accdrop(row) if _qlearning_accdrop(row) is not None else ""),
            )
        ].append(dense - acc)
    for key in sorted(grouped):
        values = grouped[key]
        std = stdev(values) if len(values) > 1 else 0.0
        print(
            f"{key[0]:8s} {key[1]:10s} l{key[2]} h{key[3]:>3s} | {key[4]:18s} "
            f"req={key[5]} accdrop={key[6] or '-'} | drop_mean={mean(values):.4f} drop_std={std:.4f} n={len(values)}"
        )


def _print_warnings(rows: List[Mapping[str, Any]]) -> None:
    print("\n=== Warnings ===")
    missing_timing = 0
    missing_diag = 0
    q_rows = 0
    for row in rows:
        if str(row.get("phase", "")) != "post_prune":
            continue
        if _parse_float(row.get("inference_time_mean_ms")) is None:
            missing_timing += 1
        if str(row.get("pruning_method", "")) == "q_learning_tabular":
            q_rows += 1
            if not str(row.get("qlearning_final_hidden_widths", "")).strip():
                missing_diag += 1
    if missing_timing:
        print(f"- Missing inference_time_mean_ms in {missing_timing} post_prune rows.")
    if missing_diag:
        print(f"- Missing q_learning diagnostics in {missing_diag}/{q_rows} q_learning post_prune rows.")
    if not missing_timing and not missing_diag:
        print("- No timing/diagnostics warnings.")


def _fmt_mean(values: Iterable[float]) -> str:
    seq = list(values)
    if not seq:
        return "-"
    return f"{mean(seq):.4f}"


if __name__ == "__main__":
    raise SystemExit(main())

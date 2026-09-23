"""Plot compact pruning benchmark results from combined pipeline CSV."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot compact pruning benchmark results.")
    parser.add_argument("--csv", default="exports/compact_pruning_benchmark_artifacts/combined_pipeline_results.csv")
    parser.add_argument("--out-dir", default="exports/compact_pruning_benchmark_plots")
    parser.add_argument("--format", default="png")
    parser.add_argument("--also-svg", action="store_true")
    parser.add_argument("--dataset", default="all", choices=["all", "cora", "citeseer", "pubmed"])
    parser.add_argument("--exclude-budget-exceeded", action="store_true")
    parser.add_argument("--dpi", type=int, default=200)
    return parser


def _to_numeric(df: pd.DataFrame, column: str) -> None:
    if column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")


def _ensure_column(df: pd.DataFrame, column: str, default: Any = "") -> None:
    if column not in df.columns:
        df[column] = default


def _normalize_method(value: Any) -> str:
    text = str(value).strip()
    if text == "q_learning_tabular":
        return "q_learning"
    return text


def _infer_max_accuracy_drop(experiment_name: str) -> float | None:
    match = re.search(r"accdrop(\d{3})", experiment_name)
    if not match:
        return None
    return float(int(match.group(1))) / 100.0


def _save(fig: plt.Figure, out_dir: Path, base_name: str, image_format: str, also_svg: bool, dpi: int, created: List[Path]) -> None:
    output = out_dir / f"{base_name}.{image_format}"
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    created.append(output)
    if also_svg:
        svg_path = out_dir / f"{base_name}.svg"
        fig.savefig(svg_path, bbox_inches="tight")
        created.append(svg_path)
    plt.close(fig)


def _seed_key(df: pd.DataFrame) -> pd.Series:
    key = pd.Series([""] * len(df), index=df.index, dtype=object)
    for column in ("seed", "run_seed", "benchmark_run_id"):
        if column in df.columns:
            values = df[column].astype(str).fillna("")
            key = key.where(key != "", values)
    return key


def _prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    for column, default in [
        ("phase", ""),
        ("dataset", ""),
        ("model", ""),
        ("num_layers", ""),
        ("hidden_channels", ""),
        ("method", ""),
        ("pruning_method", ""),
        ("requested_sparsity", ""),
        ("target_sparsity", ""),
        ("sparsity_level", ""),
        ("sparsity", ""),
        ("test_accuracy", ""),
        ("inference_time_mean_ms", ""),
        ("parameter_count", ""),
        ("stop_reason", ""),
        ("max_accuracy_drop", ""),
        ("target_gap", ""),
        ("final_hidden_widths", ""),
        ("qlearning_target_gap", ""),
        ("qlearning_final_hidden_widths", ""),
        ("qlearning_stop_reason", ""),
    ]:
        _ensure_column(df, column, default)

    for numeric in [
        "requested_sparsity",
        "target_sparsity",
        "sparsity_level",
        "sparsity",
        "achieved_sparsity",
        "test_accuracy",
        "accuracy_drop",
        "parameter_count",
        "parameter_reduction",
        "inference_time_mean_ms",
        "max_accuracy_drop",
        "target_gap",
        "qlearning_target_gap",
    ]:
        _to_numeric(df, numeric)

    method_source = df["pruning_method"].where(df["pruning_method"].astype(str).str.strip() != "", df["method"])
    df["method"] = method_source.map(_normalize_method)

    missing_budget = df["max_accuracy_drop"].isna()
    inferred = df["experiment_name"].astype(str).map(_infer_max_accuracy_drop) if "experiment_name" in df.columns else None
    if inferred is not None:
        df.loc[missing_budget, "max_accuracy_drop"] = pd.to_numeric(inferred[missing_budget], errors="coerce")

    static_mask = df["method"] != "q_learning"
    q_mask = df["method"] == "q_learning"
    df["requested_sparsity_unified"] = pd.to_numeric(df["requested_sparsity"], errors="coerce")
    if "sparsity_level" in df.columns:
        df.loc[static_mask & df["requested_sparsity_unified"].isna(), "requested_sparsity_unified"] = df.loc[
            static_mask & df["requested_sparsity_unified"].isna(), "sparsity_level"
        ]
    df.loc[static_mask & df["requested_sparsity_unified"].isna(), "requested_sparsity_unified"] = df.loc[
        static_mask & df["requested_sparsity_unified"].isna(), "sparsity"
    ]
    df.loc[q_mask & df["target_sparsity"].notna(), "requested_sparsity_unified"] = df.loc[q_mask & df["target_sparsity"].notna(), "target_sparsity"]
    df.loc[q_mask & df["requested_sparsity_unified"].isna(), "requested_sparsity_unified"] = df.loc[
        q_mask & df["requested_sparsity_unified"].isna(), "sparsity_level"
    ]
    df.loc[q_mask & df["requested_sparsity_unified"].isna(), "requested_sparsity_unified"] = df.loc[
        q_mask & df["requested_sparsity_unified"].isna(), "sparsity"
    ]

    df["seed_key"] = _seed_key(df).astype(str)
    return df


def _compute_dense_relative_metrics(df: pd.DataFrame) -> pd.DataFrame:
    dense = df[df["phase"] == "dense"].copy()
    keys = ["dataset", "model", "num_layers", "hidden_channels", "seed_key"]
    dense_seed = dense.groupby(keys, dropna=False).agg(
        dense_accuracy=("test_accuracy", "mean"),
        dense_parameter_count=("parameter_count", "mean"),
    )
    dense_arch = dense.groupby(["dataset", "model", "num_layers", "hidden_channels"], dropna=False).agg(
        dense_accuracy_arch=("test_accuracy", "mean"),
        dense_parameter_count_arch=("parameter_count", "mean"),
    )

    out = df.join(dense_seed, on=keys)
    out = out.join(dense_arch, on=["dataset", "model", "num_layers", "hidden_channels"])
    out["dense_accuracy_effective"] = out["dense_accuracy"].where(out["dense_accuracy"].notna(), out["dense_accuracy_arch"])
    out["dense_parameter_effective"] = out["dense_parameter_count"].where(
        out["dense_parameter_count"].notna(),
        out["dense_parameter_count_arch"],
    )

    if "accuracy_drop" not in out.columns or out["accuracy_drop"].isna().all():
        out["accuracy_drop"] = out["dense_accuracy_effective"] - out["test_accuracy"]
    else:
        missing = out["accuracy_drop"].isna()
        out.loc[missing, "accuracy_drop"] = out.loc[missing, "dense_accuracy_effective"] - out.loc[missing, "test_accuracy"]

    if "parameter_reduction" not in out.columns or out["parameter_reduction"].isna().all():
        out["parameter_reduction"] = out["dense_parameter_effective"] - out["parameter_count"]
    else:
        missing = out["parameter_reduction"].isna()
        out.loc[missing, "parameter_reduction"] = out.loc[missing, "dense_parameter_effective"] - out.loc[missing, "parameter_count"]
    return out


def _aggregate_for_plots(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "dataset",
        "model",
        "num_layers",
        "hidden_channels",
        "method",
        "requested_sparsity_unified",
        "max_accuracy_drop",
        "stop_reason",
    ]
    agg = (
        df.groupby(group_cols, dropna=False)
        .agg(
            achieved_sparsity_mean=("achieved_sparsity", "mean"),
            achieved_sparsity_std=("achieved_sparsity", "std"),
            test_accuracy_mean=("test_accuracy", "mean"),
            test_accuracy_std=("test_accuracy", "std"),
            accuracy_drop_mean=("accuracy_drop", "mean"),
            accuracy_drop_std=("accuracy_drop", "std"),
            parameter_count_mean=("parameter_count", "mean"),
            parameter_reduction_mean=("parameter_reduction", "mean"),
            inference_time_mean_ms_mean=("inference_time_mean_ms", "mean"),
            count=("test_accuracy", "count"),
        )
        .reset_index()
    )
    return agg


def _method_style(method: str) -> Dict[str, Any]:
    palette = {
        "random": ("tab:blue", "o"),
        "global_magnitude": ("tab:orange", "s"),
        "layerwise_magnitude": ("tab:green", "^"),
        "snip": ("tab:red", "D"),
        "q_learning": ("tab:purple", "X"),
    }
    color, marker = palette.get(method, ("gray", "o"))
    return {"color": color, "marker": marker}


def _plot_accuracy_vs_sparsity(agg: pd.DataFrame, dataset: str, out_dir: Path, image_format: str, also_svg: bool, dpi: int, created: List[Path]) -> None:
    subset = agg[agg["dataset"] == dataset]
    if subset.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in sorted(subset["method"].dropna().unique()):
        part = subset[subset["method"] == method].sort_values("achieved_sparsity_mean")
        if part.empty:
            continue
        style = _method_style(method)
        ax.plot(
            part["achieved_sparsity_mean"],
            part["test_accuracy_mean"],
            label=method,
            color=style["color"],
            marker=style["marker"],
            alpha=0.8,
            linewidth=1.0,
        )
    ax.set_title(f"Test Accuracy vs Achieved Sparsity — {dataset} (Post-hoc Descriptive)")
    ax.set_xlabel("Achieved sparsity")
    ax.set_ylabel("Test accuracy")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    _save(fig, out_dir, f"accuracy_vs_sparsity_{dataset}", image_format, also_svg, dpi, created)


def _plot_accuracy_drop_vs_sparsity(agg: pd.DataFrame, dataset: str, out_dir: Path, image_format: str, also_svg: bool, dpi: int, created: List[Path]) -> None:
    subset = agg[agg["dataset"] == dataset]
    if subset.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in sorted(subset["method"].dropna().unique()):
        part = subset[subset["method"] == method].sort_values("achieved_sparsity_mean")
        if part.empty:
            continue
        style = _method_style(method)
        ax.plot(
            part["achieved_sparsity_mean"],
            part["accuracy_drop_mean"],
            label=method,
            color=style["color"],
            marker=style["marker"],
            alpha=0.85,
            linewidth=1.0,
        )
    for budget in [0.03, 0.05, 0.07, 0.10]:
        ax.axhline(budget, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_title(f"Test Accuracy Drop vs Achieved Sparsity — {dataset} (Post-hoc Descriptive)")
    ax.set_xlabel("Achieved sparsity")
    ax.set_ylabel("Accuracy drop")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    _save(fig, out_dir, f"accuracy_drop_vs_sparsity_{dataset}", image_format, also_svg, dpi, created)


def _plot_accuracy_drop_by_model_family(
    agg: pd.DataFrame,
    dataset: str,
    model_family: str,
    out_dir: Path,
    image_format: str,
    also_svg: bool,
    dpi: int,
    created: List[Path],
) -> None:
    subset = agg[(agg["dataset"] == dataset) & (agg["model"] == model_family)].copy()
    if subset.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]
    arch_labels = sorted(subset.apply(lambda r: f"l{int(r['num_layers'])}_h{int(r['hidden_channels'])}", axis=1).unique())
    marker_by_arch = {arch: markers[i % len(markers)] for i, arch in enumerate(arch_labels)}
    for method in sorted(subset["method"].dropna().unique()):
        part = subset[subset["method"] == method].copy()
        part["arch"] = part.apply(lambda r: f"l{int(r['num_layers'])}_h{int(r['hidden_channels'])}", axis=1)
        for arch in sorted(part["arch"].unique()):
            shard = part[part["arch"] == arch].sort_values("achieved_sparsity_mean")
            if shard.empty:
                continue
            style = _method_style(method)
            ax.plot(
                shard["achieved_sparsity_mean"],
                shard["accuracy_drop_mean"],
                label=f"{method} {arch}",
                color=style["color"],
                marker=marker_by_arch[arch],
                linewidth=1.0,
                alpha=0.8,
            )
    ax.set_title(f"Test Accuracy Drop vs Achieved Sparsity — {dataset} {model_family} (Post-hoc)")
    ax.set_xlabel("Achieved sparsity")
    ax.set_ylabel("Accuracy drop")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    _save(fig, out_dir, f"accuracy_drop_vs_sparsity_{dataset}_{model_family}", image_format, also_svg, dpi, created)


def _best_sparsity_under_budgets(raw_prune: pd.DataFrame, out_dir: Path, image_format: str, also_svg: bool, dpi: int, created: List[Path]) -> None:
    budgets = [0.03, 0.05, 0.07, 0.10]
    unit_cols = ["dataset", "model", "num_layers", "hidden_channels", "seed_key"]
    rows: List[Dict[str, Any]] = []
    for budget in budgets:
        eligible = raw_prune[raw_prune["accuracy_drop"] <= budget]
        if eligible.empty:
            continue
        per_unit = (
            eligible.groupby(unit_cols + ["method"], dropna=False)["achieved_sparsity"]
            .max()
            .reset_index(name="best_achieved_sparsity")
        )
        per_unit["budget"] = budget
        rows.extend(per_unit.to_dict("records"))
    budget_df = pd.DataFrame(rows)
    if budget_df.empty:
        return
    summary = (
        budget_df.groupby(["budget", "method"], dropna=False)["best_achieved_sparsity"]
        .mean()
        .reset_index(name="mean_best_achieved_sparsity")
    )
    summary["analysis_scope"] = "post_hoc_test_descriptive_not_for_selection"
    summary.to_csv(out_dir / "best_sparsity_under_accuracy_budget.csv", index=False)

    pivot = summary.pivot(index="budget", columns="method", values="mean_best_achieved_sparsity").fillna(0.0)
    fig, ax = plt.subplots(figsize=(9, 5))
    methods = list(pivot.columns)
    width = 0.14 if methods else 0.2
    x = list(range(len(pivot.index)))
    for idx, method in enumerate(methods):
        offset = (idx - (len(methods) - 1) / 2) * width
        ax.bar([value + offset for value in x], pivot[method].tolist(), width=width, label=method, color=_method_style(method)["color"])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{float(v):.2f}" for v in pivot.index])
    ax.set_xlabel("Accuracy budget")
    ax.set_ylabel("Best achieved sparsity (mean)")
    ax.set_title("Achieved Sparsity Under Test-Accuracy Budgets (Post-hoc Descriptive)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, out_dir, "best_sparsity_under_accuracy_budget", image_format, also_svg, dpi, created)
    created.append(out_dir / "best_sparsity_under_accuracy_budget.csv")


def _plot_qlearning_target_vs_achieved(q_df: pd.DataFrame, out_dir: Path, image_format: str, also_svg: bool, dpi: int, created: List[Path]) -> None:
    if q_df.empty:
        return
    datasets = sorted(q_df["dataset"].dropna().unique())
    for dataset in datasets:
        part = q_df[q_df["dataset"] == dataset]
        if part.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 5))
        for budget in sorted(part["max_accuracy_drop"].dropna().unique()):
            shard = part[part["max_accuracy_drop"] == budget]
            ax.scatter(
                shard["requested_sparsity_unified"],
                shard["achieved_sparsity"],
                alpha=0.7,
                label=f"max_drop={budget:.2f}",
            )
        exceeded = part[part["stop_reason"] == "max_accuracy_drop_exceeded"]
        if not exceeded.empty:
            ax.scatter(
                exceeded["requested_sparsity_unified"],
                exceeded["achieved_sparsity"],
                marker="x",
                color="red",
                label="max_accuracy_drop_exceeded",
            )
        ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="gray", linewidth=1.0)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Requested/target sparsity")
        ax.set_ylabel("Achieved sparsity")
        ax.set_title(f"Q-learning Target vs Achieved Sparsity — {dataset}")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        _save(fig, out_dir, f"qlearning_target_vs_achieved_{dataset}", image_format, also_svg, dpi, created)

    fig, ax = plt.subplots(figsize=(7, 5))
    for budget in sorted(q_df["max_accuracy_drop"].dropna().unique()):
        shard = q_df[q_df["max_accuracy_drop"] == budget]
        ax.scatter(shard["requested_sparsity_unified"], shard["achieved_sparsity"], alpha=0.5, label=f"max_drop={budget:.2f}")
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="gray", linewidth=1.0)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Requested/target sparsity")
    ax.set_ylabel("Achieved sparsity")
    ax.set_title("Q-learning Target vs Achieved Sparsity")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    _save(fig, out_dir, "qlearning_target_vs_achieved", image_format, also_svg, dpi, created)


def _plot_qlearning_sensitivity(q_df: pd.DataFrame, out_dir: Path, image_format: str, also_svg: bool, dpi: int, created: List[Path]) -> None:
    if q_df.empty or q_df["max_accuracy_drop"].isna().all():
        return
    datasets = sorted(q_df["dataset"].dropna().unique())
    for dataset in datasets:
        part = q_df[q_df["dataset"] == dataset]
        if part.empty:
            continue
        grouped = (
            part.groupby(["requested_sparsity_unified", "max_accuracy_drop"], dropna=False)["achieved_sparsity"]
            .mean()
            .reset_index()
        )
        fig, ax = plt.subplots(figsize=(7, 5))
        for requested in sorted(grouped["requested_sparsity_unified"].dropna().unique()):
            shard = grouped[grouped["requested_sparsity_unified"] == requested].sort_values("max_accuracy_drop")
            ax.plot(
                shard["max_accuracy_drop"],
                shard["achieved_sparsity"],
                marker="o",
                linewidth=1.1,
                label=f"target={requested:.2f}",
            )
        ax.set_xlabel("max_accuracy_drop")
        ax.set_ylabel("Achieved sparsity")
        ax.set_title(f"Q-learning Achieved Sparsity vs max_accuracy_drop — {dataset}")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        _save(fig, out_dir, f"qlearning_sensitivity_max_accuracy_drop_{dataset}", image_format, also_svg, dpi, created)


def _plot_qlearning_stop_reason(q_df: pd.DataFrame, out_dir: Path, image_format: str, also_svg: bool, dpi: int, created: List[Path]) -> None:
    if q_df.empty or q_df["stop_reason"].astype(str).str.strip().eq("").all():
        return
    stop_df = (
        q_df.groupby(["stop_reason", "requested_sparsity_unified"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    stop_df.to_csv(out_dir / "qlearning_stop_reason_distribution.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    pivot = stop_df.pivot(index="stop_reason", columns="requested_sparsity_unified", values="count").fillna(0)
    pivot.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
    ax.set_xlabel("Stop reason")
    ax.set_ylabel("Count")
    ax.set_title("Q-learning Stop Reason Distribution")
    ax.grid(axis="y", alpha=0.25)
    _save(fig, out_dir, "qlearning_stop_reason_distribution", image_format, also_svg, dpi, created)
    created.append(out_dir / "qlearning_stop_reason_distribution.csv")


def _plot_parameter_vs_sparsity(agg: pd.DataFrame, out_dir: Path, image_format: str, also_svg: bool, dpi: int, created: List[Path]) -> None:
    datasets = sorted(agg["dataset"].dropna().unique())
    for dataset in datasets:
        part = agg[agg["dataset"] == dataset]
        if part.empty:
            continue
        use_reduction = part["parameter_reduction_mean"].notna().any()
        y_col = "parameter_reduction_mean" if use_reduction else "parameter_count_mean"
        y_label = "Parameter reduction" if use_reduction else "Parameter count"
        fig, ax = plt.subplots(figsize=(8, 5))
        for method in sorted(part["method"].dropna().unique()):
            shard = part[part["method"] == method].sort_values("achieved_sparsity_mean")
            style = _method_style(method)
            ax.plot(
                shard["achieved_sparsity_mean"],
                shard[y_col],
                color=style["color"],
                marker=style["marker"],
                label=method,
                linewidth=1.0,
                alpha=0.8,
            )
        ax.set_xlabel("Achieved sparsity")
        ax.set_ylabel(y_label)
        ax.set_title(f"{y_label} vs Achieved Sparsity — {dataset}")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        _save(fig, out_dir, f"parameter_count_vs_sparsity_{dataset}", image_format, also_svg, dpi, created)


def _plot_latency_vs_sparsity(agg: pd.DataFrame, out_dir: Path, image_format: str, also_svg: bool, dpi: int, created: List[Path]) -> None:
    if "inference_time_mean_ms_mean" not in agg.columns or agg["inference_time_mean_ms_mean"].isna().all():
        return
    datasets = sorted(agg["dataset"].dropna().unique())
    for dataset in datasets:
        part = agg[agg["dataset"] == dataset]
        if part.empty or part["inference_time_mean_ms_mean"].isna().all():
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        for method in sorted(part["method"].dropna().unique()):
            shard = part[part["method"] == method].sort_values("achieved_sparsity_mean")
            style = _method_style(method)
            ax.plot(
                shard["achieved_sparsity_mean"],
                shard["inference_time_mean_ms_mean"],
                color=style["color"],
                marker=style["marker"],
                label=method,
                linewidth=1.0,
                alpha=0.8,
            )
        ax.set_xlabel("Achieved sparsity")
        ax.set_ylabel("Inference time mean (ms)")
        ax.set_title(f"Inference Time vs Achieved Sparsity — {dataset} (latency on small graphs can be noisy)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        _save(fig, out_dir, f"inference_time_vs_sparsity_{dataset}", image_format, also_svg, dpi, created)


def _write_summaries(raw_prune: pd.DataFrame, out_dir: Path, created: List[Path]) -> None:
    agg_fields = {
        "achieved_sparsity": ["mean", "std"],
        "test_accuracy": ["mean", "std"],
        "accuracy_drop": ["mean", "std"],
        "parameter_count": ["mean"],
        "inference_time_mean_ms": ["mean"],
    }
    overall = (
        raw_prune.groupby(["method", "requested_sparsity_unified", "max_accuracy_drop"], dropna=False)
        .agg(agg_fields)
        .reset_index()
    )
    overall.columns = [
        "method",
        "requested_sparsity",
        "max_accuracy_drop",
        "mean_achieved_sparsity",
        "std_achieved_sparsity",
        "mean_test_accuracy",
        "std_test_accuracy",
        "mean_accuracy_drop",
        "std_accuracy_drop",
        "mean_parameter_count",
        "mean_inference_time_mean_ms",
    ]
    overall["count_rows"] = raw_prune.groupby(["method", "requested_sparsity_unified", "max_accuracy_drop"], dropna=False).size().values
    overall["analysis_scope"] = "post_hoc_test_descriptive_not_for_selection"
    overall_path = out_dir / "plot_summary_overall.csv"
    overall.to_csv(overall_path, index=False)
    created.append(overall_path)

    by_group = (
        raw_prune.groupby(
            ["dataset", "model", "num_layers", "hidden_channels", "method", "requested_sparsity_unified", "max_accuracy_drop"],
            dropna=False,
        )
        .agg(agg_fields)
        .reset_index()
    )
    by_group.columns = [
        "dataset",
        "model",
        "num_layers",
        "hidden_channels",
        "method",
        "requested_sparsity",
        "max_accuracy_drop",
        "mean_achieved_sparsity",
        "std_achieved_sparsity",
        "mean_test_accuracy",
        "std_test_accuracy",
        "mean_accuracy_drop",
        "std_accuracy_drop",
        "mean_parameter_count",
        "mean_inference_time_mean_ms",
    ]
    by_group["count_rows"] = raw_prune.groupby(
        ["dataset", "model", "num_layers", "hidden_channels", "method", "requested_sparsity_unified", "max_accuracy_drop"],
        dropna=False,
    ).size().values
    by_group["analysis_scope"] = "post_hoc_test_descriptive_not_for_selection"
    by_path = out_dir / "plot_summary_by_dataset_model_method.csv"
    by_group.to_csv(by_path, index=False)
    created.append(by_path)

    q_df = raw_prune[raw_prune["method"] == "q_learning"].copy()
    if not q_df.empty:
        q_summary = (
            q_df.groupby(
                ["dataset", "model", "num_layers", "hidden_channels", "requested_sparsity_unified", "max_accuracy_drop", "stop_reason"],
                dropna=False,
            )
            .agg(agg_fields)
            .reset_index()
        )
        q_summary.columns = [
            "dataset",
            "model",
            "num_layers",
            "hidden_channels",
            "requested_sparsity",
            "max_accuracy_drop",
            "stop_reason",
            "mean_achieved_sparsity",
            "std_achieved_sparsity",
            "mean_test_accuracy",
            "std_test_accuracy",
            "mean_accuracy_drop",
            "std_accuracy_drop",
            "mean_parameter_count",
            "mean_inference_time_mean_ms",
        ]
        q_summary["count_rows"] = q_df.groupby(
            ["dataset", "model", "num_layers", "hidden_channels", "requested_sparsity_unified", "max_accuracy_drop", "stop_reason"],
            dropna=False,
        ).size().values
        q_summary["analysis_scope"] = "post_hoc_test_descriptive_not_for_selection"
        q_path = out_dir / "qlearning_summary.csv"
        q_summary.to_csv(q_path, index=False)
        created.append(q_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    csv_path = Path(args.csv).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    df = _prepare_data(df)
    df = _compute_dense_relative_metrics(df)

    if args.dataset != "all":
        df = df[df["dataset"] == args.dataset].copy()

    raw_prune = df[df["phase"] == "post_prune"].copy()
    raw_prune = raw_prune[raw_prune["achieved_sparsity"].notna() & raw_prune["test_accuracy"].notna()].copy()
    if args.exclude_budget_exceeded and "stop_reason" in raw_prune.columns:
        raw_prune = raw_prune[raw_prune["stop_reason"] != "max_accuracy_drop_exceeded"].copy()

    agg = _aggregate_for_plots(raw_prune)
    created: List[Path] = []

    datasets = sorted(raw_prune["dataset"].dropna().unique())
    for dataset in datasets:
        _plot_accuracy_vs_sparsity(agg, dataset, out_dir, args.format, args.also_svg, args.dpi, created)
        _plot_accuracy_drop_vs_sparsity(agg, dataset, out_dir, args.format, args.also_svg, args.dpi, created)
        for model_family in sorted(raw_prune[raw_prune["dataset"] == dataset]["model"].dropna().unique()):
            _plot_accuracy_drop_by_model_family(
                agg,
                dataset,
                model_family,
                out_dir,
                args.format,
                args.also_svg,
                args.dpi,
                created,
            )

    _best_sparsity_under_budgets(raw_prune, out_dir, args.format, args.also_svg, args.dpi, created)

    q_df = raw_prune[raw_prune["method"] == "q_learning"].copy()
    _plot_qlearning_target_vs_achieved(q_df, out_dir, args.format, args.also_svg, args.dpi, created)
    _plot_qlearning_sensitivity(q_df, out_dir, args.format, args.also_svg, args.dpi, created)
    _plot_qlearning_stop_reason(q_df, out_dir, args.format, args.also_svg, args.dpi, created)
    _plot_parameter_vs_sparsity(agg, out_dir, args.format, args.also_svg, args.dpi, created)
    _plot_latency_vs_sparsity(agg, out_dir, args.format, args.also_svg, args.dpi, created)
    _write_summaries(raw_prune, out_dir, created)

    print(f"[plot_compact_pruning_benchmark] wrote {len(created)} artifacts to {out_dir}")
    print("[note] Accuracy trade-off plots are post-hoc descriptions of final test results, not configuration-selection evidence.")
    for path in sorted(created):
        print(f"- {path}")
    if not q_df.empty and (q_df["stop_reason"] == "max_accuracy_drop_exceeded").any():
        count = int((q_df["stop_reason"] == "max_accuracy_drop_exceeded").sum())
        print(f"[warning] q_learning rows with stop_reason=max_accuracy_drop_exceeded: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

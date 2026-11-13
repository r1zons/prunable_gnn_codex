"""Render and independently check the exported CylinderFlow evaluation."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import animation, pyplot as plt, ticker, tri
import numpy as np


STEPS = [1, 20, 50, 100]
NAMES = ["Эталон COMSOL", "Прогноз MeshGraphNet", "Модуль векторной ошибки"]


def save(fig, stem):
    for suffix in ("png", "pdf"):
        fig.savefig(stem.with_suffix(f".{suffix}"), dpi=160)
    plt.close(fig)


def error_plot(model, persistence, stem, title):
    fig, ax = plt.subplots(figsize=(8, 4.1), layout="constrained")
    steps = np.arange(1, 101)
    for curves, label, color in [(model, "MeshGraphNet", "#116983"),
                                  (persistence, "Неизменная скорость", "#bd541e")]:
        curves = np.atleast_2d(curves)
        ax.plot(steps, curves.mean(axis=0), label=label, color=color)
        if len(curves) > 1:
            low, high = np.quantile(curves, [.25, .75], axis=0)
            ax.fill_between(steps, low, high, color=color, alpha=.17)
    ax.set(xlabel="Шаг прогноза", ylabel="MSE компонентов скорости", title=title)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    ax.grid(alpha=.2)
    ax.legend()
    save(fig, stem)


def label(step, frames, dt):
    return f"+{step} · кадр {frames[step]} · Δt = {step*dt:.2f} с"


def draw_fields(fig, axes, mesh, values, scales, cmap, names):
    artists = []
    for ax, field, limits, colors, name in zip(axes, values, scales, cmap, names):
        artist = ax.tripcolor(mesh, field, shading="gouraud", cmap=colors,
                              vmin=limits[0], vmax=limits[1], rasterized=True)
        artists.append(artist)
        ax.set_title(name, fontsize=10)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(3))
        bar = fig.colorbar(artist, ax=ax, orientation="horizontal", shrink=.92, pad=.16)
        bar.locator = ticker.MaxNLocator(4)
        bar.update_ticks()
    return artists


def render_example(key, row, fields, dt, root):
    mesh = tri.Triangulation(*fields["xy"].T, triangles=fields["cells"])
    frames = fields["frames"]
    ref, pred = fields["reference"], fields["prediction"]
    speed = [np.linalg.norm(field, axis=-1) for field in (ref, pred, pred-ref)]
    top = max(speed[0].max(), speed[1].max())
    err = speed[2].max()
    title = f"{key} · valid:{row['ordinal']} · начальный кадр {row['start']} (t₀ = {row['start']*dt:.2f} с)"
    scales = [(0, top), (0, top), (0, err)]
    fig, axes = plt.subplots(4, 3, figsize=(12, 10), layout="constrained")
    for step, group in zip(STEPS, axes):
        draw_fields(fig, group, mesh, [field[step] for field in speed], scales,
                    ["viridis", "viridis", "magma"], NAMES)
        group[0].set_ylabel(label(step, frames, dt), fontsize=9)
    fig.suptitle(title + "\nСкорость |u| и ошибка ‖uпрогноз − uэталон‖")
    save(fig, root / "speed")

    fig, axes = plt.subplots(2, 4, figsize=(14, 5.4), layout="constrained")
    component_scales = [(min(ref[..., i].min(), pred[..., i].min()),
                         max(ref[..., i].max(), pred[..., i].max())) for i in range(2)]
    for step, group in zip([20, 100], axes):
        draw_fields(fig, group, mesh, [ref[step,:,0], pred[step,:,0], ref[step,:,1], pred[step,:,1]],
                    [component_scales[0]]*2+[component_scales[1]]*2, ["coolwarm"]*4,
                    ["Эталон uₓ", "Прогноз uₓ", "Эталон uᵧ", "Прогноз uᵧ"])
        group[0].set_ylabel(label(step, frames, dt), fontsize=9)
    fig.suptitle(title + "\nКомпоненты вектора скорости")
    save(fig, root / "components")

    error_plot(row["model"], row["persistence"], root / "error", title)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.1), layout="constrained")
    artists = draw_fields(fig, axes, mesh, [field[0] for field in speed], scales,
                          ["viridis", "viridis", "magma"], NAMES)
    heading = fig.suptitle(title + "\n" + label(0, frames, dt))

    def frame(step):
        for artist, field in zip(artists, speed):
            artist.set_array(field[step])
        heading.set_text(title + "\n" + label(step, frames, dt))
        return (*artists, heading)

    movie = animation.FuncAnimation(fig, frame, frames=range(101), interval=100, repeat=False)
    movie.save(root / "forecast.mp4", dpi=120, writer=animation.FFMpegWriter(
        fps=10, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"]))
    frame(50)
    fig.savefig(root / "preview.png", dpi=100)
    plt.close(fig)


def diagnostics(fields):
    """Describe near-wall error and signed-vorticity agreement on the original cells."""
    xy, cells, active = (fields[key] for key in ("xy", "cells", "active"))
    ref, pred = (fields[key] for key in ("reference", "prediction"))
    edges = np.sort(np.concatenate([cells[:, [0, 1]], cells[:, [1, 2]], cells[:, [2, 0]]]), axis=1)
    edges, count = np.unique(edges, axis=0, return_counts=True)
    boundary = edges[count == 1]
    midpoint = xy[boundary].mean(axis=1)
    lower, upper = xy.min(axis=0), xy.max(axis=0)
    outer = np.any(np.isclose(midpoint, lower, atol=1e-6) |
                   np.isclose(midpoint, upper, atol=1e-6), axis=1)

    def distance(segments):
        if not len(segments):
            return np.full(len(xy), np.inf)
        a, b = xy[segments].transpose(1, 0, 2)
        vector = b - a
        rel = xy[:, None] - a
        frac = np.clip(np.sum(rel * vector, axis=-1) / np.sum(vector**2, axis=-1), 0, 1)
        return np.linalg.norm(rel - frac[..., None] * vector, axis=-1).min(axis=1)

    near_hole = distance(boundary[~outer]) <= .05
    near_wall = distance(boundary[outer]) <= .05
    bulk = ~(near_hole | near_wall)
    hole_edges = boundary[~outer]
    wake = active & (xy[:, 0] < upper[0] - .16) & ~near_wall
    if len(hole_edges):
        wake &= xy[:, 0] > xy[hole_edges, 0].max() + .16
    else:
        wake &= False
    wake_xy = xy[wake]
    offsets = np.linspace(-.15, .15, 31)
    triangulation = tri.Triangulation(*xy.T, triangles=cells)
    common = np.ones(len(wake_xy), dtype=bool)
    finder = triangulation.get_trifinder()
    for shift in offsets:
        common &= finder(wake_xy[:, 0] + shift, wake_xy[:, 1]) >= 0
    wake_xy = wake_xy[common]
    points = xy[cells].astype("float64")
    dx, dy = (points[:, 1:] - points[:, :1]).transpose(2, 0, 1)
    determinant = dx[:, 0]*dy[:, 1] - dx[:, 1]*dy[:, 0]
    usable = active[cells].all(axis=1) & (np.abs(determinant) > 1e-12)
    area = np.abs(determinant[usable]) / 2

    def curl(values):
        dv = values[cells].astype("float64")
        du, dv = (dv[:, 1:] - dv[:, :1]).transpose(2, 0, 1)
        return ((dv[:, 0]*dy[:, 1] - dv[:, 1]*dy[:, 0] +
                 dx[:, 1]*du[:, 0] - dx[:, 0]*du[:, 1])[usable] / determinant[usable])

    result = []
    for step in STEPS:
        err = np.square(pred[step] - ref[step]).mean(axis=1)
        true, estimated = curl(ref[step]), curl(pred[step])
        energy_ref = np.sum(area * true**2)
        energy_pred = np.sum(area * estimated**2)
        row = {"step": step, "curl_energy_ratio": float(energy_pred / energy_ref),
               "curl_cosine": float(np.sum(area*true*estimated) / np.sqrt(energy_ref*energy_pred))}
        if step in (50, 100) and len(wake_xy):
            # Diagnostic alignment on a common wake domain, not a corrected forecast.
            interp = [tri.LinearTriInterpolator(triangulation, component)
                      for component in pred[step].T]
            target = ref[step, wake][common]
            candidates = []
            for shift in offsets:
                shifted = np.column_stack([f(wake_xy[:, 0] + shift, wake_xy[:, 1])
                                           for f in interp])
                candidates.append((float(np.square(shifted-target).mean()), float(shift)))
            aligned, offset = min(candidates, key=lambda item: (item[0], abs(item[1]), item[1]))
            row.update(wake_nodes=len(wake_xy), wake_offset=offset,
                       wake_mse=candidates[len(offsets)//2][0], wake_aligned_mse=aligned)
        for key, mask in [("obstacle", near_hole), ("wall", near_wall), ("bulk", bulk)]:
            mask = mask & active
            row[key+"_node_fraction"] = float(mask.sum() / active.sum())
            row[key+"_error_fraction"] = float(err[mask].sum() / err[active].sum())
            row[key+"_mse"] = float(err[mask].mean()) if mask.any() else None
        result.append(row)
    return result


def render(inputs, output):
    evaluation = json.loads((inputs / "evaluation.json").read_text())
    output.mkdir(exist_ok=False)
    all_diagnostics = []
    for row in evaluation["windows"]:
        with np.load(inputs / row["file"], allow_pickle=False) as fields:
            ref, pred = fields["reference"], fields["prediction"]
            active = fields["active"]
            np.testing.assert_array_equal(fields["frames"], np.arange(row["start"], row["start"]+101))
            np.testing.assert_array_equal(ref[0], pred[0])
            np.testing.assert_array_equal(active, np.isin(fields["node_type"], [0, 5]))
            np.testing.assert_array_equal(pred[:,~active], np.broadcast_to(pred[0,~active], pred[:,~active].shape))
            for method, values in [("model", pred[1:]), ("persistence", ref[:1])]:
                mse = np.square(values[:,active] - ref[1:,active]).mean(axis=(1,2), dtype="float64")
                np.testing.assert_allclose(mse, row[method], rtol=1e-6, atol=1e-10)
            all_diagnostics.extend(dict(ordinal=row["ordinal"], start=row["start"], **d)
                                   for d in diagnostics(fields))
    with (output / "spatial_diagnostics.csv").open("w") as stream:
        columns = list(dict.fromkeys(key for row in all_diagnostics for key in row))
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(all_diagnostics)
    rows = evaluation["windows"]
    aggregate = {}
    for method in ("model", "persistence"):
        aggregate[method] = np.array([[row[method] for row in rows if row["ordinal"] == ordinal]
                                      for ordinal in evaluation["ids"]]).mean(axis=1)
        np.testing.assert_allclose(aggregate[method].mean(axis=0), evaluation["aggregate"][method])
    error_plot(aggregate["model"], aggregate["persistence"], output / "aggregate",
               "Ошибка прогноза · 16 траекторий\nСреднее и межквартильный диапазон по траекториям")

    key = lambda r: (r["ordinal"], r["start"])
    fixed = next(row for row in rows if list(key(row)) == evaluation["fixed"])
    median = np.median([row["model"][-1] for row in rows])
    median_rank = sorted(rows, key=lambda r: (abs(r["model"][-1]-median), *key(r)))
    middle = next(row for row in median_rank if row != fixed)
    worst_rank = sorted(rows, key=lambda r: (-r["model"][-1], *key(r)))
    worst = next(row for row in worst_rank if row not in (fixed, middle))
    examples = [("fixed", "Фиксированное окно", fixed), ("median", "Окно у медианы", middle),
                ("worst", "Наибольшая ошибка", worst)]
    (output / "examples.json").write_text(json.dumps({
        "median_e100": median, "median_first": key(median_rank[0]), "worst_first": key(worst_rank[0]),
        "examples": {name: {"ordinal": row["ordinal"], "start": row["start"],
                            "e100": row["model"][-1], "persistence_e100": row["persistence"][-1]}
                     for name, _, row in examples},
    }, indent=2) + "\n")
    for name, title, row in examples:
        root = output / name
        root.mkdir()
        with np.load(inputs / row["file"], allow_pickle=False) as fields:
            render_example(title, row, fields, evaluation["dt_seconds"], root)
        print(name, "rendered", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    render(args.inputs, args.output)

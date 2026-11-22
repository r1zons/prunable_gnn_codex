"""Compare the seven V5 policies at the published Flickr hidden-channel budgets."""

import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import time

import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import Data
import yaml

from gnn_pruning.evaluation.metrics import classification_metrics
from gnn_pruning.experiments.flickr_dense import sha, tensors_hash, write
from gnn_pruning.experiments.frozen_validation import FrozenAccuracy
from gnn_pruning.models import build_model
from gnn_pruning.pruning.budget_v5 import Budget, METHODS, run, static_plan
from gnn_pruning.surgery.zhou import prune_zhou
from run_flickr_zhou_dense import checkpoint, load_dataset, normalize, restore, sparse


CONFIG = Path("configs/experiments/flickr_zhou_pruning.yaml")
CP = Path("dense/best.pt")


def budget_for(model, divisor, cfg):
    """Match the author's total hidden widths, keeping the original features and classes."""
    w1, w2 = (width // divisor for width in model.hidden_channel_dims)
    count = (model.in_channels + 3) * w1 + (w1 + 3 + model.out_channels) * w2 + model.out_channels
    target = 1 - count / sum(p.numel() for p in model.parameters())
    return Budget(model, target, cfg["tolerance"], cfg["minimum"], goal_widths=(w1, w2))


def select(dense, graph, train, val, budget, method, cfg, identity):
    """Use only train and validation inputs to produce a physically pruned model."""
    @torch.no_grad()
    def validate(model):
        return float((model.eval()(graph)[val].argmax(-1) == graph.y[val]).double().mean())

    evaluate = FrozenAccuracy(validate, identity)
    torch.cuda.synchronize()
    started = time.perf_counter()
    if method.startswith("static_"):
        plan = static_plan(dense, method.removeprefix("static_"), budget, cfg["policy_seed"], graph, train)
        kept = plan.kept_indices
        final = prune_zhou(prune_zhou(dense, 0, kept[0]), 1, kept[1])
        stats = dict(complete=True, terminal_reason="target_band", scoring_gradient_forwards=int(method == "static_snip"),
                     validation_evaluations=0)
        trace, decisions, training, q_table = [], [], [], []
    else:
        result = run(dense, evaluate, method, budget.target, seed=cfg["policy_seed"], budget=budget,
                     **cfg["q_learning"])
        final, kept, stats = result.model, result.kept_indices, result.statistics
        trace, decisions, training = result.trace, result.decisions, result.training_trace
        q_table = list(result.q_table.items())
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    assert evaluate.requests == stats["validation_evaluations"]
    assert stats["complete"] and budget.goal(final.hidden_channel_dims)
    assert budget.count(final.hidden_channel_dims) == sum(p.numel() for p in final.parameters()) < budget.p0
    return final, kept, dict(stats=stats, deployment=trace, decisions=decisions, training=training,
                             q_table=q_table, search_seconds=elapsed, validation_forwards=evaluate.forward_calls,
                             validation_requests=evaluate.requests, cache_hits=evaluate.requests-evaluate.forward_calls)


def assert_projection(dense, final, kept):
    """Independently reconstruct expected tensors from original concatenated IDs."""
    source, actual = dense.state_dict(), final.state_dict()
    assert source.keys() == actual.keys()
    for name, expected in source.items():
        if name.startswith("layers."):
            _, index, branch, _ = name.split(".", 3)
            layer = int(index)
            boundary = dense.branch_widths[layer][0]
            rows = ([i for i in kept[layer] if i < boundary] if branch == "self_branch"
                    else [i - boundary for i in kept[layer] if i >= boundary])
            expected = expected[rows]
            if layer and name.endswith("weight"):
                expected = expected[:, kept[layer-1]]
        elif name == "classifier.weight":
            expected = expected[:, kept[-1]]
        torch.testing.assert_close(actual[name], expected, rtol=0, atol=0)


@torch.no_grad()
def metrics(model, features, adj, labels, ids):
    logits = model.eval()(features, adj)[ids]
    assert torch.isfinite(logits).all()
    truth, predictions = labels[ids].cpu().numpy(), logits.argmax(-1).cpu().numpy()
    report = classification_metrics(truth, predictions)
    report["micro_f1"] = report["accuracy"]
    return report


@torch.no_grad()
def timing(dense, final, features, adj, cfg):
    """Pair warm, synchronized resident-graph timings, alternating execution order."""
    samples = dict(dense=[], pruned=[])
    models = (("dense", dense.eval()), ("pruned", final.eval()))
    for i in range(cfg["warmup"] + cfg["repeats"]):
        for name, model in (models if i % 2 == 0 else models[::-1]):
            torch.cuda.synchronize()
            started = time.perf_counter()
            model(features, adj)
            torch.cuda.synchronize()
            if i >= cfg["warmup"]:
                samples[name].append(time.perf_counter() - started)
    return {name: dict(samples_seconds=times, median_ms=1000*statistics.median(times),
                       mean_ms=1000*statistics.mean(times), stdev_ms=1000*statistics.stdev(times),
                       vertices_per_second=len(features)/statistics.median(times))
            for name, times in samples.items()}


def smoke(out, cfg):
    """Gate all methods, physical surgery, empty branches and reload on CUDA."""
    dense = build_model("graphsage_zhou", in_channels=5, out_channels=3, hidden_channels=4).cuda().eval()
    identity = tensors_hash(dense.state_dict())
    x = torch.randn(15, 5, device="cuda")
    adj = sparse(normalize(sp.eye(15, format="csr") + sp.eye(15, k=1, format="csr")), "cuda")
    y = torch.arange(15, device="cuda") % 3
    y[10:] = -1
    data = Data(x=x, y=y, adj=adj)
    budget = budget_for(dense, 2, cfg)
    small = dict(cfg, q_learning=dict(cfg["q_learning"], episodes=2))
    observed = {}
    for method in METHODS:
        final, kept, audit = select(dense, data, torch.arange(5, device="cuda"),
                                    torch.arange(5, 10, device="cuda"), budget, method, small, identity)
        assert_projection(dense, final, kept)
        path = out / (method + ".pt")
        checkpoint(path, final)
        reloaded, _ = restore(path)
        with torch.no_grad():
            torch.testing.assert_close(final.eval()(data), reloaded(data), rtol=0, atol=0)
        observed[method] = dict(widths=list(final.hidden_channel_dims), branch_widths=final.branch_widths,
                                validation_forwards=audit["validation_forwards"])
    empty = prune_zhou(dense, 0, [4, 5, 6, 7])
    assert_projection(dense, empty, [[4, 5, 6, 7], list(range(8))])
    with torch.no_grad():
        assert torch.isfinite(empty(data)).all()
    assert tensors_hash(dense.state_dict()) == identity
    write(out / "gate.json", dict(status="PASS", methods=observed,
          source=Path("source-commit.txt").read_text().strip(), dense_sha256=sha(CP)))


def execute(divisor, out, cfg, protocol):
    """Prune on the validation view, then measure frozen predictions and latency."""
    gate = json.loads(Path("smoke/gate.json").read_text())
    origin = json.loads(Path("dense/result.json").read_text())
    assert gate["status"] == "PASS" and gate["source"] == protocol["source"]
    assert gate["dense_sha256"] == origin["checkpoint_sha256"] == sha(CP)
    dense, original = restore(CP)
    assert original["epoch"] == origin["epoch"]
    frozen = tensors_hash(dense.state_dict())
    x, y, _, validation, full, split = load_dataset(Path("data"), out)
    manifest = json.loads((out / "data.json").read_text())
    assert manifest == json.loads(Path("dense/data.json").read_text())
    features = torch.as_tensor(x, device="cuda")
    validation = sparse(normalize(validation), "cuda")
    safe = y.copy()
    safe[split["te"]] = -1
    graph = Data(x=features, y=torch.as_tensor(safe, device="cuda"), adj=validation)
    train, val, test = (torch.as_tensor(split[key], device="cuda") for key in ("tr", "va", "te"))
    dense_val = metrics(dense, features, validation, graph.y, val)
    assert dense_val["micro_f1"] == origin["validation_micro_f1"]
    budget = budget_for(dense, divisor, cfg)
    scores = {}
    for method in METHODS:
        final, retained, audit = select(dense, graph, train, val, budget, method, cfg, manifest["adj_val_sha256"])
        assert_projection(dense, final, retained)
        assert tensors_hash(dense.state_dict()) == frozen
        folder = out / method
        folder.mkdir()
        path = folder / "checkpoint.pt"
        checkpoint(path, final, original_kept_ids=retained, dense_sha256=sha(CP))
        restored, _ = restore(path)
        assert tensors_hash(restored.state_dict()) == tensors_hash(final.state_dict())
        score = metrics(restored, features, validation, graph.y, val)
        p1 = sum(p.numel() for p in restored.parameters())
        scores[method] = dict(method=method, width_divisor=divisor, validation=score, parameters=p1,
                              compression=1-p1/budget.p0, branch_widths=restored.branch_widths,
                              search_seconds=audit["search_seconds"], validation_forwards=audit["validation_forwards"],
                              validation_requests=audit["validation_requests"], cache_hits=audit["cache_hits"],
                              strict_reload=True, dense_projection_exact=True, checkpoint_sha256=sha(path),
                              checkpoint_bytes=path.stat().st_size, parameter_bytes=4*p1,
                              terminal_reason=audit["stats"]["terminal_reason"])
        write(folder / "search.json", audit)
        print(divisor, method, score, audit["search_seconds"], flush=True)

    # Construct the original test view only after all seven policies have committed their masks.
    labels = torch.as_tensor(y, device="cuda")
    full = sparse(normalize(full), "cuda")
    dense_test = metrics(dense, features, full, labels, test)
    assert dense_test["micro_f1"] == origin["test_micro_f1"]
    for method, row in scores.items():
        folder = out / method
        restored, _ = restore(folder / "checkpoint.pt")
        assert row["checkpoint_sha256"] == sha(folder / "checkpoint.pt")
        row["test"] = metrics(restored, features, full, labels, test)
        timings = timing(dense, restored, features, full, cfg)
        row["timing"] = timings
        row["speedup"] = timings["dense"]["median_ms"] / timings["pruned"]["median_ms"]
    write(out / "result.json", dict(status="PASS", protocol=protocol, dense_validation=dense_val, dense_test=dense_test,
          dense_parameters=budget.p0, dense_sha256=sha(CP), conditions=list(scores.values()),
          peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke", "2x", "4x", "8x"))
    args = parser.parse_args()
    cfg = yaml.safe_load(CONFIG.read_text())
    assert sorted(cfg["ratios"]) == [2, 4, 8]
    assert torch.cuda.device_count() == 1 and "P100" in torch.cuda.get_device_name(0)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(cfg["policy_seed"])
    out = Path(args.stage)
    out.mkdir()
    protocol = dict(config=cfg, source=Path("source-commit.txt").read_text().strip(),
                    job=os.environ["LSB_JOBID"], host=platform.node(), gpu=torch.cuda.get_device_name(0),
                    torch=str(torch.__version__), cuda=torch.version.cuda)
    write(out / "protocol.json", protocol)
    if args.stage == "smoke":
        smoke(out, cfg)
    else:
        execute(int(args.stage[0]), out, cfg, protocol)


if __name__ == "__main__":
    main()

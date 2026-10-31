"""Frozen-checkpoint V2 workflow, isolated from legacy split/training machinery."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import torch
import torch_geometric

from gnn_pruning.config import load_yaml, resolve_config
from gnn_pruning.data import load_dataset
from gnn_pruning.models import build_model
from gnn_pruning.evaluation.metrics import classification_metrics
from .base import PruningContext
from .registry import get_pruner
from .static_v2 import StaticPlan


V2_METHODS = {"random_local_v2", "magnitude_local_v2", "snip_pt_local_v2", "grasp_pt_local_v2",
              "adaptive_greedy_local_v2", "q_learning_local_v2", "lookahead_local_v4"}


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def config_hash(path) -> str:
    """Hash both resolved typed settings and full raw V2/pruning configuration."""
    return _json_hash({"resolved": resolve_config(path).to_dict(), "source": load_yaml(path)})


def tensor_hash(tensors) -> str:
    """Canonical named tensor identity, independent of torch.save byte formatting."""
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def _metrics(model, data, indices):
    model.eval()
    with torch.no_grad():
        prediction = model(data).argmax(dim=-1)[indices]
    return classification_metrics(data.y[indices].cpu().numpy(), prediction.cpu().numpy())


def _validate_split(split, num_nodes):
    if set(split) != {"train", "val", "test"} or any(not split[k] for k in split):
        raise ValueError("Frozen split must contain nonempty train/val/test lists.")
    indices = [i for key in ("train", "val", "test") for i in split[key]]
    if any(type(i) is not int for i in indices) or sorted(indices) != list(range(num_nodes)):
        raise ValueError("Frozen split must be disjoint and cover the graph exactly.")


def _hidden_widths(value, num_layers):
    return [value] * (num_layers - 1) if isinstance(value, int) else list(value)


def _validate_dense_architecture(checkpoint, declared):
    """Reject a different dense experiment, even when state_dict loads strictly."""
    architecture = checkpoint["model_config"]
    for key in ("num_layers", "dropout", "in_channels", "out_channels"):
        expected = getattr(declared, key)
        if expected is not None and architecture.get(key) != expected:
            raise ValueError(f"Frozen checkpoint architecture differs from config: {key}.")
    widths = _hidden_widths(architecture["hidden_channels"], architecture["num_layers"])
    if widths != _hidden_widths(declared.hidden_channels, declared.num_layers):
        raise ValueError("Frozen checkpoint architecture differs from config: hidden_channels.")
    # Older signatures may omit architecture fields; the required model_config
    # still supplies them. If present, signature fields must agree with it.
    signature = checkpoint.get("run_signature", {})
    for key in ("num_layers", "dropout", "hidden_channels"):
        if key not in signature:
            continue
        saved, actual = signature[key], architecture[key]
        if key == "hidden_channels":
            saved = _hidden_widths(saved, architecture["num_layers"])
            actual = widths
        if saved != actual:
            raise ValueError(f"Frozen checkpoint architecture signature differs from model_config: {key}.")


def prune_from_checkpoint_v2(checkpoint_path: str, config_path: str):
    """Prune a pinned checkpoint with frozen splits; never fit or generate a split."""
    from .workflow import PruningArtifacts
    raw, resolved = load_yaml(config_path), resolve_config(config_path)
    cfg, pruning = raw.get("v2", {}), raw.get("pruning", {})
    method = pruning.get("method")
    if method not in V2_METHODS or not pruning.get("structured", True):
        raise ValueError("V2 methods require structured pruning.")
    target = float(pruning.get("target_sparsity", 0.5))
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    split_path = Path(cfg.get("split_path", "")).expanduser()
    if not split_path.is_absolute():
        split_path = Path(config_path).resolve().parent/split_path
    for label, path in (("checkpoint", checkpoint_path), ("split", split_path)):
        if not path.is_file() or cfg.get(label+"_sha256") != _file_hash(path):
            raise ValueError(f"Frozen {label} hash missing or mismatched.")
    expected_graph_hash = cfg.get("graph_sha256")
    if (not isinstance(expected_graph_hash, str) or len(expected_graph_hash) != 64
            or any(c not in "0123456789abcdef" for c in expected_graph_hash.lower())):
        raise ValueError("Frozen graph hash v2.graph_sha256 is required (64 hexadecimal characters).")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    split = load_yaml(split_path)
    signature = checkpoint.get("run_signature", {})
    if signature.get("dataset") != resolved.data.name:
        raise ValueError("Frozen checkpoint dataset differs from config dataset.")
    if signature.get("split_hash") != _json_hash(split):
        raise ValueError("Frozen split indices hash differs from dense checkpoint.")
    if signature.get("seed") != resolved.run.seed or signature.get("model_name") != resolved.model.name:
        raise ValueError("Frozen checkpoint model/seed differs from config.")
    if checkpoint.get("model_name") != resolved.model.name:
        raise ValueError("Checkpoint model identity mismatch.")
    _validate_dense_architecture(checkpoint, resolved.model)
    device = torch.device(resolved.device.device)
    model = build_model(checkpoint["model_name"], **checkpoint["model_config"])
    # Preserve tensor dtype when reconstructing a frozen checkpoint.
    source_dtype = next(v.dtype for v in checkpoint["model_state_dict"].values() if v.is_floating_point())
    model = model.to(device=device, dtype=source_dtype).eval()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    data = load_dataset(resolved.data.name, root=resolved.data.root)[0].clone().to(device)
    _validate_split(split, int(data.num_nodes))
    if data.x.shape[1] != model.in_channels:
        raise ValueError("Graph feature dimension differs from checkpoint.")
    data.x = data.x.to(dtype=source_dtype)
    graph_hash = tensor_hash({"x": data.x, "edge_index": data.edge_index, "y": data.y})
    if expected_graph_hash.lower() != graph_hash:
        raise ValueError("Frozen graph tensor hash mismatch.")
    idx = {key: torch.tensor(value, dtype=torch.long, device=device) for key, value in split.items()}
    # Search receives no test labels. Full labels are used only after termination.
    search_data = data.clone()
    search_data.y[idx["test"]] = -1
    def evaluate_validation(candidate):
        candidate.eval()
        with torch.no_grad():
            predicted = candidate(search_data).argmax(dim=-1)[idx["val"]]
        return float((predicted == search_data.y[idx["val"]]).float().mean())
    context = PruningContext(seed=resolved.run.seed, device=str(device),
                             config={"v2": cfg, "lookahead": raw.get("lookahead", {})},
                             data={"data": search_data, "train_idx": idx["train"],
                                   "evaluate_validation": evaluate_validation})
    pruner = get_pruner(method)()
    start = time.perf_counter()
    scores = pruner.score(model, context)
    score_seconds = time.perf_counter()-start
    start = time.perf_counter()
    plan = pruner.plan(model, scores, context, target)
    search_seconds = time.perf_counter()-start
    start = time.perf_counter()
    final = pruner.apply(model, plan, context)
    apply_seconds = time.perf_counter()-start
    p0, after = sum(p.numel() for p in model.parameters()), sum(p.numel() for p in final.parameters())
    if after > p0:
        raise RuntimeError("V2 increased parameter count.")
    achieved = 1-after/p0
    keeps = plan.kept_indices
    if isinstance(plan, StaticPlan):
        details, stats, trace, training, qtable = asdict(plan), {}, [], [], []
    else:
        details, stats, trace, training = {}, plan.statistics, plan.trace, plan.training_trace
        qtable = [{"state": list(state), "actions": row} for state, row in plan.q_table.items()]
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        git_sha, dirty = "unavailable", None
    is_lookahead = method == "lookahead_local_v4"
    protocol = "LOCAL-STRUCTURED-LOOKAHEAD-GREEDY-THRESHOLD-V1" if is_lookahead else "LOCAL-STRUCTURED-V2-R2"
    provenance = {"protocol": protocol, "method": method, "seed": resolved.run.seed,
                  "dataset": resolved.data.name, "checkpoint_sha256": _file_hash(checkpoint_path),
                  "dense_tensor_sha256": tensor_hash(model.state_dict()), "split_sha256": _file_hash(split_path),
                  "dense_model_config": checkpoint["model_config"],
                  "split_indices": split, "config_sha256": config_hash(config_path),
                  "graph_tensor_sha256": graph_hash,
                  "python_version": platform.python_version(), "torch_version": str(torch.__version__),
                  "pyg_version": torch_geometric.__version__, "device": str(device), "platform": platform.platform(),
                  "git_sha": git_sha, "tracked_worktree_dirty": dirty}
    if is_lookahead:
        provenance["termination_policy"] = "greedy_threshold"
    # Unique directory prevents accidental replacement of V1 or earlier V2 evidence.
    group = "local_structured_lookahead_threshold" if is_lookahead else "local_structured_v2"
    output = Path(resolved.run.output_dir).expanduser()/group/method/f"target_{target:g}"/uuid.uuid4().hex[:12]
    output.mkdir(parents=True, exist_ok=False)
    architecture = final.export_architecture_config()
    architecture.pop("name")
    checkpoint_out = output/"pruned.pt"
    torch.save({"method": method, "model_name": checkpoint["model_name"], "model_config": architecture,
                "model_state_dict": final.state_dict(), "source_checkpoint": str(checkpoint_path),
                "original_kept_ids": keeps, "provenance": provenance}, checkpoint_out)
    report = {"method": method, "requested_sparsity": target, "achieved_sparsity": achieved,
              "target_gap": abs(achieved-target), "target_violation": abs(achieved-target) > cfg.get("target_sparsity_tolerance", 0.01),
              "parameters_before": p0, "parameters_after": after, "unpruned": after == p0,
              "parameter_bytes": sum(p.numel()*p.element_size() for p in final.parameters()),
              "checkpoint_bytes": checkpoint_out.stat().st_size, "original_kept_ids": keeps,
              "hidden_widths": final.hidden_channel_dims, "provenance": provenance, "static_plan": details,
              "statistics": stats, "scoring_seconds": score_seconds, "search_seconds": search_seconds,
              "materialization_seconds": apply_seconds,
              "physical_surgery_seconds": apply_seconds if isinstance(plan, StaticPlan) else stats["surgery_seconds"],
              "total_scoring_seconds": score_seconds+stats.get("scoring_seconds", 0.),
              "reporting_evaluations": 4,
              "inference_benchmark": None,
              "inference_benchmark_status": "not measured in prototype correctness tests",
              "dense_metrics": {key: _metrics(model, data, idx[key]) for key in ("val", "test")},
              "zero_score_parameters": sum(int((s == 0).sum()) for s in scores.values()),
              "final_metrics": {key: _metrics(final, data, idx[key]) for key in ("val", "test")}}
    if is_lookahead:
        report.update(termination_policy="greedy_threshold",
                      signed_target_deviation=achieved-target,
                      overshoot=max(0.,achieved-target),
                      overshoot_beyond_tolerance=achieved > target+cfg.get("target_sparsity_tolerance", .01),
                      threshold_reached=achieved >= target-cfg.get("target_sparsity_tolerance", .01))
    metrics_path = output/"metrics.json"
    _write_json(metrics_path, report)
    _write_json(output/"config_snapshot.json", {"source": raw, "resolved": resolved.to_dict()})
    _write_json(output/"trace.json", {"deployment": trace, "training": training})
    _write_json(output/"q_table.json", qtable)
    decisions_path = None
    if is_lookahead:
        decisions_path = output/"lookahead_decisions.json"
        _write_json(decisions_path, plan.decisions)
    return PruningArtifacts(checkpoint_out, metrics_path, post_prune_metrics_path=metrics_path,
                            adaptive_trace_path=decisions_path,
                            rl_trace_path=output/"trace.json" if trace else None,
                            q_table_path=output/"q_table.json" if qtable else None)

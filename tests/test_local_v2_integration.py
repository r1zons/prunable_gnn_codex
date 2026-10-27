"""Synthetic checkpoint-to-CLI integration; no training or dataset downloads."""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import pytest
import torch
import yaml

from gnn_pruning.pruning import get_pruner
from gnn_pruning.pruning import workflow_v2
from gnn_pruning.pruning.workflow import prune_from_checkpoint, evaluate_pruned_checkpoint
from gnn_pruning.models import build_model
from test_local_surgery_v2 import graph, assert_tensors


METHODS = ["random_local_v2", "magnitude_local_v2", "snip_pt_local_v2", "grasp_pt_local_v2",
           "adaptive_greedy_local_v2", "q_learning_local_v2"]


def fixture_files(tmp_path, method, kind="gcn"):
    model = build_model(kind, in_channels=5, hidden_channels=8, out_channels=3,
                        num_layers=3, dropout=0.6).eval()
    data = graph(torch.float32)
    data.y = torch.tensor([0, 1, 2, 0, 1, 2, 0])
    split = {"train": [0, 2, 4], "val": [1, 3], "test": [5, 6]}
    split_path = tmp_path/"splits.yaml"
    split_path.write_text(yaml.safe_dump(split))
    split_hash = hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest()
    checkpoint = tmp_path/"dense.pt"
    cfg = model.export_architecture_config()
    cfg.pop("name")
    torch.save({"model_name": kind, "model_config": cfg, "model_state_dict": model.state_dict(),
                "run_signature": {"dataset": "cora", "model_name": kind, "seed": 101, "split_hash": split_hash,
                                  "num_layers": 3, "hidden_channels": 8, "dropout": 0.6}}, checkpoint)
    config = {"dataset": "cora", "model": {"name": kind, "num_layers": 3, "hidden_channels": 8, "dropout": 0.6},
              "run": {"seed": 101, "output_dir": str(tmp_path/"out")},
              "pruning": {"method": method, "target_sparsity": 0.3, "structured": True},
              "v2": {"split_path": str(split_path), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                     "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
                     "graph_sha256": workflow_v2.tensor_hash({"x": data.x, "edge_index": data.edge_index, "y": data.y}),
                     "penalty_lambda": 2., "episodes": 2, "max_steps": 2, "epsilon": 0.}}
    path = tmp_path/"config.yaml"
    path.write_text(yaml.safe_dump(config))
    return model, data, checkpoint, path, config


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("kind", ["gcn", "graphsage"])
def test_registry_and_existing_prune_command_routes_to_v2_frozen_provenance(tmp_path, monkeypatch, method, kind):
    model, data, checkpoint, path, cfg = fixture_files(tmp_path, method, kind)
    assert get_pruner(method)().supports_structured
    monkeypatch.setattr(workflow_v2, "load_dataset", lambda *args, **kwargs: [data])
    artifacts = prune_from_checkpoint(str(checkpoint), str(path))
    payload = torch.load(artifacts.pruned_checkpoint_path, map_location="cpu", weights_only=True)
    assert payload["method"] == method
    rebuilt = build_model(payload["model_name"], **payload["model_config"]).eval()
    rebuilt.load_state_dict(payload["model_state_dict"], strict=True)
    assert sum(p.numel() for p in rebuilt.parameters()) < sum(p.numel() for p in model.parameters())
    report = json.loads(artifacts.pruning_metrics_path.read_text())
    assert report["provenance"]["checkpoint_sha256"] == cfg["v2"]["checkpoint_sha256"]
    assert report["provenance"]["dense_model_config"] == {key: value for key, value in model.export_architecture_config().items() if key != "name"}
    assert report["provenance"]["split_indices"] == {"train": [0, 2, 4], "val": [1, 3], "test": [5, 6]}
    assert report["provenance"]["python_version"] == platform.python_version()
    assert len(report["provenance"]["config_sha256"]) == 64
    assert report["original_kept_ids"] == payload["original_kept_ids"]
    assert "test" in report["final_metrics"] and "macro_f1" in report["final_metrics"]["test"]
    before_hash = report["provenance"]["config_sha256"]
    cfg["v2"]["penalty_lambda"] = 9.
    path.write_text(yaml.safe_dump(cfg))
    assert workflow_v2.config_hash(path) != before_hash


@pytest.mark.parametrize("field", ["checkpoint_sha256", "split_sha256"])
def test_provenance_mismatch_fails_before_dataset_loading(tmp_path, monkeypatch, field):
    _, _, checkpoint, path, cfg = fixture_files(tmp_path, "magnitude_local_v2")
    cfg["v2"][field] = "wrong"
    path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(workflow_v2, "load_dataset", lambda *a, **k: pytest.fail("Loaded dataset before hash rejection"))
    with pytest.raises(ValueError, match="hash"):
        prune_from_checkpoint(str(checkpoint), str(path))


def test_split_dataset_mismatch_not_silently_regenerated(tmp_path, monkeypatch):
    _, _, checkpoint, path, cfg = fixture_files(tmp_path, "magnitude_local_v2")
    cfg["dataset"] = "pubmed"
    path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(workflow_v2, "load_dataset", lambda *a, **k: pytest.fail("Loaded mismatched dataset"))
    with pytest.raises(ValueError, match="dataset"):
        prune_from_checkpoint(str(checkpoint), str(path))


def test_test_labels_do_not_influence_q_policy_or_final_model(tmp_path, monkeypatch):
    _, data, checkpoint, path, cfg = fixture_files(tmp_path, "q_learning_local_v2")
    monkeypatch.setattr(workflow_v2, "load_dataset", lambda *a, **k: [data])
    first = prune_from_checkpoint(str(checkpoint), str(path))
    a = torch.load(first.pruned_checkpoint_path, weights_only=True)
    data.y[[5, 6]] = torch.tensor([1, 2])
    # This intentional label permutation is a separately pinned synthetic graph.
    cfg["v2"]["graph_sha256"] = workflow_v2.tensor_hash({"x": data.x, "edge_index": data.edge_index, "y": data.y})
    path.write_text(yaml.safe_dump(cfg))
    second = prune_from_checkpoint(str(checkpoint), str(path))
    b = torch.load(second.pruned_checkpoint_path, weights_only=True)
    assert_tensors(a["model_state_dict"], b["model_state_dict"])
    assert a["original_kept_ids"] == b["original_kept_ids"]
    assert Path(first.q_table_path).read_text() == Path(second.q_table_path).read_text()
    assert first.pruned_checkpoint_path != second.pruned_checkpoint_path
    assert first.pruned_checkpoint_path.exists()  # No overwrite on repeated invocation.


@pytest.mark.parametrize("kind", ["gcn", "graphsage"])
@pytest.mark.parametrize("field,value", [("hidden_channels", 32), ("num_layers", 4),
                                        ("dropout", 0.5), ("in_channels", 99), ("out_channels", 99)])
def test_dense_architecture_mismatch_rejected_before_dataset(tmp_path, monkeypatch, kind, field, value):
    _, _, checkpoint, path, cfg = fixture_files(tmp_path, "magnitude_local_v2", kind)
    cfg["model"][field] = value
    path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(workflow_v2, "load_dataset", lambda *a, **k: pytest.fail("Loaded graph before architecture rejection"))
    with pytest.raises(ValueError, match="architecture"):
        prune_from_checkpoint(str(checkpoint), str(path))


@pytest.mark.parametrize("kind", ["gcn", "graphsage"])
def test_equivalent_uniform_width_list_in_dense_checkpoint_is_accepted(tmp_path, monkeypatch, kind):
    _, data, checkpoint, path, cfg = fixture_files(tmp_path, "magnitude_local_v2", kind)
    payload = torch.load(checkpoint, weights_only=True)
    payload["model_config"]["hidden_channels"] = [8, 8]
    torch.save(payload, checkpoint)
    cfg["v2"]["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(workflow_v2, "load_dataset", lambda *a, **k: [data])
    assert prune_from_checkpoint(str(checkpoint), str(path)).pruned_checkpoint_path.exists()


@pytest.mark.parametrize("field,value", [("hidden_channels", 32), ("num_layers", 4), ("dropout", 0.5)])
def test_dense_architecture_signature_must_agree_with_payload(tmp_path, monkeypatch, field, value):
    _, _, checkpoint, path, cfg = fixture_files(tmp_path, "magnitude_local_v2")
    payload = torch.load(checkpoint, weights_only=True)
    payload["run_signature"][field] = value
    torch.save(payload, checkpoint)
    cfg["v2"]["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(workflow_v2, "load_dataset", lambda *a, **k: pytest.fail("Loaded graph before signature rejection"))
    with pytest.raises(ValueError, match="architecture"):
        prune_from_checkpoint(str(checkpoint), str(path))


@pytest.mark.parametrize("expected", [None, "", "FILL_GRAPH_SHA256", "x" * 64])
def test_graph_hash_is_required_before_dataset_loading(tmp_path, monkeypatch, expected):
    _, _, checkpoint, path, cfg = fixture_files(tmp_path, "magnitude_local_v2")
    cfg["v2"]["graph_sha256"] = expected
    path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(workflow_v2, "load_dataset", lambda *a, **k: pytest.fail("Loaded unpinned graph"))
    with pytest.raises(ValueError, match="graph.*hash|graph_sha256"):
        prune_from_checkpoint(str(checkpoint), str(path))


@pytest.mark.parametrize("kind", ["gcn", "graphsage"])
@pytest.mark.parametrize("component", ["features", "edges", "train_labels", "test_labels"])
def test_changed_graph_rejected_before_any_pruning(tmp_path, monkeypatch, kind, component):
    _, data, checkpoint, path, _ = fixture_files(tmp_path, "magnitude_local_v2", kind)
    if component == "features":
        data.x[0, 0] += 10
    elif component == "edges":
        data.edge_index = data.edge_index[:, :2]
    else:
        index = 0 if component == "train_labels" else 5
        data.y[index] = (data.y[index] + 1) % 3
    monkeypatch.setattr(workflow_v2, "load_dataset", lambda *a, **k: [data])
    monkeypatch.setattr(workflow_v2, "get_pruner", lambda *a, **k: pytest.fail("Scored changed graph"))
    with pytest.raises(ValueError, match="graph.*hash"):
        prune_from_checkpoint(str(checkpoint), str(path))


@pytest.mark.parametrize("kind", ["gcn", "graphsage"])
@pytest.mark.parametrize("existing_split", [False, True])
def test_legacy_evaluation_rejects_v2_without_generating_or_replacing_split(tmp_path, monkeypatch, kind, existing_split):
    from gnn_pruning.pruning import workflow
    _, data, checkpoint, path, _ = fixture_files(tmp_path, "magnitude_local_v2", kind)
    monkeypatch.setattr(workflow_v2, "load_dataset", lambda *a, **k: [data])
    artifacts = prune_from_checkpoint(str(checkpoint), str(path))
    split_path = artifacts.pruned_checkpoint_path.parent / "splits.yaml"
    if existing_split:
        split_path.write_text("unrelated split must remain untouched\n")
    before = split_path.read_bytes() if split_path.exists() else None
    monkeypatch.setattr(workflow, "_load_or_create_split", lambda **k: pytest.fail("Reached legacy split path"))
    with pytest.raises(ValueError, match="V2.*evaluate-pruned"):
        evaluate_pruned_checkpoint(str(artifacts.pruned_checkpoint_path), str(path))
    assert (split_path.read_bytes() if split_path.exists() else None) == before


def test_legacy_evaluation_still_accepts_v1_checkpoint(tmp_path, monkeypatch):
    from gnn_pruning.pruning import workflow
    _, _, checkpoint, path, _ = fixture_files(tmp_path, "magnitude_local_v2")
    original_split = object()
    monkeypatch.setattr(workflow, "_load_or_create_split", lambda **k: original_split)
    def evaluate(**kwargs):
        assert kwargs["split_override"] is original_split
        return {"test": {"accuracy": 0.5}}
    monkeypatch.setattr(workflow, "evaluate_dense", evaluate)
    artifacts = evaluate_pruned_checkpoint(str(checkpoint), str(path))
    assert json.loads(artifacts.metrics_path.read_text()) == {"test": {"accuracy": 0.5}}


def test_older_signature_without_architecture_still_uses_model_config(tmp_path, monkeypatch):
    _, data, checkpoint, path, cfg = fixture_files(tmp_path, "magnitude_local_v2")
    payload = torch.load(checkpoint, weights_only=True)
    for field in ("hidden_channels", "num_layers", "dropout"):
        del payload["run_signature"][field]
    torch.save(payload, checkpoint)
    cfg["v2"]["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(workflow_v2, "load_dataset", lambda *a, **k: [data])
    assert prune_from_checkpoint(str(checkpoint), str(path)).pruned_checkpoint_path.exists()

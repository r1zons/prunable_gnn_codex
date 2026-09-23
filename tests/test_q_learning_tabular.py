"""Tests for tabular Q-learning structural pruning MVP."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import torch
from torch_geometric.data import Data

from gnn_pruning.pruning import get_pruner
from gnn_pruning.pruning.workflow import prune_from_checkpoint
from gnn_pruning.rl.q_learning import action_key, select_action, update_q


def _dummy_dataset(num_nodes: int = 40, in_channels: int = 8, num_classes: int = 3):
    x = torch.randn((num_nodes, in_channels), dtype=torch.float32)
    y = torch.randint(0, num_classes, (num_nodes,), dtype=torch.long)
    edge_index = torch.vstack(
        [
            torch.arange(0, num_nodes, dtype=torch.long),
            torch.roll(torch.arange(0, num_nodes, dtype=torch.long), shifts=-1),
        ]
    )
    data = Data(x=x, y=y, edge_index=edge_index)

    class DummyDataset:
        def __getitem__(self, idx: int):
            _ = idx
            return data

    return DummyDataset()


def _make_checkpoint(path: Path, num_layers: int = 2, hidden_channels: int = 16) -> Path:
    from gnn_pruning.models import GraphSAGENodeClassifier

    model = GraphSAGENodeClassifier(
        in_channels=8,
        hidden_channels=hidden_channels,
        out_channels=3,
        num_layers=num_layers,
        dropout=0.0,
    )
    torch.save(
        {
            "model_name": "graphsage",
            "model_config": {
                "in_channels": 8,
                "hidden_channels": hidden_channels,
                "out_channels": 3,
                "num_layers": num_layers,
                "dropout": 0.0,
            },
            "model_state_dict": model.state_dict(),
        },
        path,
    )
    return path


def _make_config(path: Path, output_dir: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "base: base/default",
                "dataset: pubmed",
                "model: graphsage",
                "run:",
                f"  output_dir: {output_dir.as_posix()}",
                "device:",
                "  device: cpu",
                "pruning:",
                "  method: q_learning_tabular",
                "  target_sparsity: 0.5",
                "  structured: true",
                "  finetune_enabled: false",
                "q_learning:",
                "  episodes: 3",
                "  max_steps: 3",
                "  step_prune_ratios: [0.05, 0.10]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _make_multilayer_config(
    path: Path,
    output_dir: Path,
    step_ratios: str = "[0.05, 0.10]",
    episodes: int = 2,
    max_steps: int = 6,
    min_channels_per_layer: int = 4,
    epsilon_start: float = 0.0,
    epsilon_end: float = 0.0,
    epsilon_decay: float = 1.0,
    max_accuracy_drop: float = 0.05,
    allow_nonmonotonic_layer_order: bool = False,
    structural_pruning_mode: str = "cascade",
) -> Path:
    path.write_text(
        "\n".join(
            [
                "base: base/default",
                "dataset: pubmed",
                "model: graphsage",
                "model:",
                "  name: graphsage",
                "  num_layers: 3",
                "  hidden_channels: 16",
                "run:",
                f"  output_dir: {output_dir.as_posix()}",
                "device:",
                "  device: cpu",
                "pruning:",
                "  method: q_learning_tabular",
                "  target_sparsity: 0.7",
                "  structured: true",
                "  finetune_enabled: false",
                "q_learning:",
                f"  episodes: {episodes}",
                f"  max_steps: {max_steps}",
                f"  step_prune_ratios: {step_ratios}",
                f"  min_channels_per_layer: {min_channels_per_layer}",
                f"  epsilon_start: {epsilon_start}",
                f"  epsilon_end: {epsilon_end}",
                f"  epsilon_decay: {epsilon_decay}",
                f"  max_accuracy_drop: {max_accuracy_drop}",
                f"  allow_nonmonotonic_layer_order: {'true' if allow_nonmonotonic_layer_order else 'false'}",
                f"  structural_pruning_mode: {structural_pruning_mode}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _param_count(path: Path) -> int:
    payload = torch.load(path, map_location="cpu")
    state = payload["model_state_dict"]
    return int(sum(value.numel() for value in state.values()))


def _load_pruning_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_q_learning_tabular_pruner_is_registered() -> None:
    pruner_cls = get_pruner("q_learning_tabular")
    assert pruner_cls.name == "q_learning_tabular"


def test_q_learning_tabular_outputs_artifacts_and_state_features(monkeypatch, tmp_path: Path) -> None:
    training_workflow = importlib.import_module("gnn_pruning.training.workflow")
    pruning_workflow = importlib.import_module("gnn_pruning.pruning.workflow")
    monkeypatch.setattr(training_workflow, "load_dataset", lambda name, root: _dummy_dataset())
    monkeypatch.setattr(pruning_workflow, "load_dataset", lambda name, root: _dummy_dataset())

    ckpt = _make_checkpoint(tmp_path / "dense.pt")
    cfg = _make_config(tmp_path / "cfg.yaml", tmp_path / "run")
    artifacts = prune_from_checkpoint(str(ckpt), str(cfg))

    assert artifacts.q_table_path is not None and artifacts.q_table_path.exists()
    assert artifacts.rl_trace_path is not None and artifacts.rl_trace_path.exists()
    assert artifacts.pruning_metrics_path.exists()

    trace = json.loads(artifacts.rl_trace_path.read_text(encoding="utf-8"))
    assert trace
    state = trace[0]["state"]
    for key in [
        "num_nodes_bucket",
        "num_edges_bucket",
        "avg_degree_bucket",
        "density_bucket",
        "num_features_bucket",
        "num_classes_bucket",
        "current_sparsity_bucket",
        "target_gap_bucket",
        "accuracy_drop_bucket",
        "remaining_channels_bucket",
    ]:
        assert key in state

    assert "speed_proxy" in trace[0]["info"]


def test_q_learning_tabular_is_structural(monkeypatch, tmp_path: Path) -> None:
    training_workflow = importlib.import_module("gnn_pruning.training.workflow")
    pruning_workflow = importlib.import_module("gnn_pruning.pruning.workflow")
    monkeypatch.setattr(training_workflow, "load_dataset", lambda name, root: _dummy_dataset())
    monkeypatch.setattr(pruning_workflow, "load_dataset", lambda name, root: _dummy_dataset())

    ckpt = _make_checkpoint(tmp_path / "dense.pt")
    cfg = _make_config(tmp_path / "cfg.yaml", tmp_path / "run")
    artifacts = prune_from_checkpoint(str(ckpt), str(cfg))

    assert _param_count(artifacts.pruned_checkpoint_path) < _param_count(ckpt)


def test_q_learning_multistep_graphsage_l3_has_no_oob_keep_indices(monkeypatch, tmp_path: Path) -> None:
    training_workflow = importlib.import_module("gnn_pruning.training.workflow")
    pruning_workflow = importlib.import_module("gnn_pruning.pruning.workflow")
    monkeypatch.setattr(training_workflow, "load_dataset", lambda name, root: _dummy_dataset())
    monkeypatch.setattr(pruning_workflow, "load_dataset", lambda name, root: _dummy_dataset())

    ckpt = _make_checkpoint(tmp_path / "dense_l3.pt", num_layers=3, hidden_channels=16)
    cfg = _make_multilayer_config(tmp_path / "cfg_l3.yaml", tmp_path / "run_l3")
    artifacts = prune_from_checkpoint(str(ckpt), str(cfg))

    assert artifacts.rl_trace_path is not None and artifacts.rl_trace_path.exists()
    trace = json.loads(artifacts.rl_trace_path.read_text(encoding="utf-8"))
    assert trace
    assert all("current_layer_width" in step["info"] for step in trace)
    assert all("num_keep" in step["info"] for step in trace)
    assert all("min_channels_per_layer" in step["info"] for step in trace)
    assert _param_count(artifacts.pruned_checkpoint_path) < _param_count(ckpt)


def test_q_learning_repeated_pruning_uses_current_layer_width(monkeypatch, tmp_path: Path) -> None:
    training_workflow = importlib.import_module("gnn_pruning.training.workflow")
    pruning_workflow = importlib.import_module("gnn_pruning.pruning.workflow")
    monkeypatch.setattr(training_workflow, "load_dataset", lambda name, root: _dummy_dataset())
    monkeypatch.setattr(pruning_workflow, "load_dataset", lambda name, root: _dummy_dataset())

    ckpt = _make_checkpoint(tmp_path / "dense_repeat.pt", num_layers=3, hidden_channels=16)
    cfg = _make_multilayer_config(tmp_path / "cfg_repeat.yaml", tmp_path / "run_repeat", step_ratios="[0.10]")
    artifacts = prune_from_checkpoint(str(ckpt), str(cfg))
    trace = json.loads(artifacts.rl_trace_path.read_text(encoding="utf-8"))

    width_progress = [
        int(step["info"].get("current_layer_width", 0))
        for step in trace
        if not step.get("stop_selected", False) and not step["info"].get("invalid_action_reason", "")
    ]
    assert width_progress
    assert min(width_progress) <= max(width_progress)
    assert all(width > 0 for width in width_progress)


def test_q_learning_invalid_action_is_skipped_or_stops_gracefully(monkeypatch, tmp_path: Path) -> None:
    training_workflow = importlib.import_module("gnn_pruning.training.workflow")
    pruning_workflow = importlib.import_module("gnn_pruning.pruning.workflow")
    monkeypatch.setattr(training_workflow, "load_dataset", lambda name, root: _dummy_dataset())
    monkeypatch.setattr(pruning_workflow, "load_dataset", lambda name, root: _dummy_dataset())

    ckpt = _make_checkpoint(tmp_path / "dense_invalid.pt", num_layers=3, hidden_channels=16)
    cfg = _make_multilayer_config(tmp_path / "cfg_invalid.yaml", tmp_path / "run_invalid", step_ratios="[0.0, 0.10]")
    artifacts = prune_from_checkpoint(str(ckpt), str(cfg))
    trace = json.loads(artifacts.rl_trace_path.read_text(encoding="utf-8"))
    assert trace
    invalid_entries = [step for step in trace if step["info"].get("invalid_action_reason", "")]
    assert all("invalid_action_reason" in step["info"] for step in invalid_entries)
    assert all(not str(step["info"].get("invalid_action_reason", "")).startswith("surgery_failed:IndexError") for step in trace)


def test_q_learning_final_model_metrics_use_deployment_rollout(monkeypatch, tmp_path: Path) -> None:
    training_workflow = importlib.import_module("gnn_pruning.training.workflow")
    pruning_workflow = importlib.import_module("gnn_pruning.pruning.workflow")
    methods_module = importlib.import_module("gnn_pruning.pruning.methods")
    monkeypatch.setattr(training_workflow, "load_dataset", lambda name, root: _dummy_dataset())
    monkeypatch.setattr(pruning_workflow, "load_dataset", lambda name, root: _dummy_dataset())

    def _forced_select_action(*, state_key_value, actions, q_table, epsilon, rng):  # type: ignore[no-untyped-def]
        _ = (state_key_value, q_table, rng)
        target_ratio = 0.05 if float(epsilon) > 0.0 else 0.10
        for action in actions:
            if "prune_ratio" in action and float(action["prune_ratio"]) == target_ratio:
                return dict(action)
        return dict(actions[0])

    def _forced_update_q(*, q_table, state_key_value, action_key_value, reward, next_state_key_value, alpha, gamma, terminal=False):  # type: ignore[no-untyped-def]
        _ = (action_key_value, reward, next_state_key_value, alpha, gamma, terminal)
        row = q_table.setdefault(state_key_value, {})
        row["layer:0|ratio:0.050000"] = -1.0
        row["layer:0|ratio:0.100000"] = 2.0

    monkeypatch.setattr(methods_module, "select_action", _forced_select_action)
    monkeypatch.setattr(methods_module, "update_q", _forced_update_q)

    ckpt = _make_checkpoint(tmp_path / "dense_deploy.pt", num_layers=3, hidden_channels=16)
    cfg = _make_multilayer_config(
        tmp_path / "cfg_deploy.yaml",
        tmp_path / "run_deploy",
        step_ratios="[0.05, 0.10]",
        episodes=1,
        max_steps=1,
        epsilon_start=1.0,
        epsilon_end=1.0,
        epsilon_decay=1.0,
    )
    artifacts = prune_from_checkpoint(str(ckpt), str(cfg))
    trace = json.loads(artifacts.rl_trace_path.read_text(encoding="utf-8"))
    metrics = _load_pruning_metrics(artifacts.pruning_metrics_path)

    training_valid = [step for step in trace if step.get("phase") == "training" and not step.get("stop_selected", False) and not step["info"].get("invalid_action_reason", "")]
    deployment_valid = [step for step in trace if step.get("phase") == "deployment" and not step.get("stop_selected", False) and not step["info"].get("invalid_action_reason", "")]
    assert training_valid and deployment_valid
    assert float(training_valid[0]["action"]["prune_ratio"]) == 0.05
    assert float(deployment_valid[0]["action"]["prune_ratio"]) == 0.10

    achieved = float(metrics["achieved_sparsity"])
    deployment_sparsity = float(deployment_valid[-1]["info"]["current_sparsity"])
    training_sparsity = float(training_valid[-1]["info"]["current_sparsity"])
    assert abs(achieved - deployment_sparsity) < 1e-9
    assert abs(achieved - training_sparsity) > 1e-6


def test_q_learning_num_adaptive_steps_counts_valid_deployment_steps(monkeypatch, tmp_path: Path) -> None:
    training_workflow = importlib.import_module("gnn_pruning.training.workflow")
    pruning_workflow = importlib.import_module("gnn_pruning.pruning.workflow")
    monkeypatch.setattr(training_workflow, "load_dataset", lambda name, root: _dummy_dataset())
    monkeypatch.setattr(pruning_workflow, "load_dataset", lambda name, root: _dummy_dataset())

    ckpt = _make_checkpoint(tmp_path / "dense_steps.pt", num_layers=3, hidden_channels=16)
    cfg = _make_multilayer_config(tmp_path / "cfg_steps.yaml", tmp_path / "run_steps", step_ratios="[0.0, 0.10]")
    artifacts = prune_from_checkpoint(str(ckpt), str(cfg))
    trace = json.loads(artifacts.rl_trace_path.read_text(encoding="utf-8"))
    metrics = _load_pruning_metrics(artifacts.pruning_metrics_path)
    details = metrics.get("details", {})

    deployment_entries = [step for step in trace if step.get("phase") == "deployment"]
    deployment_valid = [step for step in deployment_entries if not step.get("stop_selected", False) and not step["info"].get("invalid_action_reason", "")]
    deployment_invalid = [step for step in deployment_entries if step["info"].get("invalid_action_reason", "")]
    assert int(details.get("num_adaptive_steps", -1)) == len(deployment_valid)
    assert int(details.get("deployment_steps", -1)) == len(deployment_valid)
    assert int(details.get("deployment_invalid_attempts", -1)) == len(deployment_invalid)


def test_q_learning_deployment_rollout_starts_from_dense_state(monkeypatch, tmp_path: Path) -> None:
    training_workflow = importlib.import_module("gnn_pruning.training.workflow")
    pruning_workflow = importlib.import_module("gnn_pruning.pruning.workflow")
    monkeypatch.setattr(training_workflow, "load_dataset", lambda name, root: _dummy_dataset())
    monkeypatch.setattr(pruning_workflow, "load_dataset", lambda name, root: _dummy_dataset())

    ckpt = _make_checkpoint(tmp_path / "dense_reset.pt", num_layers=3, hidden_channels=16)
    cfg = _make_multilayer_config(tmp_path / "cfg_reset.yaml", tmp_path / "run_reset", episodes=2, max_steps=3)
    artifacts = prune_from_checkpoint(str(ckpt), str(cfg))
    trace = json.loads(artifacts.rl_trace_path.read_text(encoding="utf-8"))

    first_training_state = next(step["state"] for step in trace if step.get("phase") == "training")
    first_deployment_state = next(step["state"] for step in trace if step.get("phase") == "deployment")
    assert first_training_state == first_deployment_state


def test_q_learning_invalid_layer_order_is_filtered_before_selection(monkeypatch, tmp_path: Path) -> None:
    training_workflow = importlib.import_module("gnn_pruning.training.workflow")
    pruning_workflow = importlib.import_module("gnn_pruning.pruning.workflow")
    methods_module = importlib.import_module("gnn_pruning.pruning.methods")
    monkeypatch.setattr(training_workflow, "load_dataset", lambda name, root: _dummy_dataset())
    monkeypatch.setattr(pruning_workflow, "load_dataset", lambda name, root: _dummy_dataset())

    def _deeper_layer_first(*, state_key_value, actions, q_table, epsilon, rng):  # type: ignore[no-untyped-def]
        _ = (state_key_value, q_table, epsilon, rng)
        pruning_actions = [action for action in actions if "layer_index" in action]
        if pruning_actions:
            return max(pruning_actions, key=lambda action: int(action["layer_index"]))
        return dict(actions[0])

    monkeypatch.setattr(methods_module, "select_action", _deeper_layer_first)

    ckpt = _make_checkpoint(tmp_path / "dense_order.pt", num_layers=3, hidden_channels=16)
    cfg = _make_multilayer_config(
        tmp_path / "cfg_order.yaml",
        tmp_path / "run_order",
        step_ratios="[0.50]",
        episodes=1,
        max_steps=3,
        min_channels_per_layer=15,
    )
    artifacts = prune_from_checkpoint(str(ckpt), str(cfg))
    trace = json.loads(artifacts.rl_trace_path.read_text(encoding="utf-8"))
    deployment_entries = [step for step in trace if step.get("phase") == "deployment"]
    invalid_reasons = [str(step["info"].get("invalid_action_reason", "")) for step in deployment_entries]
    assert "invalid_layer_order" not in invalid_reasons
    assert all(step.get("selected_from_valid_actions", False) for step in deployment_entries)


def test_q_learning_nonmonotonic_layer_actions_allowed_when_enabled() -> None:
    methods_module = importlib.import_module("gnn_pruning.pruning.methods")
    from gnn_pruning.models import GraphSAGENodeClassifier
    from gnn_pruning.surgery import structurally_prune_hidden_channels_local

    model = GraphSAGENodeClassifier(in_channels=8, hidden_channels=16, out_channels=3, num_layers=3, dropout=0.0)
    stage1 = structurally_prune_hidden_channels_local(model, layer_index=1, keep_indices=list(range(8)))
    actions = [{"layer_index": 0, "prune_ratio": 0.10}]
    snapshot = methods_module._analyze_action_space(
        model=stage1,
        actions=actions,
        min_channels_per_layer=4,
        pruned_layers={1},
        allow_nonmonotonic_layer_order=True,
        structural_pruning_mode="local",
    )
    assert int(snapshot["num_valid_actions"]) == 1


def test_q_learning_uses_local_surgery_in_local_mode(monkeypatch, tmp_path: Path) -> None:
    training_workflow = importlib.import_module("gnn_pruning.training.workflow")
    pruning_workflow = importlib.import_module("gnn_pruning.pruning.workflow")
    env_module = importlib.import_module("gnn_pruning.rl.environment")
    monkeypatch.setattr(training_workflow, "load_dataset", lambda name, root: _dummy_dataset())
    monkeypatch.setattr(pruning_workflow, "load_dataset", lambda name, root: _dummy_dataset())

    calls = {"local": 0, "cascade": 0}
    real_local = env_module.structurally_prune_hidden_channels_local
    real_cascade = env_module.structurally_prune_hidden_channels

    def _tracked_local(model, layer_index, keep_indices):  # type: ignore[no-untyped-def]
        calls["local"] += 1
        return real_local(model, layer_index=layer_index, keep_indices=keep_indices)

    def _tracked_cascade(model, layer_index, keep_indices):  # type: ignore[no-untyped-def]
        calls["cascade"] += 1
        return real_cascade(model, layer_index=layer_index, keep_indices=keep_indices)

    monkeypatch.setattr(env_module, "structurally_prune_hidden_channels_local", _tracked_local)
    monkeypatch.setattr(env_module, "structurally_prune_hidden_channels", _tracked_cascade)

    ckpt = _make_checkpoint(tmp_path / "dense_local_mode.pt", num_layers=3, hidden_channels=16)
    cfg = _make_multilayer_config(
        tmp_path / "cfg_local_mode.yaml",
        tmp_path / "run_local_mode",
        episodes=1,
        max_steps=2,
        allow_nonmonotonic_layer_order=True,
        structural_pruning_mode="local",
    )
    prune_from_checkpoint(str(ckpt), str(cfg))
    assert calls["local"] > 0
    assert calls["cascade"] == 0


def test_q_learning_invalid_layer_order_preserved_when_flag_false(monkeypatch, tmp_path: Path) -> None:
    training_workflow = importlib.import_module("gnn_pruning.training.workflow")
    pruning_workflow = importlib.import_module("gnn_pruning.pruning.workflow")
    monkeypatch.setattr(training_workflow, "load_dataset", lambda name, root: _dummy_dataset())
    monkeypatch.setattr(pruning_workflow, "load_dataset", lambda name, root: _dummy_dataset())

    ckpt = _make_checkpoint(tmp_path / "dense_order_false.pt", num_layers=3, hidden_channels=16)
    cfg = _make_multilayer_config(
        tmp_path / "cfg_order_false.yaml",
        tmp_path / "run_order_false",
        step_ratios="[0.50]",
        episodes=1,
        max_steps=3,
        min_channels_per_layer=15,
        allow_nonmonotonic_layer_order=False,
        structural_pruning_mode="local",
    )
    artifacts = prune_from_checkpoint(str(ckpt), str(cfg))
    trace = json.loads(artifacts.rl_trace_path.read_text(encoding="utf-8"))
    deployment_entries = [step for step in trace if step.get("phase") == "deployment"]
    assert deployment_entries
    assert any(int(step.get("filtered_actions_by_reason", {}).get("invalid_layer_order", 0)) > 0 for step in deployment_entries)


def test_q_learning_diagnostics_include_structural_feasibility_failed(monkeypatch, tmp_path: Path) -> None:
    training_workflow = importlib.import_module("gnn_pruning.training.workflow")
    pruning_workflow = importlib.import_module("gnn_pruning.pruning.workflow")
    methods_module = importlib.import_module("gnn_pruning.pruning.methods")
    from types import SimpleNamespace

    monkeypatch.setattr(training_workflow, "load_dataset", lambda name, root: _dummy_dataset())
    monkeypatch.setattr(pruning_workflow, "load_dataset", lambda name, root: _dummy_dataset())
    monkeypatch.setattr(
        methods_module,
        "can_apply_structural_prune",
        lambda *args, **kwargs: SimpleNamespace(
            valid=(int(kwargs.get("layer_index", args[1] if len(args) > 1 else -1)) == 1),
            reason="forced_failure",
        ),
    )

    ckpt = _make_checkpoint(tmp_path / "dense_feas_fail.pt", num_layers=3, hidden_channels=16)
    cfg = _make_multilayer_config(
        tmp_path / "cfg_feas_fail.yaml",
        tmp_path / "run_feas_fail",
        episodes=1,
        max_steps=2,
        allow_nonmonotonic_layer_order=True,
        structural_pruning_mode="local",
    )
    artifacts = prune_from_checkpoint(str(ckpt), str(cfg))
    trace = json.loads(artifacts.rl_trace_path.read_text(encoding="utf-8"))
    assert any(int(step.get("filtered_actions_by_reason", {}).get("structural_feasibility_failed", 0)) > 0 for step in trace)


def test_q_learning_stop_action_has_stable_key_and_can_be_selected() -> None:
    stop = {"type": "stop"}
    prune = {"layer_index": 0, "prune_ratio": 0.1}
    assert action_key(stop) == "stop"
    assert action_key(prune) == "layer:0|ratio:0.100000"

    q_table = {"s": {"stop": 1.0, "layer:0|ratio:0.100000": -1.0}}
    selected = select_action(
        state_key_value="s",
        actions=[prune, stop],
        q_table=q_table,
        epsilon=0.0,
        rng=__import__("random").Random(42),
    )
    assert action_key(selected) == "stop"


def test_q_learning_update_q_supports_terminal_stop() -> None:
    q_table: dict[str, dict[str, float]] = {}
    update_q(
        q_table=q_table,
        state_key_value="s0",
        action_key_value="stop",
        reward=0.2,
        next_state_key_value="s1",
        alpha=1.0,
        gamma=0.9,
        terminal=True,
    )
    assert abs(float(q_table["s0"]["stop"]) - 0.2) < 1e-9


def test_q_learning_deployment_can_stop_with_agent_stop(monkeypatch, tmp_path: Path) -> None:
    training_workflow = importlib.import_module("gnn_pruning.training.workflow")
    pruning_workflow = importlib.import_module("gnn_pruning.pruning.workflow")
    methods_module = importlib.import_module("gnn_pruning.pruning.methods")
    monkeypatch.setattr(training_workflow, "load_dataset", lambda name, root: _dummy_dataset())
    monkeypatch.setattr(pruning_workflow, "load_dataset", lambda name, root: _dummy_dataset())

    def _always_stop(*, state_key_value, actions, q_table, epsilon, rng):  # type: ignore[no-untyped-def]
        _ = (state_key_value, q_table, rng)
        if float(epsilon) > 0.0:
            for action in actions:
                if "prune_ratio" in action:
                    return dict(action)
        if _always_stop.deployment_calls == 0:
            _always_stop.deployment_calls += 1
            for action in actions:
                if "prune_ratio" in action:
                    return dict(action)
        for action in actions:
            if action_key(action) == "stop":
                return dict(action)
        return dict(actions[0])
    _always_stop.deployment_calls = 0

    monkeypatch.setattr(methods_module, "select_action", _always_stop)
    ckpt = _make_checkpoint(tmp_path / "dense_stop.pt", num_layers=3, hidden_channels=16)
    cfg = _make_multilayer_config(
        tmp_path / "cfg_stop.yaml",
        tmp_path / "run_stop",
        episodes=1,
        max_steps=5,
        step_ratios="[0.05]",
        max_accuracy_drop=1.0,
    )
    artifacts = prune_from_checkpoint(str(ckpt), str(cfg))
    trace = json.loads(artifacts.rl_trace_path.read_text(encoding="utf-8"))
    deployment_entries = [step for step in trace if step.get("phase") == "deployment"]
    assert deployment_entries
    stop_entries = [step for step in deployment_entries if step.get("stop_selected", False)]
    assert stop_entries
    assert str(stop_entries[-1]["info"].get("stop_reason", "")) == "agent_stop"
    assert float(stop_entries[-1].get("terminal_reward", 0.0)) != 0.0


def test_q_learning_exploration_samples_only_from_valid_actions(monkeypatch, tmp_path: Path) -> None:
    training_workflow = importlib.import_module("gnn_pruning.training.workflow")
    pruning_workflow = importlib.import_module("gnn_pruning.pruning.workflow")
    monkeypatch.setattr(training_workflow, "load_dataset", lambda name, root: _dummy_dataset())
    monkeypatch.setattr(pruning_workflow, "load_dataset", lambda name, root: _dummy_dataset())

    ckpt = _make_checkpoint(tmp_path / "dense_valid_actions.pt", num_layers=3, hidden_channels=16)
    cfg = _make_multilayer_config(
        tmp_path / "cfg_valid_actions.yaml",
        tmp_path / "run_valid_actions",
        step_ratios="[0.0, 0.05, 0.10]",
        episodes=2,
        max_steps=5,
        epsilon_start=1.0,
        epsilon_end=1.0,
        epsilon_decay=1.0,
    )
    artifacts = prune_from_checkpoint(str(ckpt), str(cfg))
    trace = json.loads(artifacts.rl_trace_path.read_text(encoding="utf-8"))
    train_entries = [step for step in trace if step.get("phase") == "training" and step.get("selected_from_valid_actions", False)]
    assert train_entries
    assert all(int(step.get("num_valid_actions", 0)) >= 0 for step in train_entries)
    assert all(not str(step["info"].get("invalid_action_reason", "")).startswith("surgery_failed:IndexError") for step in train_entries)


def test_target_gap_computation_and_bucket_in_state(monkeypatch, tmp_path: Path) -> None:
    training_workflow = importlib.import_module("gnn_pruning.training.workflow")
    pruning_workflow = importlib.import_module("gnn_pruning.pruning.workflow")
    monkeypatch.setattr(training_workflow, "load_dataset", lambda name, root: _dummy_dataset())
    monkeypatch.setattr(pruning_workflow, "load_dataset", lambda name, root: _dummy_dataset())

    ckpt = _make_checkpoint(tmp_path / "dense_gap.pt")
    cfg = _make_config(tmp_path / "cfg_gap.yaml", tmp_path / "run_gap")
    artifacts = prune_from_checkpoint(str(ckpt), str(cfg))
    trace = json.loads(artifacts.rl_trace_path.read_text(encoding="utf-8"))

    assert trace
    assert "target_gap_bucket" in trace[0]["state"]
    assert "target_gap_bucket" in trace[0]
    assert "target_sparsity" in trace[0]
    assert "target_gap" in trace[0]
    q_table = json.loads(artifacts.q_table_path.read_text(encoding="utf-8"))
    if q_table:
        first_state_key = next(iter(q_table.keys()))
        assert "target_gap_bucket" in first_state_key


def test_terminal_reward_penalizes_early_stop_before_target() -> None:
    methods_module = importlib.import_module("gnn_pruning.pruning.methods")
    reward = methods_module._terminal_reward(
        stop_selected=True,
        achieved_sparsity=0.25,
        accuracy_drop=0.01,
        target_sparsity=0.8,
        max_accuracy_drop=0.05,
        stop_reason="agent_stop",
        terminal_cfg={},
    )
    assert reward < 0.0


def test_terminal_reward_stop_near_target_has_tolerant_penalty() -> None:
    methods_module = importlib.import_module("gnn_pruning.pruning.methods")
    reward = methods_module._terminal_reward(
        stop_selected=True,
        achieved_sparsity=0.76,
        accuracy_drop=0.01,
        target_sparsity=0.8,
        max_accuracy_drop=0.05,
        stop_reason="agent_stop",
        terminal_cfg={},
    )
    assert reward > -1e-3


def test_terminal_reward_reaching_target_within_budget_gets_bonus() -> None:
    methods_module = importlib.import_module("gnn_pruning.pruning.methods")
    reward = methods_module._terminal_reward(
        stop_selected=False,
        achieved_sparsity=0.8,
        accuracy_drop=0.02,
        target_sparsity=0.8,
        max_accuracy_drop=0.05,
        stop_reason="target_sparsity_reached",
        terminal_cfg={},
    )
    assert reward > 0.0


def test_terminal_reward_accuracy_failure_before_target_gets_penalty() -> None:
    methods_module = importlib.import_module("gnn_pruning.pruning.methods")
    reward = methods_module._terminal_reward(
        stop_selected=False,
        achieved_sparsity=0.4,
        accuracy_drop=0.2,
        target_sparsity=0.8,
        max_accuracy_drop=0.05,
        stop_reason="max_accuracy_drop_exceeded",
        terminal_cfg={},
    )
    assert reward < 0.0


def test_terminal_reward_nonlinear_squared_gap_penalty() -> None:
    methods_module = importlib.import_module("gnn_pruning.pruning.methods")
    reward_far = methods_module._terminal_reward(
        stop_selected=True,
        achieved_sparsity=0.5,  # gap=0.3 -> effective=0.25 -> severity=1.0 -> factor=1.0
        accuracy_drop=0.01,
        target_sparsity=0.8,
        max_accuracy_drop=0.05,
        stop_reason="agent_stop",
        terminal_cfg={},
    )
    reward_mid = methods_module._terminal_reward(
        stop_selected=True,
        achieved_sparsity=0.625,  # gap=0.175 -> effective=0.125 -> severity=0.5 -> factor=0.25
        accuracy_drop=0.01,
        target_sparsity=0.8,
        max_accuracy_drop=0.05,
        stop_reason="agent_stop",
        terminal_cfg={},
    )
    assert reward_far < reward_mid
    assert abs(float(reward_far) + 0.08) < 1e-6
    assert abs(float(reward_mid) + 0.02) < 1e-6


def test_terminal_reward_no_valid_action_penalty_uses_no_valid_coefficient() -> None:
    methods_module = importlib.import_module("gnn_pruning.pruning.methods")
    reward_no_valid = methods_module._terminal_reward(
        stop_selected=False,
        achieved_sparsity=0.5,
        accuracy_drop=0.01,
        target_sparsity=0.8,
        max_accuracy_drop=0.05,
        stop_reason="no_valid_action",
        terminal_cfg={},
    )
    reward_agent_stop = methods_module._terminal_reward(
        stop_selected=True,
        achieved_sparsity=0.5,
        accuracy_drop=0.01,
        target_sparsity=0.8,
        max_accuracy_drop=0.05,
        stop_reason="agent_stop",
        terminal_cfg={},
    )
    assert reward_no_valid < reward_agent_stop
    assert abs(float(reward_no_valid) + 0.12) < 1e-6


def test_terminal_reward_backward_compatibility_for_old_keys() -> None:
    methods_module = importlib.import_module("gnn_pruning.pruning.methods")
    reward = methods_module._terminal_reward(
        stop_selected=False,
        achieved_sparsity=0.4,
        accuracy_drop=0.2,
        target_sparsity=0.8,
        max_accuracy_drop=0.05,
        stop_reason="max_accuracy_drop_exceeded",
        terminal_cfg={"accuracy_exceeded_penalty": 0.23},
    )
    assert abs(float(reward) + 0.23) < 1e-9


def test_trace_target_gap_uses_actual_current_sparsity_on_no_valid_action(monkeypatch, tmp_path: Path) -> None:
    training_workflow = importlib.import_module("gnn_pruning.training.workflow")
    pruning_workflow = importlib.import_module("gnn_pruning.pruning.workflow")
    monkeypatch.setattr(training_workflow, "load_dataset", lambda name, root: _dummy_dataset())
    monkeypatch.setattr(pruning_workflow, "load_dataset", lambda name, root: _dummy_dataset())

    ckpt = _make_checkpoint(tmp_path / "dense_no_valid_gap.pt", num_layers=3, hidden_channels=16)
    cfg = _make_multilayer_config(
        tmp_path / "cfg_no_valid_gap.yaml",
        tmp_path / "run_no_valid_gap",
        step_ratios="[0.50]",
        episodes=1,
        max_steps=4,
        min_channels_per_layer=15,
    )
    artifacts = prune_from_checkpoint(str(ckpt), str(cfg))
    trace = json.loads(artifacts.rl_trace_path.read_text(encoding="utf-8"))
    no_valid_entries = [
        step for step in trace if step.get("phase") == "deployment" and str(step.get("stop_reason", "")) == "no_valid_action"
    ]
    assert no_valid_entries
    entry = no_valid_entries[-1]
    assert float(entry.get("current_sparsity", 0.0)) > 0.0
    assert float(entry.get("target_gap", 1.0)) < float(entry.get("target_sparsity", 1.0))
    assert "gap_penalty_factor" in entry
    assert "target_tolerance" in entry
    assert "gap_scale" in entry


def test_action_space_filter_diagnostics_present_in_trace(monkeypatch, tmp_path: Path) -> None:
    training_workflow = importlib.import_module("gnn_pruning.training.workflow")
    pruning_workflow = importlib.import_module("gnn_pruning.pruning.workflow")
    monkeypatch.setattr(training_workflow, "load_dataset", lambda name, root: _dummy_dataset())
    monkeypatch.setattr(pruning_workflow, "load_dataset", lambda name, root: _dummy_dataset())

    ckpt = _make_checkpoint(tmp_path / "dense_diag_trace.pt", num_layers=3, hidden_channels=16)
    cfg = _make_multilayer_config(tmp_path / "cfg_diag_trace.yaml", tmp_path / "run_diag_trace", episodes=1, max_steps=3)
    artifacts = prune_from_checkpoint(str(ckpt), str(cfg))
    trace = json.loads(artifacts.rl_trace_path.read_text(encoding="utf-8"))
    assert trace
    first = trace[0]
    assert "hidden_widths_before" in first
    assert "filtered_actions_by_reason" in first
    assert "valid_actions_by_layer" in first
    assert "total_candidate_actions" in first
    assert "num_valid_actions" in first


def test_no_valid_action_terminal_contains_action_space_diagnostics(monkeypatch, tmp_path: Path) -> None:
    training_workflow = importlib.import_module("gnn_pruning.training.workflow")
    pruning_workflow = importlib.import_module("gnn_pruning.pruning.workflow")
    monkeypatch.setattr(training_workflow, "load_dataset", lambda name, root: _dummy_dataset())
    monkeypatch.setattr(pruning_workflow, "load_dataset", lambda name, root: _dummy_dataset())

    ckpt = _make_checkpoint(tmp_path / "dense_no_valid_diag.pt", num_layers=3, hidden_channels=16)
    cfg = _make_multilayer_config(
        tmp_path / "cfg_no_valid_diag.yaml",
        tmp_path / "run_no_valid_diag",
        step_ratios="[0.50]",
        episodes=1,
        max_steps=4,
        min_channels_per_layer=15,
    )
    artifacts = prune_from_checkpoint(str(ckpt), str(cfg))
    trace = json.loads(artifacts.rl_trace_path.read_text(encoding="utf-8"))
    no_valid_entries = [step for step in trace if step.get("phase") == "deployment" and str(step.get("stop_reason", "")) == "no_valid_action"]
    assert no_valid_entries
    entry = no_valid_entries[-1]
    assert "hidden_widths_before" in entry
    assert "filtered_actions_by_reason" in entry
    assert "filtered_action_examples" in entry
    assert "last_pruned_layer" in entry
    assert "stop_available" in entry


def test_action_space_diagnostics_artifact_is_saved(monkeypatch, tmp_path: Path) -> None:
    training_workflow = importlib.import_module("gnn_pruning.training.workflow")
    pruning_workflow = importlib.import_module("gnn_pruning.pruning.workflow")
    monkeypatch.setattr(training_workflow, "load_dataset", lambda name, root: _dummy_dataset())
    monkeypatch.setattr(pruning_workflow, "load_dataset", lambda name, root: _dummy_dataset())

    ckpt = _make_checkpoint(tmp_path / "dense_diag_artifact.pt")
    cfg = _make_config(tmp_path / "cfg_diag_artifact.yaml", tmp_path / "run_diag_artifact")
    artifacts = prune_from_checkpoint(str(ckpt), str(cfg))
    diagnostics_path = artifacts.rl_trace_path.parent / "action_space_diagnostics.json"
    assert diagnostics_path.exists()
    payload = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert "training" in payload
    assert "deployment" in payload
    assert "most_common_filter_reasons" in payload["deployment"]

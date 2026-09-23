"""Config/script checks for large-dataset and Flickr GraphSAGE sweep experiments."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from gnn_pruning.config import load_yaml, resolve_config


def test_flickr_config_resolves() -> None:
    cfg = resolve_config("configs/experiments/presentation_flickr.yaml")
    raw = load_yaml("configs/experiments/presentation_flickr.yaml")
    pruning = raw.get("pruning", {})
    assert cfg.data.name == "flickr"
    assert cfg.model.name == "gcn"
    assert pruning["methods"] == ["random", "global_magnitude"]
    assert pruning["sparsity_levels"] == [0.5, 0.9]


def test_reddit_config_resolves() -> None:
    cfg = resolve_config("configs/experiments/presentation_reddit.yaml")
    raw = load_yaml("configs/experiments/presentation_reddit.yaml")
    pruning = raw.get("pruning", {})
    assert cfg.data.name == "reddit"
    assert cfg.model.name == "gcn"
    assert pruning["methods"] == ["random", "global_magnitude"]


def test_flickr_graphsage_sweep_configs_resolve() -> None:
    expectations = {
        "configs/experiments/flickr_graphsage_l2_h64.yaml": (2, 64),
        "configs/experiments/flickr_graphsage_l2_h128.yaml": (2, 128),
        "configs/experiments/flickr_graphsage_l3_h64.yaml": (3, 64),
        "configs/experiments/flickr_graphsage_l3_h128.yaml": (3, 128),
        "configs/experiments/flickr_graphsage_l4_h128.yaml": (4, 128),
    }
    for path, (layers, hidden) in expectations.items():
        cfg = resolve_config(path)
        raw = load_yaml(path)
        pruning = raw.get("pruning", {})
        assert cfg.data.name == "flickr"
        assert cfg.model.name == "graphsage"
        assert cfg.model.num_layers == layers
        assert cfg.model.hidden_channels == hidden
        methods = pruning.get("methods", [])
        assert "random" in methods
        assert "global_magnitude" in methods
        sparsity_levels = pruning.get("sparsity_levels", [])
        assert 0.5 in sparsity_levels
        assert 0.9 in sparsity_levels


def test_large_dataset_scripts_exist() -> None:
    scripts = [
        Path("scripts/run_large_datasets.py"),
        Path("scripts/run_flickr_graphsage_sweep.py"),
        Path("scripts/summarize_flickr_graphsage_sweep.py"),
    ]
    for script in scripts:
        assert script.exists()
        spec = importlib.util.spec_from_file_location(script.stem, script)
        assert spec is not None
        assert spec.loader is not None


def test_flickr_config_light_smoke() -> None:
    cfg = resolve_config("configs/experiments/flickr_graphsage_l2_h64.yaml")
    assert cfg.training.epochs > 0


def test_pubmed_qlearning_comparison_debug_config_resolves() -> None:
    path = "configs/experiments/rl_qlearning_pubmed_comparison_debug.yaml"
    cfg = resolve_config(path)
    raw = load_yaml(path)
    pruning = raw.get("pruning", {})
    q_learning = raw.get("q_learning", {})
    assert cfg.data.name == "pubmed"
    assert cfg.model.name == "graphsage"
    assert cfg.model.num_layers == 2
    assert cfg.model.hidden_channels == 64
    assert cfg.training.epochs == 30
    assert cfg.device.device in {"cpu", "cuda"}
    assert pruning.get("methods") == ["random", "global_magnitude", "layerwise_magnitude", "snip", "q_learning_tabular"]
    assert pruning.get("sparsity_levels") == [0.5, 0.7]
    assert bool(pruning.get("finetune_enabled", True)) is False
    assert q_learning.get("episodes") == 30
    assert q_learning.get("max_steps") == 16


def test_pubmed_stress_configs_resolve_and_match_filename() -> None:
    expectations = {
        "configs/experiments/stress_pubmed_graphsage_l2_h64.yaml": ("graphsage", 2, 64),
        "configs/experiments/stress_pubmed_graphsage_l3_h32.yaml": ("graphsage", 3, 32),
        "configs/experiments/stress_pubmed_graphsage_l3_h16.yaml": ("graphsage", 3, 16),
        "configs/experiments/stress_pubmed_graphsage_l4_h16.yaml": ("graphsage", 4, 16),
        "configs/experiments/stress_pubmed_gcn_l2_h64.yaml": ("gcn", 2, 64),
        "configs/experiments/stress_pubmed_gcn_l3_h32.yaml": ("gcn", 3, 32),
    }
    required_methods = {"random", "global_magnitude", "layerwise_magnitude", "snip"}
    required_sparsity = {0.5, 0.7, 0.8, 0.9, 0.95}
    for path, (model_name, layers, hidden) in expectations.items():
        cfg = resolve_config(path)
        raw = load_yaml(path)
        pruning = raw.get("pruning", {})
        methods = set(pruning.get("methods", []))
        sparsity_levels = set(float(v) for v in pruning.get("sparsity_levels", []))
        assert cfg.data.name == "pubmed"
        assert cfg.model.name == model_name
        assert cfg.model.num_layers == layers
        assert cfg.model.hidden_channels == hidden
        assert cfg.training.epochs == 30
        assert cfg.training.early_stopping_patience == 10
        assert cfg.device.device in {"cpu", "cuda"}
        assert bool(pruning.get("structured", False)) is True
        assert bool(pruning.get("finetune_enabled", True)) is False
        assert required_methods.issubset(methods)
        assert required_sparsity.issubset(sparsity_levels)


def test_pubmed_qlearning_candidate_configs_resolve() -> None:
    expectations = {
        "configs/experiments/rl_qlearning_pubmed_graphsage_l3_h16.yaml": 16,
        "configs/experiments/rl_qlearning_pubmed_graphsage_l3_h32.yaml": 32,
    }
    for path, hidden in expectations.items():
        cfg = resolve_config(path)
        raw = load_yaml(path)
        pruning = raw.get("pruning", {})
        q_learning = raw.get("q_learning", {})
        assert cfg.data.name == "pubmed"
        assert cfg.model.name == "graphsage"
        assert cfg.model.num_layers == 3
        assert cfg.model.hidden_channels == hidden
        assert cfg.training.epochs == 30
        assert cfg.training.early_stopping_patience == 10
        assert cfg.device.device in {"cpu", "cuda"}
        assert pruning.get("methods") == ["q_learning_tabular"]
        assert float(pruning.get("target_sparsity")) == 0.8
        assert pruning.get("sparsity_levels") == [0.8]
        assert bool(pruning.get("finetune_enabled", True)) is False
        assert int(q_learning.get("episodes", 0)) == 30
        assert int(q_learning.get("max_steps", 0)) == 20
        assert q_learning.get("step_prune_ratios") == [0.05, 0.10]
        reward = q_learning.get("reward", {})
        assert float(reward.get("alpha", -1)) == 0.4
        assert float(reward.get("beta", -1)) == 0.2
        assert float(reward.get("gamma", -1)) == 0.4


def test_pubmed_stress_suite_config_exists() -> None:
    path = Path("configs/suites/pubmed_stress_sweep.yaml")
    assert path.exists()
    payload = load_yaml(path)
    assert payload.get("suite_name") == "pubmed_stress_sweep"
    run = payload.get("run", {})
    assert int(run.get("num_runs", 0)) == 1
    experiments = payload.get("experiments", [])
    assert isinstance(experiments, list)
    assert len(experiments) == 6


def test_pubmed_summary_script_exists() -> None:
    script = Path("scripts/summarize_pubmed_stress_sweep.py")
    assert script.exists()
    spec = importlib.util.spec_from_file_location(script.stem, script)
    assert spec is not None
    assert spec.loader is not None


def test_deeper_graphsage_has_multiple_prunable_layers() -> None:
    cfg_l3 = resolve_config("configs/experiments/stress_pubmed_graphsage_l3_h16.yaml")
    cfg_l4 = resolve_config("configs/experiments/stress_pubmed_graphsage_l4_h16.yaml")
    assert (cfg_l3.model.num_layers - 1) > 1
    assert (cfg_l4.model.num_layers - 1) > 1


def test_pubmed_rl_comparison_config_resolves() -> None:
    path = "configs/experiments/rl_comparison_pubmed_graphsage_l3_h16.yaml"
    cfg = resolve_config(path)
    raw = load_yaml(path)
    pruning = raw.get("pruning", {})
    q_learning = raw.get("q_learning", {})
    assert cfg.data.name == "pubmed"
    assert cfg.model.name == "graphsage"
    assert cfg.model.num_layers == 3
    assert cfg.model.hidden_channels == 16
    assert cfg.training.epochs == 30
    assert cfg.training.early_stopping_patience == 10
    assert cfg.device.device in {"cpu", "cuda"}
    assert bool(pruning.get("structured", False)) is True
    assert bool(pruning.get("finetune_enabled", True)) is False
    assert pruning.get("methods") == ["random", "global_magnitude", "layerwise_magnitude", "snip", "q_learning_tabular"]
    assert pruning.get("sparsity_levels") == [0.7, 0.8]
    assert int(q_learning.get("episodes", 0)) == 30
    assert int(q_learning.get("max_steps", 0)) == 20
    assert q_learning.get("step_prune_ratios") == [0.05, 0.10]
    reward = q_learning.get("reward", {})
    assert float(reward.get("alpha", -1)) == 0.4
    assert float(reward.get("beta", -1)) == 0.2
    assert float(reward.get("gamma", -1)) == 0.4


def test_pubmed_rl_comparison_accdrop_sensitivity_configs_resolve() -> None:
    expectations = {
        "configs/experiments/rl_comparison_pubmed_graphsage_l3_h16_accdrop007.yaml": 0.07,
        "configs/experiments/rl_comparison_pubmed_graphsage_l3_h16_accdrop010.yaml": 0.10,
    }
    for path, expected_accdrop in expectations.items():
        cfg = resolve_config(path)
        raw = load_yaml(path)
        pruning = raw.get("pruning", {})
        q_learning = raw.get("q_learning", {})
        assert cfg.data.name == "pubmed"
        assert cfg.model.name == "graphsage"
        assert cfg.model.num_layers == 3
        assert cfg.model.hidden_channels == 16
        assert bool(pruning.get("structured", False)) is True
        assert bool(pruning.get("finetune_enabled", True)) is False
        assert pruning.get("methods") == ["random", "global_magnitude", "layerwise_magnitude", "snip", "q_learning_tabular"]
        assert pruning.get("sparsity_levels") == [0.7, 0.8]
        assert bool(q_learning.get("allow_nonmonotonic_layer_order", False)) is True
        assert str(q_learning.get("structural_pruning_mode", "")) == "local"
        assert float(q_learning.get("max_accuracy_drop", -1.0)) == expected_accdrop


def _compact_benchmark_matrix() -> list[tuple[str, str, int, int]]:
    datasets = ["cora", "citeseer", "pubmed"]
    models = [
        ("gcn", 2, 64),
        ("gcn", 3, 32),
        ("graphsage", 2, 64),
        ("graphsage", 3, 16),
        ("graphsage", 3, 32),
    ]
    return [(dataset, model, layers, hidden) for dataset in datasets for (model, layers, hidden) in models]


def test_compact_benchmark_static_configs_resolve() -> None:
    expected_grid = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9]
    for dataset, model, layers, hidden in _compact_benchmark_matrix():
        path = f"configs/experiments/compact_benchmark_{dataset}_{model}_l{layers}_h{hidden}_static.yaml"
        cfg = resolve_config(path)
        raw = load_yaml(path)
        pruning = raw.get("pruning", {})
        assert cfg.data.name == dataset
        assert cfg.model.name == model
        assert cfg.model.num_layers == layers
        assert cfg.model.hidden_channels == hidden
        assert cfg.training.epochs == 30
        assert cfg.training.early_stopping_patience == 10
        assert cfg.device.device in {"cpu", "cuda"}
        assert pruning.get("methods") == ["random", "global_magnitude", "layerwise_magnitude", "snip"]
        assert pruning.get("sparsity_levels") == expected_grid
        assert bool(pruning.get("structured", False)) is True
        assert bool(pruning.get("finetune_enabled", True)) is False


def test_compact_benchmark_qlearning_configs_resolve() -> None:
    expected_grid = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9]
    seen_targets: set[float] = set()
    seen_accdrops: set[float] = set()
    for dataset, model, layers, hidden in _compact_benchmark_matrix():
        for expected_accdrop, suffix in [(0.05, "005"), (0.07, "007"), (0.10, "010")]:
            path = (
                f"configs/experiments/compact_benchmark_{dataset}_{model}_l{layers}_h{hidden}"
                f"_qlearning_accdrop{suffix}.yaml"
            )
            cfg = resolve_config(path)
            raw = load_yaml(path)
            pruning = raw.get("pruning", {})
            q_learning = raw.get("q_learning", {})
            assert cfg.data.name == dataset
            assert cfg.model.name == model
            assert cfg.model.num_layers == layers
            assert cfg.model.hidden_channels == hidden
            assert cfg.training.epochs == 30
            assert cfg.training.early_stopping_patience == 10
            assert cfg.device.device in {"cpu", "cuda"}
            assert pruning.get("methods") == ["q_learning_tabular"]
            assert pruning.get("sparsity_levels") == expected_grid
            assert bool(pruning.get("structured", False)) is True
            assert bool(pruning.get("finetune_enabled", True)) is False
            assert bool(q_learning.get("allow_nonmonotonic_layer_order", False)) is True
            assert str(q_learning.get("structural_pruning_mode", "")) == "local"
            assert int(q_learning.get("episodes", 0)) == 30
            assert int(q_learning.get("max_steps", 0)) == 20
            assert q_learning.get("step_prune_ratios") == [0.05, 0.10]
            assert float(q_learning.get("max_accuracy_drop", -1.0)) == expected_accdrop
            seen_accdrops.add(float(q_learning.get("max_accuracy_drop", -1.0)))
            seen_targets.update(float(v) for v in pruning.get("sparsity_levels", []))
    assert seen_targets == set(expected_grid)
    assert seen_accdrops == {0.05, 0.07, 0.10}


def test_compact_benchmark_suite_config_resolves() -> None:
    path = Path("configs/suites/compact_pruning_benchmark.yaml")
    assert path.exists()
    payload = load_yaml(path)
    assert payload.get("suite_name") == "compact_pruning_benchmark"
    run = payload.get("run", {})
    assert int(run.get("num_runs", 0)) == 3
    assert int(run.get("base_seed", 0)) == 42
    experiments = payload.get("experiments", [])
    assert isinstance(experiments, list)
    assert len(experiments) == (3 * 5) + (3 * 5 * 3)
    for experiment_path in experiments:
        assert Path(str(experiment_path)).exists()


def test_compact_benchmark_summary_script_exists() -> None:
    script = Path("scripts/summarize_compact_pruning_benchmark.py")
    assert script.exists()
    spec = importlib.util.spec_from_file_location(script.stem, script)
    assert spec is not None
    assert spec.loader is not None

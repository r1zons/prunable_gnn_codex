"""Basic smoke tests for package wiring."""

from gnn_pruning.cli import main


def test_cli_version_returns_success() -> None:
    """CLI version command should return successful exit code."""
    exit_code = main(["version"])
    assert exit_code == 0


def test_cli_show_config_returns_success() -> None:
    """CLI show-config command should return successful exit code."""
    exit_code = main(["show-config", "--config", "configs/experiments/example.yaml"])
    assert exit_code == 0


def test_cli_train_and_evaluate_commands(monkeypatch, tmp_path) -> None:
    from gnn_pruning import cli
    from gnn_pruning.pipelines.dense_pipeline import DensePipelineArtifacts
    from gnn_pruning.training.workflow import EvalArtifacts, TrainArtifacts

    train_artifacts = TrainArtifacts(
        resolved_config_path=tmp_path / "resolved_config.yaml",
        split_path=tmp_path / "splits.yaml",
        checkpoint_path=tmp_path / "dense_checkpoint.pt",
        metrics_path=tmp_path / "metrics_train.json",
    )
    dense_artifacts = DensePipelineArtifacts(
        output_dir=tmp_path,
        checkpoint_path=tmp_path / "dense_checkpoint.pt",
        train_metrics_path=tmp_path / "metrics_train.json",
        eval_metrics_path=tmp_path / "metrics_eval.json",
        config_snapshot_path=tmp_path / "resolved_config.yaml",
        split_path=tmp_path / "splits.yaml",
        summary_path=tmp_path / "summary.md",
        csv_path=tmp_path / "dense_results.csv",
    )

    eval_artifacts = EvalArtifacts(
        resolved_config_path=tmp_path / "resolved_config.yaml",
        metrics_path=tmp_path / "metrics_eval.json",
    )

    monkeypatch.setattr(cli, "train_dense", lambda config_path, resume: train_artifacts)
    monkeypatch.setattr(cli, "evaluate_dense_and_save", lambda config_path, checkpoint_path: eval_artifacts)
    monkeypatch.setattr(cli, "run_dense_pipeline", lambda config_path: dense_artifacts)

    assert main(["train", "--config", "configs/experiments/example.yaml", "--no-resume"]) == 0
    assert main(["evaluate", "--config", "configs/experiments/example.yaml"]) == 0
    assert main(["run-dense", "--config", "configs/experiments/example.yaml"]) == 0


def test_cli_list_pruners_command() -> None:
    assert main(["list-pruners"]) == 0

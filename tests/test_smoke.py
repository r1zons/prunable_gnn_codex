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

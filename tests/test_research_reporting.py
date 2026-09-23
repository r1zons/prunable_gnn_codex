"""Regression tests for validation-based research result selection."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_script(name: str):
    path = Path("scripts") / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_flickr_best_summary_selects_by_validation_not_test(tmp_path: Path) -> None:
    module = _load_script("run_flickr_graphsage_sweep.py")
    rows = [
        {
            "phase": "post_finetune",
            "method": "random",
            "val_accuracy": "0.80",
            "test_accuracy": "0.95",
            "achieved_sparsity": "0.50",
        },
        {
            "phase": "post_finetune",
            "method": "global_magnitude",
            "val_accuracy": "0.85",
            "test_accuracy": "0.70",
            "achieved_sparsity": "0.49",
        },
    ]
    output = tmp_path / "best_summary.json"

    module._write_best_summary(rows, output)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["selection_metric"] == "val_accuracy"
    assert payload["validation_selected_post_finetune"]["method"] == "global_magnitude"
    assert "not selection inputs" in payload["note"]


def test_flickr_best_summary_does_not_fall_back_to_test_for_selection(tmp_path: Path) -> None:
    module = _load_script("run_flickr_graphsage_sweep.py")
    output = tmp_path / "best_summary.json"

    module._write_best_summary(
        [{"phase": "post_finetune", "method": "random", "test_accuracy": "0.99"}],
        output,
    )

    assert not output.exists()

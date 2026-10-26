"""Regression for launcher output naming; substitutes the GNN subprocess entirely."""
from pathlib import Path
import importlib.util
import subprocess
import sys
import tempfile
from unittest.mock import patch

root = Path(__file__).resolve().parent

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

adapter = load('polus_adapter_check', root / 'prepare-gnn-pilot.py')
runner = load('polus_runner_check', root / 'repo/scripts/run_kaggle_pilot.py')
adapter.adapt_runner_log_directory(runner)
assert 'torch' not in sys.modules and 'torch_geometric' not in sys.modules
with tempfile.TemporaryDirectory(prefix='polus-launcher-check-') as temporary:
    runner.OUTPUT_ROOT = Path(temporary)
    condition = 'cora_gcn_l3_h32'
    runner.CONFIGS = {condition: '/not-executed.yaml'}
    expected = runner.OUTPUT_ROOT / 'seed_101' / condition
    calls = []
    def fake_run(command, **kwargs):
        assert not expected.exists(), 'Launcher pre-created the pipeline output'
        expected.mkdir(parents=True)
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)
    runner._validate_condition = lambda directory, name: []
    runner._extract_convergence = lambda path: []
    with patch.object(runner.subprocess, 'run', side_effect=fake_run):
        runner._run_condition(root / 'repo', condition, '72293149d8fff186dd268789e7b8fea5848c9360')
    assert len(calls) == 1
    assert (expected / 'condition_complete.json').exists()
    assert (runner.OUTPUT_ROOT / 'launcher-logs' / (condition + '.log')).exists()
    assert not expected.with_name(condition + '_001').exists()
assert 'torch' not in sys.modules and 'torch_geometric' not in sys.modules
print('LAUNCHER_DIRECTORY_REGRESSION_PASS; scientific subprocess was replaced, never executed.')

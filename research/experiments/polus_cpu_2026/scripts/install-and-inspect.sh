#!/usr/bin/env bash
# Installs the built wheel and inspects imports/configs; never starts training.
set -euo pipefail
task_root="$HOME/gnn-polus-20261023"
source "$task_root/activate-polus.sh"
cd "$task_root"
wheel="$(python - <<'PY'
from pathlib import Path
import os
paths = list((Path(os.environ['HOME']) / 'gnn-polus-20261023/sources/pytorch-v2.0.1/dist').glob('torch-2.0.1+polus.power8.1-*.whl'))
if len(paths) != 1:
    raise SystemExit('Expected exactly one completed POWER8 wheel, found {}'.format(len(paths)))
print(paths[0])
PY
)"
sha256sum "$wheel" > logs/torch-wheel.sha256
python -m pip install --no-deps "$wheel" > logs/torch-install.log 2>&1
python -m pip check > logs/pip-check.txt
python - <<'PY'
import json
from importlib import metadata
from pathlib import Path
import platform
import sys
import numpy
import scipy
import pandas
import sklearn
import torch
import torch_geometric
payload = {
    'machine': platform.machine(), 'python': sys.version,
    'torch': torch.__version__, 'torch_distribution': metadata.version('torch'),
    'pyg': torch_geometric.__version__,
    'numpy': numpy.__version__, 'scipy': scipy.__version__,
    'pandas': pandas.__version__, 'scikit_learn': sklearn.__version__,
    'cuda_runtime': torch.version.cuda, 'cuda_available': torch.cuda.is_available(),
    'openmp_available': torch.backends.openmp.is_available(),
    'training_or_pruning_executed': False,
}
Path('logs/runtime-versions.json').write_text(json.dumps(payload, indent=2))
Path('logs/torch-build-config.txt').write_text(torch.__config__.show())
print(json.dumps(payload, indent=2))
PY
python -m gnn_pruning list-pruners > logs/pruner-capabilities.txt
for config in results/pilot-cpu-101/prepared-configs/*.yaml; do
    name="$(basename "$config" .yaml)"
    python -m gnn_pruning show-config --config "$config" > "logs/resolved-$name.txt"
done
python -m pip freeze > logs/pip-freeze-final.txt
conda list --explicit > logs/conda-explicit-final.txt
git -C repo rev-parse HEAD > logs/research-commit.txt
git -C repo status --porcelain > logs/research-status.txt
printf 'ENVIRONMENT_INSTALLED_AND_INSPECTED\nNo training or pruning was started.\n'

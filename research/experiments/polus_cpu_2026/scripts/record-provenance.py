"""Record metadata after successful setup; imports no model or dataset code."""
from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--build-job', required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parent
logs = root / 'logs'
ready_path = logs / 'environment-ready.json'
ready = json.loads(ready_path.read_text())
if ready['status'] != 'READY_FOR_MANUAL_PILOT' or ready['vsx_regression'] != 'PASS':
    raise SystemExit('Successful environment inspection is required first')


def git(path: Path, *arguments: str) -> str:
    return subprocess.check_output(['git', '-C', str(path), *arguments], text=True)


if git(root / 'repo', 'status', '--porcelain'):
    raise SystemExit('Research repository is not clean')
patch = (root / 'patches/pytorch-vsx.diff').read_bytes()
actual = git(root / 'sources/pytorch-v2.0.1', 'diff', '--no-ext-diff', '--ignore-submodules=dirty').encode()
if actual != patch:
    raise SystemExit('Source changes differ from the recorded VSX backport')
runtime = json.loads((logs / 'runtime-versions.json').read_text())
runtime['torch_distribution'] = metadata.version('torch')
if runtime['torch_distribution'] != '2.0.1+polus.power8.1':
    raise SystemExit('Unexpected installed dependency distribution')
(logs / 'runtime-versions.json').write_text(json.dumps(runtime, indent=2))
ready['runtime'] = runtime
ready['build_job'] = args.build_job
ready['torch_source_commit'] = git(root / 'sources/pytorch-v2.0.1', 'rev-parse', 'HEAD').strip()
ready['source_patch_sha256'] = hashlib.sha256(patch).hexdigest()
ready['resolved_config_count'] = len(list(logs.glob('resolved-*.txt')))
ready['pip_check'] = (logs / 'pip-check.txt').read_text().strip()
(logs / 'torch-source-commit.txt').write_text(ready['torch_source_commit'] + '\n')
(logs / 'torch-root-submodules.txt').write_text(git(root / 'sources/pytorch-v2.0.1', 'submodule', 'status'))
names = ['activate-polus.sh', 'build-torch-cpu.sh', 'build-cpu.lsf', 'install-and-inspect.sh',
         'inspect-env.lsf', 'check-vsx-shifts.py', 'pilot-cpu.lsf', 'prepare-gnn-pilot.py',
         'apply-vsx-backport.py', 'materialize-source-archives.py', 'record-provenance.py']
(logs / 'setup-scripts.sha256').write_text(''.join(
    '{}  {}\n'.format(hashlib.sha256((root / name).read_bytes()).hexdigest(), name)
    for name in names
))
temporary = ready_path.with_suffix('.tmp')
temporary.write_text(json.dumps(ready, indent=2))
temporary.replace(ready_path)
print(json.dumps(ready, indent=2))

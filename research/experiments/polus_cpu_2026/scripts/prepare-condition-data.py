"""Copy verified raw Planetoid inputs; never unpickle data or import GNN code."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--destination', type=Path, required=True)
parser.add_argument('--condition', required=True)
args = parser.parse_args()
dataset = {'cora': 'Cora', 'citeseer': 'CiteSeer', 'pubmed': 'PubMed'}[args.condition.split('_', 1)[0]]
source = args.source.resolve()
destination = args.destination.resolve()
if source == destination:
    raise SystemExit('Source and isolated data directories must differ')
manifest = json.loads((source / 'raw-manifest.json').read_text())
rows = [row for row in manifest if Path(row['file']).parts[:3] == ('planetoid', dataset, 'raw')]
if len(rows) != 8:
    raise SystemExit('Expected exactly eight raw files for this dataset')
for row in rows:
    relative = Path(row['file'])
    if relative.is_absolute() or '..' in relative.parts:
        raise SystemExit('Invalid raw manifest path')
    original = source / relative
    target = destination / relative
    if hashlib.sha256(original.read_bytes()).hexdigest() != row['sha256']:
        raise SystemExit('Source checksum mismatch: ' + str(original))
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copy2(original, target)
    if hashlib.sha256(target.read_bytes()).hexdigest() != row['sha256']:
        raise SystemExit('Isolated raw checksum mismatch: ' + str(target))
(destination / 'raw-manifest.json').write_text(json.dumps(rows, indent=2))
print('Verified isolated raw input:', args.condition, destination)

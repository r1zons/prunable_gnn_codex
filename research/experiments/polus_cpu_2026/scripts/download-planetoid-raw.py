"""Download only the raw files used by installed PyG 2.4; do not unpickle them."""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

root = Path(sys.argv[1]).expanduser().resolve()
base_url = "https://github.com/kimiyoung/planetoid/raw/master/data"
parts = ["x", "tx", "allx", "y", "ty", "ally", "graph", "test.index"]


def download(item: tuple[str, str]) -> dict:
    name, part = item
    filename = "ind.{}.{}".format(name.lower(), part)
    target = root / "planetoid" / name / "raw" / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    url = base_url + "/" + filename
    if not target.exists():
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
        temporary = target.with_name(target.name + ".download")
        temporary.write_bytes(data)
        temporary.replace(target)
    content = target.read_bytes()
    print(name, filename, len(content), flush=True)
    return {"file": str(target.relative_to(root)), "url": url,
            "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    rows = list(pool.map(download, [(name, part) for name in ["Cora", "CiteSeer", "PubMed"] for part in parts]))
(root / "raw-manifest.json").write_text(json.dumps(rows, indent=2))
print("Downloaded raw files:", len(rows), flush=True)

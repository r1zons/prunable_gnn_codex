"""Restore pinned CPU-build sources using small Git metadata plus source archives."""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.parse
import urllib.request

root = Path(sys.argv[1]).resolve()
task = root.parent.parent
git = str(task / "env/bin/git")
archive_dir = task / "source-archives"
archive_dir.mkdir(exist_ok=True)


def read(repo: Path, *args: str) -> str:
    return subprocess.check_output([git, "-C", str(repo), *args], text=True,
                                   stderr=subprocess.DEVNULL).strip()


def run(*args: str) -> None:
    subprocess.run([git, "-c", "http.version=HTTP/1.1", *args], check=True, timeout=180)


def entries(repo: Path):
    modules = repo / ".gitmodules"
    if not modules.exists():
        return []
    config = read(repo, "config", "-f", str(modules), "--get-regexp", r"^submodule\..*\.path$")
    result = []
    for line in config.splitlines():
        key, path = line.split(maxsplit=1)
        fields = read(repo, "ls-tree", "HEAD", "--", path).split()
        if not fields or fields[0] != "160000":
            continue
        name = key[len("submodule."):-len(".path")]
        url = read(repo, "config", "-f", str(modules), "--get", "submodule." + name + ".url")
        result.append((repo, path, url, fields[2]))
    return result


def hydrate(entry):
    parent, path, url, revision = entry
    target = parent / path
    markers = ["CMakeLists.txt", "Makefile", "setup.py", "LICENSE", "LICENSE.md", "LICENSE.txt"]
    try:
        complete = (read(target, "rev-parse", "HEAD") == revision
                    and any((target / name).is_file() for name in markers))
    except subprocess.CalledProcessError:
        complete = False
    if complete:
        print("REUSED", target.relative_to(root), revision, flush=True)
        return None
    if not url.startswith("https://github.com/"):
        raise RuntimeError("Unsupported source archive URL: " + url)
    repo_path = urllib.parse.urlparse(url).path.strip("/").removesuffix(".git")
    archive_url = "https://codeload.github.com/{}/tar.gz/{}".format(repo_path, revision)
    archive = archive_dir / (repo_path.replace("/", "-") + "-" + revision + ".tar.gz")
    if not archive.exists():
        with urllib.request.urlopen(archive_url, timeout=60) as response:
            content = response.read()
        archive.write_bytes(content)
    if target.exists():
        backup = task / "source-download-partials" / "archive-replaced" / target.relative_to(root)
        backup.parent.mkdir(parents=True, exist_ok=True)
        if backup.exists():
            raise RuntimeError("Backup already exists: " + str(backup))
        target.rename(backup)
    target.mkdir(parents=True)
    run("init", str(target))
    run("-C", str(target), "remote", "add", "origin", url)
    # Fetch commits/trees only; large blob payloads arrive through codeload.
    run("-C", str(target), "-c", "remote.origin.promisor=true", "-c",
        "remote.origin.partialclonefilter=blob:none", "fetch", "--filter=blob:none",
        "--depth=1", "origin", revision)
    run("-C", str(target), "reset", "--mixed", "FETCH_HEAD")
    with tempfile.TemporaryDirectory(dir=archive_dir) as temporary:
        with tarfile.open(archive) as bundle:
            bundle.extractall(temporary, filter="data")
        tops = list(Path(temporary).iterdir())
        if len(tops) != 1 or not tops[0].is_dir():
            raise RuntimeError("Unexpected source archive layout")
        for child in tops[0].iterdir():
            shutil.move(str(child), str(target / child.name))
    subprocess.run([git, "-C", str(target), "update-index", "--refresh"],
                   check=False, stdout=subprocess.DEVNULL)
    receipt = {"path": str(target.relative_to(root)), "revision": revision,
               "archive_url": archive_url,
               "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}
    print("ARCHIVE", json.dumps(receipt), flush=True)
    return receipt


if read(root, "describe", "--tags", "--exact-match", "HEAD") != "v2.0.1":
    raise RuntimeError("Expected PyTorch v2.0.1")
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    receipts = [item for item in pool.map(hydrate, entries(root)) if item]
# setup.py explicitly checks these even with CPU-only options.
for parent_name, nested_path in [("fbgemm", "third_party/asmjit"), ("onnx", "third_party/benchmark")]:
    parent = root / "third_party" / parent_name
    for entry in entries(parent):
        if entry[1] == nested_path:
            receipt = hydrate(entry)
            if receipt:
                receipts.append(receipt)
for parent in [root, root / "third_party/fbgemm", root / "third_party/onnx"]:
    run("-C", str(parent), "submodule", "init")
(archive_dir / "materialization-manifest.json").write_text(json.dumps(receipts, indent=2))
print("PINNED ROOT SOURCES READY", flush=True)

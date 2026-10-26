"""Fetch only the exact gitlink revisions of a new PyTorch source checkout."""
from __future__ import annotations

import concurrent.futures
import os
from pathlib import Path
import subprocess
import sys
import urllib.parse


ROOT = Path(sys.argv[1]).resolve()
BACKUP = ROOT.parent.parent / "source-download-partials"
GIT = str(ROOT.parent.parent / "env/bin/git")
OPTIONS = ["-c", "http.version=HTTP/1.1", "-c", "http.lowSpeedLimit=1024",
           "-c", "http.lowSpeedTime=45"]


def output(repo: Path, *args: str) -> str:
    return subprocess.check_output([GIT, "-C", str(repo), *args], text=True,
                                   stderr=subprocess.DEVNULL).strip()


def run(*args: str) -> None:
    subprocess.run([GIT, *OPTIONS, *args], check=True, timeout=900)


def entries(repo: Path, parent_url: str) -> list[tuple[Path, str, str]]:
    modules = repo / ".gitmodules"
    if not modules.exists():
        return []
    try:
        config = output(repo, "config", "-f", str(modules), "--get-regexp", r"^submodule\..*\.path$")
    except subprocess.CalledProcessError as exc:
        if exc.returncode == 1:
            return []
        raise
    result = []
    for line in config.splitlines():
        key, path = line.split(maxsplit=1)
        name = key[len("submodule."):-len(".path")]
        url = output(repo, "config", "-f", str(modules), "--get", "submodule." + name + ".url")
        if url.startswith("."):
            url = urllib.parse.urljoin(parent_url.rstrip("/") + "/", url)
        fields = output(repo, "ls-tree", "HEAD", "--", path).split()
        if len(fields) < 3 or fields[0] != "160000":
            # Some historical .gitmodules list entries absent from the tree.
            continue
        run("-C", str(repo), "submodule", "init", "--", path)
        target = (repo / path).resolve()
        if ROOT not in target.parents:
            raise RuntimeError("Submodule escapes source checkout")
        result.append((target, url, fields[2]))
    return result


def hydrate(entry: tuple[Path, str, str]) -> None:
    target, url, revision = entry
    valid = False
    if (target / ".git").exists():
        try:
            valid = Path(output(target, "rev-parse", "--show-toplevel")).resolve() == target
            current = output(target, "rev-parse", "HEAD")
        except subprocess.CalledProcessError:
            valid = False
    if valid:
        if current != revision:
            run("-C", str(target), "fetch", "--depth=1", "origin", revision)
            run("-C", str(target), "checkout", "--detach", revision)
    else:
        if target.exists():
            backup = BACKUP / target.relative_to(ROOT)
            backup.parent.mkdir(parents=True, exist_ok=True)
            if backup.exists():
                backup = backup.with_name(backup.name + ".retry-" + str(os.getpid()))
            target.rename(backup)
        target.parent.mkdir(parents=True, exist_ok=True)
        run("clone", "--revision=" + revision, "--depth=1", url, str(target))
    if output(target, "rev-parse", "HEAD") != revision:
        raise RuntimeError("Submodule revision mismatch: " + str(target))
    print("PINNED", target.relative_to(ROOT), revision, flush=True)
    for child in entries(target, url):
        hydrate(child)


if output(ROOT, "describe", "--tags", "--exact-match", "HEAD") != "v2.0.1":
    raise RuntimeError("Expected upstream PyTorch v2.0.1")
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    list(executor.map(hydrate, entries(ROOT, "https://github.com/pytorch/pytorch.git")))
print(output(ROOT, "submodule", "status", "--recursive"), flush=True)

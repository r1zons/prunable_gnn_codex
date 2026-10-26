"""Backport PyTorch PRs 98511/109886; adapt only old helper-file context."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

root = Path(sys.argv[1]).resolve()
task = root.parent.parent
git = str(task / "env/bin/git")
patches = task / "patches"
first = patches / "upstream-98511.diff"
second = patches / "upstream-109886.diff"
helper_name = "aten/src/ATen/cpu/vec/vec256/vsx/vsx_helpers.h"
helper = root / helper_name


def run(*args: str) -> None:
    subprocess.run([git, "-C", str(root), *args], check=True)


current = subprocess.check_output([git, "-C", str(root), "diff", "--no-ext-diff", "--ignore-submodules=dirty"])
recorded = patches / "pytorch-vsx.diff"
if recorded.exists() and recorded.read_bytes() == current:
    text = helper.read_text()
    if "#include <climits>" not in text:
        anchor = "#include <ATen/cpu/vec/vec_base.h>\n"
        if text.count(anchor) != 1:
            raise RuntimeError("Unexpected backported helper include context")
        helper.write_text(text.replace(anchor, anchor + "\n#include <climits>\n", 1))
        current = subprocess.check_output([git, "-C", str(root), "diff", "--no-ext-diff", "--ignore-submodules=dirty"])
        recorded.write_bytes(current)
        manifest_path = patches / "vsx-backport-manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["backport_diff_sha256"] = hashlib.sha256(current).hexdigest()
        manifest["portability_include"] = "Explicit <climits> for CHAR_BIT on the older PyTorch 2.0.1 include graph."
        manifest_path.write_text(json.dumps(manifest, indent=2))
    print("Recorded VSX backport already present")
    raise SystemExit(0)
if not current:
    run("apply", "--check", str(first))
    run("apply", str(first))
else:
    # This session already applied PR 98511 before detecting the old header context.
    run("apply", "--reverse", "--check", str(first))

text = helper.read_text()
section = second.read_text().split("diff --git a/" + helper_name + " b/" + helper_name)[1]
hunks = re.split(r"(?m)^@@[^\n]*\n", section)[1:]
added = ["\n".join(line[1:] for line in hunk.splitlines()
                   if line.startswith("+") and not line.startswith("+++")) + "\n"
         for hunk in hunks]
if len(added) != 3:
    raise RuntimeError("Unexpected upstream helper patch structure")
added[0] += "\n#include <climits>\n"
anchors = [
    "#include <ATen/cpu/vec/intrinsics.h>\n",
    "using vfloat64 =  __attribute__((altivec(vector__)))  double;\n",
    "const vfloat64 vd_pi_2 = vfloat64{M_PI / 2.0, 0.0};\n",
]
for anchor, insertion in zip(anchors, added):
    if text.count(anchor) != 1:
        raise RuntimeError("Unexpected PyTorch 2.0.1 helper context")
    text = text.replace(anchor, anchor + "\n" + insertion, 1)
run("apply", "--check", "--exclude=" + helper_name, str(second))
run("apply", "--exclude=" + helper_name, str(second))
helper.write_text(text)
diff = subprocess.check_output([git, "-C", str(root), "diff", "--no-ext-diff", "--ignore-submodules=dirty"])
recorded.write_bytes(diff)
manifest = {
    "upstream_prs": ["https://github.com/pytorch/pytorch/pull/98511", "https://github.com/pytorch/pytorch/pull/109886"],
    "upstream_diff_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [first, second]},
    "backport_diff_sha256": hashlib.sha256(diff).hexdigest(),
    "adaptation": "Only helper include/type-declaration context differs from newer upstream; inserted code is copied verbatim from PR 109886.",
    "portability_include": "Explicit <climits> for CHAR_BIT on the older PyTorch 2.0.1 include graph.",
}
(patches / "vsx-backport-manifest.json").write_text(json.dumps(manifest, indent=2))
run("diff", "--stat", "--ignore-submodules=dirty")
print(json.dumps(manifest, indent=2))

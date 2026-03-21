"""Dedicated DBLP adapter path.

DBLP in PyG is heterogeneous and requires explicit handling. This adapter keeps the
code path explicit and prevents accidental treatment as a homogeneous graph.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union


def load_dblp_dataset(root: Union[str, Path]) -> Any:
    """Placeholder for DBLP heterogeneous adapter implementation."""
    _ = root
    raise NotImplementedError(
        "DBLP loading must go through a dedicated heterogeneous adapter. "
        "TODO: implement explicit conversion/selection strategy for node classification."
    )

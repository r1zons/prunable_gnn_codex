"""General utility functions for gnn_pruning."""

from .output_dir import resolve_output_dir
from .seed import set_seed

__all__ = ["set_seed", "resolve_output_dir"]

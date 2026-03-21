"""Layer surgery subsystem."""

from .structural import structurally_prune_hidden_channels, validate_structural_compression

__all__ = ["structurally_prune_hidden_channels", "validate_structural_compression"]

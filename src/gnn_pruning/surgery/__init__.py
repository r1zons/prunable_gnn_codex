"""Layer surgery subsystem."""

from .structural import (
    FeasibilityResult,
    can_apply_structural_prune,
    structurally_prune_hidden_channels,
    structurally_prune_hidden_channels_local,
    validate_structural_compression,
)

__all__ = [
    "FeasibilityResult",
    "can_apply_structural_prune",
    "structurally_prune_hidden_channels",
    "structurally_prune_hidden_channels_local",
    "validate_structural_compression",
]

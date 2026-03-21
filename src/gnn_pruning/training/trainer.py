"""Trainer scaffolding."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Trainer:
    """Minimal trainer scaffold for future dense/finetune loops."""

    def fit(self) -> None:
        """Run training loop placeholder."""
        return None

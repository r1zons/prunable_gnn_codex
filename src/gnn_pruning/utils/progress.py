"""Lightweight progress reporting helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from . import simple_yaml


@dataclass
class ProgressReporter:
    """Conditional stdout + file progress reporter."""

    enabled: bool
    log_path: Path | None = None

    def stage(self, index: int, total: int, message: str) -> None:
        self._emit(f"[{index}/{total}] {message}")

    def info(self, message: str) -> None:
        self._emit(message)

    def epoch(
        self,
        epoch: int,
        max_epochs: int,
        train_loss: float,
        val_loss: float,
        best_epoch: int,
        early_stopping_counter: int,
        train_acc: float | None = None,
        val_acc: float | None = None,
        elapsed_sec: float | None = None,
    ) -> None:
        payload = [
            f"Epoch {epoch}/{max_epochs}",
            f"train_loss={train_loss:.4f}",
            f"val_loss={val_loss:.4f}",
        ]
        if train_acc is not None:
            payload.append(f"train_acc={train_acc:.4f}")
        if val_acc is not None:
            payload.append(f"val_acc={val_acc:.4f}")
        payload.append(f"best_epoch={best_epoch}")
        payload.append(f"early_stop={early_stopping_counter}")
        if elapsed_sec is not None:
            payload.append(f"elapsed_sec={elapsed_sec:.2f}")
        self._emit(" | ".join(payload))

    def phase_metrics(self, phase: str, metrics: Dict[str, Any]) -> None:
        test = metrics.get("test", {}) if isinstance(metrics.get("test", {}), dict) else {}
        benchmark = metrics.get("benchmark", {}) if isinstance(metrics.get("benchmark", {}), dict) else {}
        self._emit(
            f"{phase}: acc={test.get('accuracy', 'n/a')} "
            f"macro_f1={test.get('macro_f1', 'n/a')} "
            f"parameter_count={benchmark.get('parameter_count', 'n/a')} "
            f"inference_time_mean_sec={_ms_to_sec(benchmark.get('inference_time_mean_ms'))}"
        )

    def _emit(self, message: str) -> None:
        if not self.enabled:
            return
        stamped = f"{datetime.now(timezone.utc).isoformat()} | {message}"
        print(message)
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(stamped + "\n")


def resolve_show_progress(config_path: str, cli_progress: bool | None) -> bool:
    """Resolve progress flag with CLI override over config."""
    if cli_progress is not None:
        return bool(cli_progress)
    path = Path(config_path).expanduser()
    payload = simple_yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        return False
    logging_cfg = payload.get("logging", {}) if isinstance(payload.get("logging", {}), dict) else {}
    return bool(logging_cfg.get("show_progress", False))


def _ms_to_sec(value: Any) -> Any:
    if value in ("", None):
        return "n/a"
    return round(float(value) / 1000.0, 6)

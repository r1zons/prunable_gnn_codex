"""Typed configuration schema for experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

SUPPORTED_DBLP_STRATEGIES = ("author_homogeneous",)


@dataclass
class RunConfig:
    """Runtime and output settings."""

    seed: int = 42
    output_dir: str = "runs/default"
    experiment_name: str = "default_experiment"


@dataclass
class DataConfig:
    """Dataset settings."""

    name: str = "cora"
    root: str = "./data"
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    dblp_strategy: str = "author_homogeneous"


@dataclass
class ModelConfig:
    """Model settings."""

    name: str = "gcn"
    in_channels: Optional[int] = None
    hidden_channels: int = 64
    out_channels: Optional[int] = None
    num_layers: int = 2
    dropout: float = 0.5


@dataclass
class TrainingConfig:
    """Training and optimization settings."""

    epochs: int = 200
    lr: float = 0.01
    weight_decay: float = 5e-4
    early_stopping_patience: int = 50


@dataclass
class DeviceConfig:
    """Device selection settings."""

    device: str = "cpu"




@dataclass
class BenchmarkConfig:
    """Benchmark settings."""

    inference_warmup_passes: int = 5
    inference_timed_passes: int = 20

@dataclass
class ExperimentConfig:
    """Resolved experiment config."""

    run: RunConfig = field(default_factory=RunConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ExperimentConfig":
        """Create and validate a typed config from a dict."""
        run = RunConfig(**payload.get("run", {}))
        data = DataConfig(**payload.get("data", {}))
        model = ModelConfig(**payload.get("model", {}))
        training = TrainingConfig(**payload.get("training", {}))
        device = DeviceConfig(**payload.get("device", {}))
        benchmark = BenchmarkConfig(**payload.get("benchmark", {}))

        _validate_split_ratios(data)
        _validate_positive_values(model, training)
        _validate_benchmark(benchmark)
        _validate_dblp(data)

        return cls(run=run, data=data, model=model, training=training, device=device, benchmark=benchmark)

    def to_dict(self) -> Dict[str, Any]:
        """Convert typed config to a standard dictionary."""
        return asdict(self)


def _validate_split_ratios(data: DataConfig) -> None:
    ratios: List[float] = [data.train_ratio, data.val_ratio, data.test_ratio]
    if any(r <= 0 for r in ratios):
        raise ValueError("Split ratios must all be > 0.")
    total = sum(ratios)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}.")


def _validate_positive_values(model: ModelConfig, training: TrainingConfig) -> None:
    if model.in_channels is not None and model.in_channels <= 0:
        raise ValueError("model.in_channels must be positive when provided.")
    if model.hidden_channels <= 0:
        raise ValueError("model.hidden_channels must be positive.")
    if model.out_channels is not None and model.out_channels <= 0:
        raise ValueError("model.out_channels must be positive when provided.")
    if model.num_layers <= 0:
        raise ValueError("model.num_layers must be positive.")
    if not 0.0 <= model.dropout < 1.0:
        raise ValueError("model.dropout must be in [0, 1).")
    if training.epochs <= 0:
        raise ValueError("training.epochs must be positive.")
    if training.lr <= 0:
        raise ValueError("training.lr must be positive.")
    if training.weight_decay < 0:
        raise ValueError("training.weight_decay must be >= 0.")
    if training.early_stopping_patience <= 0:
        raise ValueError("training.early_stopping_patience must be positive.")



def _validate_benchmark(benchmark: BenchmarkConfig) -> None:
    if benchmark.inference_warmup_passes < 0:
        raise ValueError("benchmark.inference_warmup_passes must be >= 0.")
    if benchmark.inference_timed_passes <= 0:
        raise ValueError("benchmark.inference_timed_passes must be > 0.")


def _validate_dblp(data: DataConfig) -> None:
    name = data.name.strip().lower()
    strategy = data.dblp_strategy.strip().lower()
    if name == "dblp" and strategy not in SUPPORTED_DBLP_STRATEGIES:
        raise ValueError(
            f"Unsupported data.dblp_strategy '{data.dblp_strategy}' for DBLP. "
            f"Supported: {list(SUPPORTED_DBLP_STRATEGIES)}"
        )


def snapshot_path(output_dir: Union[str, Path]) -> Path:
    """Return output path for resolved config snapshots."""
    return Path(output_dir).expanduser() / "resolved_config.yaml"

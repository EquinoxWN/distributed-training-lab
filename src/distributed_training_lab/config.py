"""Validated configuration for the model and for a training run."""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from typing import Any, Literal

Strategy = Literal["manual", "torch"]
Algorithm = Literal["ring", "builtin"]
Device = Literal["cpu", "cuda"]
DataSource = Literal["synthetic", "tinystories"]

MAX_WORLD_SIZE = 8
MAX_STEPS = 100_000


@dataclass(frozen=True)
class ModelConfig:
    """Shape of the decoder-only transformer."""

    vocab_size: int = 256
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    max_seq_len: int = 128

    def __post_init__(self) -> None:
        """Reject shapes the model cannot build."""
        for name in ("vocab_size", "d_model", "n_layers", "n_heads", "max_seq_len"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.d_model > 4096 or self.n_layers > 64 or self.max_seq_len > 4096:
            raise ValueError("model is larger than this lab supports")


# "tiny" is the size tiny-transformer-scratch targets; "scaled" (4.84M parameters) is about 39 times larger.
PRESETS: dict[str, ModelConfig] = {
    "tiny": ModelConfig(d_model=64, n_layers=2, n_heads=4, max_seq_len=128),
    "scaled": ModelConfig(d_model=256, n_layers=6, n_heads=8, max_seq_len=128),
}


@dataclass(frozen=True)
class TrainConfig:
    """One data-parallel training run."""

    model: ModelConfig = ModelConfig()
    world_size: int = 1
    device: Device = "cpu"
    strategy: Strategy = "manual"
    algorithm: Algorithm = "ring"
    data: DataSource = "synthetic"
    steps: int = 20
    global_batch: int = 16
    seq_len: int = 64
    lr: float = 3e-3
    seed: int = 0
    bucket_mb: float = 4.0
    threads_per_rank: int = 2
    warmup_steps: int = 3
    eval_batches: int = 8

    def __post_init__(self) -> None:
        """Reject runs that cannot be split evenly or are out of bounds."""
        if not 1 <= self.world_size <= MAX_WORLD_SIZE:
            raise ValueError(f"world_size must be in 1..{MAX_WORLD_SIZE}")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        if self.strategy not in ("manual", "torch"):
            raise ValueError("strategy must be manual or torch")
        if self.strategy == "torch" and self.device == "cuda" and sys.platform == "win32":
            raise ValueError("PyTorch DDP over gloo with CUDA tensors crashes on Windows; use cpu")
        if self.algorithm not in ("ring", "builtin"):
            raise ValueError("algorithm must be ring or builtin")
        if self.data not in ("synthetic", "tinystories"):
            raise ValueError("data must be synthetic or tinystories")
        if not 1 <= self.steps <= MAX_STEPS:
            raise ValueError(f"steps must be in 1..{MAX_STEPS}")
        if self.global_batch < 1 or self.global_batch % self.world_size:
            raise ValueError("global_batch must be a positive multiple of world_size")
        if not 1 <= self.seq_len <= self.model.max_seq_len:
            raise ValueError("seq_len must be in 1..model.max_seq_len")
        if not 0.0 < self.lr < 1.0:
            raise ValueError("lr must be in (0, 1)")
        if not 0.0 < self.bucket_mb <= 1024.0:
            raise ValueError("bucket_mb must be in (0, 1024]")
        if not 1 <= self.threads_per_rank <= 64:
            raise ValueError("threads_per_rank must be in 1..64")
        if not 0 <= self.warmup_steps < self.steps:
            raise ValueError("warmup_steps must be smaller than steps")
        if not 1 <= self.eval_batches <= 1000:
            raise ValueError("eval_batches must be in 1..1000")

    @property
    def local_batch(self) -> int:
        """Sequences each rank processes per step."""
        return self.global_batch // self.world_size

    def to_dict(self) -> dict[str, Any]:
        """Plain dict for JSON results."""
        return asdict(self)

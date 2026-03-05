from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ModelConfig:
    vocab_size: int = 50257
    seq_len: int = 128
    n_layers: int = 6
    n_heads: int = 8
    d_model: int = 512
    d_ff: int = 2048
    dropout: float = 0.1
    mask_token_id: int = 50256

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_heads == 0
        return self.d_model // self.n_heads

    @property
    def ff_dim(self) -> int:
        return self.d_ff if self.d_ff > 0 else 4 * self.d_model


@dataclass
class ScheduleConfig:
    kind: Literal["cosine", "linear", "sqrt"] = "cosine"
    T: int = 1000
    eps: float = 1e-4  # Small offset to keep β(0) > 0 and β(T) < 1.


@dataclass
class DataConfig:
    dataset: Literal["wikitext103", "text8", "custom"] = "wikitext103"
    data_path: str = ""
    seq_len: int = 128
    num_workers: int = 4
    cache_dir: str = ".cache"


@dataclass
class TrainConfig:
    seed: int = 42
    batch_size: int = 256
    grad_accum_steps: int = 1
    max_steps: int = 100_000
    warmup_steps: int = 2_000
    lr: float = 3e-4
    min_lr: float = 1e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.98
    clip_grad_norm: float = 1.0
    log_every: int = 50
    eval_every: int = 1_000
    save_every: int = 5_000
    out_dir: str = "runs/default"
    resume: str = ""
    compile: bool = False  # JIT-compile the train step with jax.jit (always True in practice, flag kept for debugging).


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def validate(cfg: Config) -> None:
    assert cfg.model.seq_len == cfg.data.seq_len, (
        f"model.seq_len ({cfg.model.seq_len}) must equal data.seq_len ({cfg.data.seq_len})"
    )
    assert cfg.train.batch_size % 1 == 0, "batch_size must be a positive integer"
    assert cfg.model.d_model % cfg.model.n_heads == 0, (
        "d_model must be divisible by n_heads"
    )

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ModelConfig:
    init_seed: int = 123
    vocab_size: int = 50304
    seq_len: int = 1024
    n_layers: int = 2
    n_heads: int = 4
    d_model: int = 64
    d_ff: int = 256
    dropout: float = 0.1
    mask_token_id: int = 50257

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
    eps: float = 1e-4


@dataclass
class DataConfig:
    trainset_path: str = "/fast/jsingh/data/openwebtext/tokenized-9b-gpt2/train"
    validset_path: str = "/fast/jsingh/data/openwebtext/tokenized-9b-gpt2/val"
    shuffle_seed: int = 123
    seq_len: int = 1024
    num_workers: int = 8


@dataclass
class TrainConfig:
    seed: int = 123
    batch_size: int = 8
    grad_acc_steps: int = 2
    max_steps: int = 100

    optimizer: str = "adamw"
    beta1: float = 0.9
    beta2: float = 0.98
    weight_decay: float = 0.1
    eps: float = 1e-8

    lr: float = 3e-4
    min_lr: float = 1e-5
    lr_schedule: str = "linear"
    warmup_steps: int = 20
    cooldown_start_steps: int = 2000
    clip_grad_norm: float = 1.0

    log_every: int = 50
    eval_every: int = 20
    save_every: int = 20
    out_dir: str = "runs/default"
    resume: str = ""
    compile: bool = False


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

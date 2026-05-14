from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ModelConfig:
    init_seed: int = 123
    vocab_size: int = 50304
    seq_len: int = 1024
    n_layers: int = 8
    n_heads: int = 8
    d_model: int = 256
    d_ff: int = 1024
    dropout: float = 0.0
    mask_token_id: int = 50257
    T: int = 1000
    time_conditioning: bool = False

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
    batch_size: int = 32
    grad_acc_steps: int = 8
    max_steps: int = 6400

    optimizer: str = "adamw"
    beta1: float = 0.9
    beta2: float = 0.98
    weight_decay: float = 0.1
    eps: float = 1e-8

    lr: float = 2e-2
    min_lr: float = 1e-5
    lr_schedule: str = "warmup_cosine"
    warmup_steps: int = 20
    cooldown_start_steps: int = 2000
    clip_grad_norm: float = 1.0


@dataclass
class ExperimentConfig:
    run_name: str = "single_gpu_test"
    project_name: str = "nano-dlm"
    use_wandb: bool = False

    log_every: int = 50
    eval_every: int = 50
    saving: bool = True
    save_every: int = 50

    out_dir: str = "runs/default"
    resume: bool = False
    resume_folder: str = ""


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    exp: ExperimentConfig = field(default_factory=ExperimentConfig)


def validate(cfg: Config) -> None:
    assert cfg.model.seq_len == cfg.data.seq_len, (
        f"model.seq_len ({cfg.model.seq_len}) must equal data.seq_len ({cfg.data.seq_len})"
    )
    assert cfg.train.batch_size % 1 == 0, "batch_size must be a positive integer"
    assert cfg.model.d_model % cfg.model.n_heads == 0, (
        "d_model must be divisible by n_heads"
    )

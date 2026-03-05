"""
config.py — All dataclass configs for nano-dlm.
Use tyro to parse CLI arguments:
    python train.py --help
    python train.py --model.n_layers 12 --train.lr 3e-4
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Transformer denoiser architecture."""

    vocab_size: int = 50257
    """GPT-2 BPE vocabulary size (includes [MASK] token appended at end)."""

    seq_len: int = 128
    """Maximum sequence length."""

    n_layers: int = 6
    """Number of Transformer encoder layers."""

    n_heads: int = 8
    """Number of attention heads."""

    d_model: int = 512
    """Model / embedding dimension."""

    d_ff: int = 2048
    """Feed-forward hidden dimension (0 → 4 * d_model)."""

    dropout: float = 0.1
    """Dropout probability applied in attention and FFN."""

    mask_token_id: int = 50256
    """Token ID used as the [MASK] absorbing state."""

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_heads == 0
        return self.d_model // self.n_heads

    @property
    def ff_dim(self) -> int:
        return self.d_ff if self.d_ff > 0 else 4 * self.d_model


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------

@dataclass
class ScheduleConfig:
    """Noise schedule configuration."""

    kind: Literal["cosine", "linear", "sqrt"] = "cosine"
    """Which masking rate schedule to use."""

    T: int = 1000
    """Number of discrete diffusion steps."""

    eps: float = 1e-4
    """Small offset to keep β(0) > 0 and β(T) < 1."""


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    """Dataset and tokenization settings."""

    dataset: Literal["wikitext103", "text8", "custom"] = "wikitext103"
    """Which dataset to use. Set 'custom' and supply data_path for your own files."""

    data_path: str = ""
    """Path to a .txt file (used when dataset='custom')."""

    seq_len: int = 128
    """Sequence length (must match ModelConfig.seq_len)."""

    num_workers: int = 4
    """Number of CPU workers for data loading (passed to DataLoader)."""

    cache_dir: str = ".cache"
    """Directory to cache tokenised dataset shards."""


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """Training loop hyper-parameters."""

    seed: int = 42
    """Global PRNG seed."""

    batch_size: int = 256
    """Global batch size (will be split evenly across devices)."""

    grad_accum_steps: int = 1
    """Gradient accumulation steps before an optimizer update."""

    max_steps: int = 100_000
    """Total number of gradient update steps."""

    warmup_steps: int = 2_000
    """Linear LR warm-up steps."""

    lr: float = 3e-4
    """Peak learning rate."""

    min_lr: float = 1e-5
    """Minimum LR after cosine decay."""

    weight_decay: float = 0.1
    """AdamW weight decay."""

    beta1: float = 0.9
    """AdamW β₁."""

    beta2: float = 0.98
    """AdamW β₂."""

    clip_grad_norm: float = 1.0
    """Global gradient norm clipping (0 → disabled)."""

    log_every: int = 50
    """Log training metrics every N steps."""

    eval_every: int = 1_000
    """Run validation every N steps."""

    save_every: int = 5_000
    """Save checkpoint every N steps."""

    out_dir: str = "runs/default"
    """Directory to write checkpoints and logs."""

    resume: str = ""
    """Path to a checkpoint directory to resume from (empty → start fresh)."""

    compile: bool = False
    """JIT-compile the train step with jax.jit (always True in practice, flag kept for debugging)."""


# ---------------------------------------------------------------------------
# Top-level config (composed)
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """Master config for nano-dlm — all sub-configs composed here."""

    model: ModelConfig = field(default_factory=ModelConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


# ---------------------------------------------------------------------------
# Quick sanity-check helper
# ---------------------------------------------------------------------------

def validate(cfg: Config) -> None:
    assert cfg.model.seq_len == cfg.data.seq_len, (
        f"model.seq_len ({cfg.model.seq_len}) must equal data.seq_len ({cfg.data.seq_len})"
    )
    assert cfg.train.batch_size % 1 == 0, "batch_size must be a positive integer"
    assert cfg.model.d_model % cfg.model.n_heads == 0, (
        "d_model must be divisible by n_heads"
    )

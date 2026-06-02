from __future__ import annotations

import math
import typing as tp
from functools import partial

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from src.config import ModelConfig

# ── data‑parallel sharding: all parameters replicated ──────────────────────
_KERNEL_INIT = nnx.with_partitioning(nnx.initializers.normal(stddev=0.02), (None, None))
_BIAS_INIT = nnx.with_partitioning(nnx.initializers.zeros_init(), (None,))
_SCALE_INIT = nnx.with_partitioning(nnx.initializers.ones_init(), (None,))
_EMBED_INIT = nnx.with_partitioning(nnx.initializers.normal(stddev=0.02), (None, None))


def sinusoidal_embedding(t: jax.Array, dim: int) -> jax.Array:
  half = dim // 2
  freqs = jnp.exp(-math.log(10000) * jnp.arange(half) / half)
  args = t[:, None] * freqs[None, :]
  emb = jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)
  if dim % 2 == 1:
    emb = jnp.concatenate([emb, jnp.zeros_like(emb[:, :1])], axis=-1)
  return emb


def rope_freqs(seq_len: int, head_dim: int, base: int = 10_000) -> jax.Array:
  half = head_dim // 2
  theta = 1.0 / (base ** (jnp.arange(half, dtype=jnp.float32) / half))
  pos = jnp.arange(seq_len, dtype=jnp.float32)
  return jnp.outer(pos, theta)  # (L, D/2)


def apply_rope(x: jax.Array, freqs: jax.Array) -> jax.Array:
  half = x.shape[-1] // 2
  x1, x2 = x[..., :half], x[..., half:]
  cos_ = jnp.cos(freqs)[None, None]  # (1, 1, L, D/2)
  sin_ = jnp.sin(freqs)[None, None]
  return jnp.concatenate([x1 * cos_ - x2 * sin_, x1 * sin_ + x2 * cos_], axis=-1)


class AdaLN(nnx.Module):
  def __init__(self, d_model: int, time_emb_dim: int, rngs: nnx.Rngs):
    self.norm = nnx.LayerNorm(d_model, rngs=rngs, scale_init=_SCALE_INIT, bias_init=_BIAS_INIT)
    self.proj = nnx.Linear(
      time_emb_dim,
      2 * d_model,
      use_bias=True,
      rngs=rngs,
      kernel_init=_KERNEL_INIT,
      bias_init=_BIAS_INIT,
    )

  def __call__(self, x: jax.Array, time_emb: jax.Array) -> jax.Array:
    # x        : (B, L, D)
    # time_emb : (B, time_emb_dim)
    x_n = self.norm(x)  # (B, L, D)
    gb = self.proj(time_emb)  # (B, 2D)
    D = x.shape[-1]
    gamma, beta = gb[:, :D], gb[:, D:]  # (B, D) each
    # broadcast over sequence length
    return x_n * (1.0 + gamma[:, None, :]) + beta[:, None, :]


class MultiHeadAttention(nnx.Module):
  def __init__(self, d_model: int, n_heads: int, rngs: nnx.Rngs):
    assert d_model % n_heads == 0
    self.n_heads = n_heads
    self.head_dim = d_model // n_heads
    self.q_proj = nnx.Linear(d_model, d_model, use_bias=False, rngs=rngs, kernel_init=_KERNEL_INIT)
    self.k_proj = nnx.Linear(d_model, d_model, use_bias=False, rngs=rngs, kernel_init=_KERNEL_INIT)
    self.v_proj = nnx.Linear(d_model, d_model, use_bias=False, rngs=rngs, kernel_init=_KERNEL_INIT)
    self.out_proj = nnx.Linear(
      d_model,
      d_model,
      use_bias=True,
      rngs=rngs,
      kernel_init=_KERNEL_INIT,
      bias_init=_BIAS_INIT,
    )

  def __call__(
    self,
    x: jax.Array,  # (B, L, D)
    freqs: jax.Array,  # (L, head_dim//2)
  ) -> jax.Array:
    B, L, D = x.shape
    H, Dh = self.n_heads, self.head_dim

    # Project and reshape to (B, H, L, Dh)
    def proj_reshape(proj):
      return proj(x).reshape(B, L, H, Dh).transpose(0, 2, 1, 3)

    q = apply_rope(proj_reshape(self.q_proj), freqs)  # (B, H, L, Dh)
    k = apply_rope(proj_reshape(self.k_proj), freqs)
    v = proj_reshape(self.v_proj)

    # Scaled dot-product attention (bidirectional — no causal mask)
    scale = math.sqrt(Dh)
    weights = jnp.einsum('bhqd,bhkd->bhqk', q, k) / scale  # (B, H, L, L)
    weights = jax.nn.softmax(weights, axis=-1)

    out = jnp.einsum('bhqk,bhkd->bhqd', weights, v)  # (B, H, L, Dh)
    out = out.transpose(0, 2, 1, 3).reshape(B, L, D)  # (B, L, D)
    return self.out_proj(out)


class FFN(nnx.Module):
  def __init__(self, d_model: int, d_ff: int, rngs: nnx.Rngs):
    self.fc1 = nnx.Linear(
      d_model,
      d_ff,
      use_bias=True,
      rngs=rngs,
      kernel_init=_KERNEL_INIT,
      bias_init=_BIAS_INIT,
    )
    self.fc2 = nnx.Linear(
      d_ff,
      d_model,
      use_bias=True,
      rngs=rngs,
      kernel_init=_KERNEL_INIT,
      bias_init=_BIAS_INIT,
    )

  def __call__(
    self,
    x: jax.Array,
  ) -> jax.Array:
    x = jax.nn.gelu(self.fc1(x), approximate=True)
    return self.fc2(x)


class TransformerBlock(nnx.Module):
  def __init__(self, cfg: ModelConfig, rngs: nnx.Rngs, time_emb_dim: tp.Union[int, None] = None):
    self.time_conditioning = cfg.time_conditioning
    if self.time_conditioning and time_emb_dim is not None:
      self.adnorm1 = AdaLN(cfg.d_model, time_emb_dim, rngs)
      self.adnorm2 = AdaLN(cfg.d_model, time_emb_dim, rngs)
    else:
      self.norm1 = nnx.RMSNorm(cfg.d_model, rngs=rngs, scale_init=_SCALE_INIT)
      self.norm2 = nnx.RMSNorm(cfg.d_model, rngs=rngs, scale_init=_SCALE_INIT)

    self.attn = MultiHeadAttention(cfg.d_model, cfg.n_heads, rngs)
    self.ffn = FFN(cfg.d_model, cfg.ff_dim, rngs)

  def __call__(
    self,
    x: jax.Array,
    freqs: jax.Array,
    time_emb: tp.Union[jax.Array, None] = None,
  ) -> jax.Array:
    if self.time_conditioning and time_emb is not None:
      x = x + self.attn(self.adnorm1(x, time_emb), freqs)
      x = x + self.ffn(self.adnorm2(x, time_emb))
    else:
      x = x + self.attn(self.norm1(x), freqs)
      x = x + self.ffn(self.norm2(x))
    return x


class DiffusionTransformer(nnx.Module):
  def __init__(self, cfg: ModelConfig, rngs: nnx.Rngs):
    self.cfg = cfg
    self.token_emb = nnx.Embed(cfg.vocab_size, cfg.d_model, rngs=rngs, embedding_init=_EMBED_INIT)
    self.time_emb_dim = None

    if self.cfg.time_conditioning:
      self.time_emb_dim = cfg.d_model * 4
      # Time MLP: sinusoidal(t) → Linear → SiLU → Linear
      self.time_fc1 = nnx.Linear(
        cfg.d_model,
        self.time_emb_dim,
        rngs=rngs,
        kernel_init=_KERNEL_INIT,
        bias_init=_BIAS_INIT,
      )
      self.time_fc2 = nnx.Linear(
        self.time_emb_dim,
        self.time_emb_dim,
        rngs=rngs,
        kernel_init=_KERNEL_INIT,
        bias_init=_BIAS_INIT,
      )

    self.blocks = nnx.List([TransformerBlock(cfg, rngs, time_emb_dim=self.time_emb_dim) for _ in range(cfg.n_layers)])
    self.final_norm = nnx.RMSNorm(cfg.d_model, rngs=rngs, scale_init=_SCALE_INIT)
    self.lm_head = nnx.Linear(
      cfg.d_model,
      cfg.vocab_size,
      use_bias=False,
      rngs=rngs,
      kernel_init=_KERNEL_INIT,
    )

  def count_params(self, nonembed: bool = True) -> int:
    state = nnx.state(self, nnx.Param)
    total = sum(x.size for x in jax.tree_util.tree_leaves(state))
    if nonembed:
      embed_state = nnx.state(self.token_emb, nnx.Param)
      embed_params = sum(x.size for x in jax.tree_util.tree_leaves(embed_state))
      total -= embed_params
    return total

  def __call__(self, input_ids: jax.Array, timesteps: jax.Array, training: bool = False) -> jax.Array:
    L = input_ids.shape[1]
    x = self.token_emb(input_ids)
    x = jax.device_put(x, P('data', None))

    if self.cfg.time_conditioning and self.time_emb_dim is not None:
      t_norm = timesteps.astype(jnp.float32) / self.cfg.T
      t_sin = sinusoidal_embedding(t_norm, self.cfg.d_model)
      t_emb = self.time_fc2(jax.nn.silu(self.time_fc1(t_sin)))
    else:
      t_emb = None

    freqs = rope_freqs(L, self.cfg.head_dim)

    for block in self.blocks:
      x = block(x, freqs, time_emb=t_emb)

    x = self.final_norm(x)
    logits = self.lm_head(x)
    return logits


def init_model(cfg):
  @nnx.jit
  def _init(rngs):
    return DiffusionTransformer(cfg, rngs=rngs)

  return _init

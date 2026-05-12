from __future__ import annotations

import math
from typing import Optional

import flax.nnx as nnx
import jax
import jax.numpy as jnp

# from flax.nnx.graph import state
from src.config import ModelConfig


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
        self.norm = nnx.LayerNorm(d_model, rngs=rngs)
        # projects time_emb → (γ, β), each of size d_model
        self.proj = nnx.Linear(time_emb_dim, 2 * d_model, use_bias=True, rngs=rngs)

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
    def __init__(self, d_model: int, n_heads: int, dropout: float, rngs: nnx.Rngs):
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = nnx.Dropout(dropout, rngs=rngs)
        self.q_proj = nnx.Linear(d_model, d_model, use_bias=False, rngs=rngs)
        self.k_proj = nnx.Linear(d_model, d_model, use_bias=False, rngs=rngs)
        self.v_proj = nnx.Linear(d_model, d_model, use_bias=False, rngs=rngs)
        self.out_proj = nnx.Linear(d_model, d_model, use_bias=True, rngs=rngs)

    def __call__(
        self,
        x: jax.Array,  # (B, L, D)
        freqs: jax.Array,  # (L, head_dim//2)
        training: bool = False,
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
        weights = jnp.einsum("bhqd,bhkd->bhqk", q, k) / scale  # (B, H, L, L)
        weights = jax.nn.softmax(weights, axis=-1)

        if training:
            weights = self.dropout(weights)

        out = jnp.einsum("bhqk,bhkd->bhqd", weights, v)  # (B, H, L, Dh)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, D)  # (B, L, D)
        return self.out_proj(out)


class FFN(nnx.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(d_model, d_ff, use_bias=True, rngs=rngs)
        self.fc2 = nnx.Linear(d_ff, d_model, use_bias=True, rngs=rngs)
        self.dropout = nnx.Dropout(dropout, rngs=rngs)

    def __call__(
        self,
        x: jax.Array,
        training: bool = False,
    ) -> jax.Array:
        x = jax.nn.gelu(self.fc1(x), approximate=True)
        if training:
            x = self.dropout(x)
        return self.fc2(x)


class TransformerBlock(nnx.Module):
    def __init__(self, cfg: ModelConfig, rngs: nnx.Rngs, time_emb_dim: int = None):
        self.time_conditioning = cfg.time_conditioning
        if self.time_conditioning and time_emb_dim is not None:
            self.norm1 = AdaLN(cfg.d_model, time_emb_dim, rngs)
            self.norm2 = AdaLN(cfg.d_model, time_emb_dim, rngs)
        else:
            self.norm1 = nnx.RMSNorm(cfg.d_model, rngs=rngs)
            self.norm2 = nnx.RMSNorm(cfg.d_model, rngs=rngs)

        self.attn = MultiHeadAttention(cfg.d_model, cfg.n_heads, cfg.dropout, rngs)
        self.ffn = FFN(cfg.d_model, cfg.ff_dim, cfg.dropout, rngs)

    def __call__(
        self,
        x: jax.Array,
        freqs: jax.Array,
        training: bool = False,
        time_emb: jax.Array = None,
    ) -> jax.Array:
        if self.time_conditioning and time_emb is not None:
            x = x + self.attn(self.norm1(x, time_emb), freqs, training=training)
            x = x + self.ffn(self.norm2(x, time_emb), training=training)
        else:
            x = x + self.attn(self.norm1(x), freqs, training=training)
            x = x + self.ffn(self.norm2(x), training=training)
        return x


class DiffusionTransformer(nnx.Module):
    def __init__(self, cfg: ModelConfig, rngs: nnx.Rngs):
        self.cfg = cfg
        self.token_emb = nnx.Embed(cfg.vocab_size, cfg.d_model, rngs=rngs)
        self.time_emb_dim = None

        if self.cfg.time_conditioning:
            self.time_emb_dim = cfg.d_model * 4
            # Time MLP: sinusoidal(t) → Linear → SiLU → Linear
            self.time_fc1 = nnx.Linear(cfg.d_model, self.time_emb_dim, rngs=rngs)
            self.time_fc2 = nnx.Linear(self.time_emb_dim, self.time_emb_dim, rngs=rngs)

        self.blocks = nnx.List(
            [
                TransformerBlock(cfg, rngs, time_emb_dim=self.time_emb_dim)
                for _ in range(cfg.n_layers)
            ]
        )
        self.final_norm = nnx.RMSNorm(cfg.d_model, rngs=rngs)
        self.lm_head = nnx.Linear(
            cfg.d_model, cfg.vocab_size, use_bias=False, rngs=rngs
        )

    def count_params(self, nonembed: bool = True) -> int:
        state = nnx.state(self, nnx.Param)
        total = sum(x.size for x in jax.tree_util.tree_leaves(state))
        if nonembed:
            embed_state = nnx.state(self.token_emb, nnx.Param)
            embed_params = sum(x.size for x in jax.tree_util.tree_leaves(embed_state))
            total -= embed_params
        return total

    def __call__(
        self,
        xt: jax.Array,
        t: jax.Array = None,
        training: bool = False,
        rng: Optional[jax.Array] = None,
    ) -> jax.Array:
        x = self.token_emb(xt)

        if (
            self.cfg.time_conditioning
            and self.time_emb_dim is not None
            and t is not None
        ):
            t_norm = t.astype(jnp.float32) / self.cfg.T
            t_sin = sinusoidal_embedding(t_norm, self.cfg.d_model)
            t_emb = self.time_fc2(jax.nn.silu(self.time_fc1(t_sin)))
        else:
            t_emb = None

        freqs = rope_freqs(L, self.cfg.head_dim)

        for block in self.blocks:
            x = block(x, freqs, training=training, time_emb=t_emb)

        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits

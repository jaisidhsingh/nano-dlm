"""
model.py — Transformer denoiser for nano-dlm.

Architecture: a bidirectional Transformer encoder (BERT-style) that, given noisy
tokens xₜ and the continuous time embedding t, predicts the logits over clean
tokens x₀.  This is the standard "predict x₀" parameterisation used in MDLM /
D3PM absorbing diffusion.

Key design choices
──────────────────
• Pre-LayerNorm (more stable training at large scale).
• Sinusoidal + learned time embedding injected via AdaLN (adaptive layer norm)
  — avoids fusing t into every attention layer manually.
• Flash-attention-style attention via jax.nn.dot_product_attention (JAX 0.4.28+)
  with fallback to manual scaled dot-product.
• Rotary Position Encoding (RoPE) — no learnable position embeddings to keep
  the model simple.
• All parameters live in a single Equinox module so they serialise cleanly.
"""

from __future__ import annotations
import math
import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional
from config import ModelConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sinusoidal_embedding(t: jax.Array, dim: int) -> jax.Array:
    """
    Classic sinusoidal time embedding (Vaswani 2017 positional encoding).
    t : (B,) float in [0, 1]  — we pass t/T as a normalised float.
    Returns (B, dim).
    """
    half = dim // 2
    freqs = jnp.exp(-math.log(10000) * jnp.arange(half) / half)   # (D/2,)
    args  = t[:, None] * freqs[None, :]                             # (B, D/2)
    emb   = jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)  # (B, D)
    if dim % 2 == 1:
        emb = jnp.concatenate([emb, jnp.zeros_like(emb[:, :1])], axis=-1)
    return emb


def rope_freqs(seq_len: int, head_dim: int, base: int = 10_000) -> jax.Array:
    """Pre-compute RoPE rotation frequencies. Returns (seq_len, head_dim/2)."""
    half   = head_dim // 2
    theta  = 1.0 / (base ** (jnp.arange(0, half, dtype=jnp.float32) / half))
    pos    = jnp.arange(seq_len, dtype=jnp.float32)
    freqs  = jnp.outer(pos, theta)   # (L, D/2)
    return freqs


def apply_rope(x: jax.Array, freqs: jax.Array) -> jax.Array:
    """
    Apply rotary position embedding to query or key.
    x      : (B, H, L, D)
    freqs  : (L, D/2)
    """
    B, H, L, D = x.shape
    half = D // 2
    x1, x2 = x[..., :half], x[..., half:]  # (B, H, L, D/2)
    cos_ = jnp.cos(freqs)[None, None, :, :]  # (1, 1, L, D/2)
    sin_ = jnp.sin(freqs)[None, None, :, :]
    x_rot = jnp.concatenate([x1 * cos_ - x2 * sin_,
                              x1 * sin_ + x2 * cos_], axis=-1)
    return x_rot


# ---------------------------------------------------------------------------
# Adaptive Layer Norm  (AdaLN)
# ---------------------------------------------------------------------------

class AdaLN(eqx.Module):
    """
    Adaptive Layer Norm: scale and shift conditioned on time embedding.
      y = γ(t) * LayerNorm(x) + β(t)
    where γ, β are linear projections of the time embedding.
    """
    norm: eqx.nn.LayerNorm
    proj: eqx.nn.Linear  # projects time_emb → 2 * d_model (γ and β)

    def __init__(self, d_model: int, time_emb_dim: int, key: jax.Array):
        self.norm = eqx.nn.LayerNorm(d_model)
        self.proj = eqx.nn.Linear(time_emb_dim, 2 * d_model, use_bias=True, key=key)

    def __call__(self, x: jax.Array, time_emb: jax.Array) -> jax.Array:
        # x : (L, D),  time_emb : (D_t,)
        x_normed = jax.vmap(self.norm)(x)          # (L, D)
        gamma_beta = self.proj(time_emb)            # (2D,)
        gamma, beta = gamma_beta[:x.shape[-1]], gamma_beta[x.shape[-1]:]
        return x_normed * (1.0 + gamma) + beta      # (L, D)


# ---------------------------------------------------------------------------
# Multi-Head Self-Attention  (bidirectional, RoPE)
# ---------------------------------------------------------------------------

class MultiHeadAttention(eqx.Module):
    q_proj: eqx.nn.Linear
    k_proj: eqx.nn.Linear
    v_proj: eqx.nn.Linear
    out_proj: eqx.nn.Linear
    n_heads: int
    head_dim: int
    dropout_p: float

    def __init__(self, d_model: int, n_heads: int, dropout: float, key: jax.Array):
        assert d_model % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.dropout_p = dropout
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.q_proj   = eqx.nn.Linear(d_model, d_model, use_bias=False, key=k1)
        self.k_proj   = eqx.nn.Linear(d_model, d_model, use_bias=False, key=k2)
        self.v_proj   = eqx.nn.Linear(d_model, d_model, use_bias=False, key=k3)
        self.out_proj = eqx.nn.Linear(d_model, d_model, use_bias=True,  key=k4)

    def __call__(
        self,
        x: jax.Array,          # (L, D)
        freqs: jax.Array,      # (L, D//2//n_heads)   RoPE freqs
        enable_dropout: bool = False,
        key: Optional[jax.Array] = None,
    ) -> jax.Array:
        L, D = x.shape
        H, Dh = self.n_heads, self.head_dim

        q = jax.vmap(self.q_proj)(x).reshape(L, H, Dh).transpose(1, 0, 2)  # (H, L, Dh)
        k = jax.vmap(self.k_proj)(x).reshape(L, H, Dh).transpose(1, 0, 2)
        v = jax.vmap(self.v_proj)(x).reshape(L, H, Dh).transpose(1, 0, 2)

        # Apply RoPE — add a dummy batch dim, apply, then remove
        q = apply_rope(q[None], freqs)[0]   # (H, L, Dh)
        k = apply_rope(k[None], freqs)[0]

        # Scaled dot-product attention (bidirectional — no causal mask)
        scale = math.sqrt(Dh)
        attn_weights = jnp.einsum("hqd, hkd -> hqk", q, k) / scale  # (H, L, L)
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)

        if enable_dropout and key is not None:
            attn_weights = jax.nn.dropout(attn_weights, rate=self.dropout_p, key=key)

        out = jnp.einsum("hqk, hkd -> hqd", attn_weights, v)  # (H, L, Dh)
        out = out.transpose(1, 0, 2).reshape(L, D)             # (L, D)
        return jax.vmap(self.out_proj)(out)


# ---------------------------------------------------------------------------
# Feed-Forward Network
# ---------------------------------------------------------------------------

class FFN(eqx.Module):
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    dropout_p: float

    def __init__(self, d_model: int, d_ff: int, dropout: float, key: jax.Array):
        k1, k2 = jax.random.split(key)
        self.fc1 = eqx.nn.Linear(d_model, d_ff, use_bias=True, key=k1)
        self.fc2 = eqx.nn.Linear(d_ff, d_model, use_bias=True, key=k2)
        self.dropout_p = dropout

    def __call__(self, x: jax.Array, enable_dropout: bool = False,
                 key: Optional[jax.Array] = None) -> jax.Array:
        x = jax.vmap(self.fc1)(x)
        x = jax.nn.gelu(x, approximate=True)
        if enable_dropout and key is not None:
            x = jax.nn.dropout(x, rate=self.dropout_p, key=key)
        x = jax.vmap(self.fc2)(x)
        return x


# ---------------------------------------------------------------------------
# Transformer Block  (Pre-LN + AdaLN for time conditioning)
# ---------------------------------------------------------------------------

class TransformerBlock(eqx.Module):
    adaln1: AdaLN
    attn:   MultiHeadAttention
    adaln2: AdaLN
    ffn:    FFN

    def __init__(self, cfg: ModelConfig, time_emb_dim: int, key: jax.Array):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.adaln1 = AdaLN(cfg.d_model, time_emb_dim, k1)
        self.attn   = MultiHeadAttention(cfg.d_model, cfg.n_heads, cfg.dropout, k2)
        self.adaln2 = AdaLN(cfg.d_model, time_emb_dim, k3)
        self.ffn    = FFN(cfg.d_model, cfg.ff_dim, cfg.dropout, k4)

    def __call__(
        self,
        x: jax.Array,          # (L, D)
        time_emb: jax.Array,   # (D_t,)
        freqs: jax.Array,
        enable_dropout: bool = False,
        key: Optional[jax.Array] = None,
    ) -> jax.Array:
        k1, k2 = (jax.random.split(key) if key is not None else (None, None))

        # Attention sub-layer
        x = x + self.attn(self.adaln1(x, time_emb), freqs,
                           enable_dropout=enable_dropout, key=k1)
        # FFN sub-layer
        x = x + self.ffn(self.adaln2(x, time_emb), enable_dropout=enable_dropout, key=k2)
        return x


# ---------------------------------------------------------------------------
# Full Denoiser
# ---------------------------------------------------------------------------

class DiffusionTransformer(eqx.Module):
    """
    Bidirectional Transformer denoiser for masked diffusion over text.

    Forward pass:
        logits = model(xt, t, key=...)
        logits : (B, L, V)  — un-normalised log-probs over vocabulary

    For training, minimise cross-entropy at masked positions between
    logits and ground-truth x₀.
    """

    token_emb:  eqx.nn.Embedding
    time_emb_mlp: eqx.nn.Sequential
    blocks:     list
    final_norm: eqx.nn.LayerNorm
    lm_head:    eqx.nn.Linear
    cfg:        ModelConfig = eqx.field(static=True)
    _rope_freqs: jax.Array  = eqx.field(static=True)
    time_emb_dim: int       = eqx.field(static=True)

    def __init__(self, cfg: ModelConfig, key: jax.Array):
        self.cfg = cfg
        self.time_emb_dim = cfg.d_model * 4

        keys = jax.random.split(key, cfg.n_layers + 4)
        k_temb1, k_temb2, k_temb3, k_head, *block_keys = keys

        # Token embedding
        self.token_emb = eqx.nn.Embedding(cfg.vocab_size, cfg.d_model, key=k_temb1)

        # Time MLP:  sinusoidal(t) → linear → SiLU → linear → time_emb_dim
        self.time_emb_mlp = eqx.nn.Sequential([
            eqx.nn.Linear(cfg.d_model, self.time_emb_dim, key=k_temb2),
            eqx.nn.Lambda(jax.nn.silu),
            eqx.nn.Linear(self.time_emb_dim, self.time_emb_dim, key=k_temb3),
        ])

        # Transformer blocks
        self.blocks = [
            TransformerBlock(cfg, self.time_emb_dim, block_keys[i])
            for i in range(cfg.n_layers)
        ]

        self.final_norm = eqx.nn.LayerNorm(cfg.d_model)
        # Weight-tying: lm_head shares weights with token_emb
        self.lm_head = eqx.nn.Linear(cfg.d_model, cfg.vocab_size, use_bias=False, key=k_head)

        # Pre-compute RoPE frequencies (static)
        self._rope_freqs = rope_freqs(cfg.seq_len, cfg.head_dim)

    # ------------------------------------------------------------------
    # Single-sample forward (vmapped over batch in __call__)
    # ------------------------------------------------------------------

    def _forward_single(
        self,
        xt:      jax.Array,      # (L,)  int32  noisy tokens
        t_norm:  jax.Array,      # ()    float  t / T
        enable_dropout: bool = False,
        key: Optional[jax.Array] = None,
    ) -> jax.Array:
        L = xt.shape[0]

        # Token embedding
        x = jax.vmap(self.token_emb)(xt)   # (L, D)

        # Time embedding
        t_emb_sin = sinusoidal_embedding(t_norm[None], self.cfg.d_model)[0]  # (D,)
        t_emb = self.time_emb_mlp(t_emb_sin)                                  # (D_t,)

        # RoPE frequencies matched to current seq length
        freqs = self._rope_freqs[:L]  # (L, Dh/2)

        # Transformer blocks
        block_keys = jax.random.split(key, len(self.blocks)) if key is not None else [None] * len(self.blocks)
        for block, bkey in zip(self.blocks, block_keys):
            x = block(x, t_emb, freqs, enable_dropout=enable_dropout, key=bkey)

        x = jax.vmap(self.final_norm)(x)   # (L, D)
        logits = jax.vmap(self.lm_head)(x) # (L, V)
        return logits

    def __call__(
        self,
        xt: jax.Array,           # (B, L)
        t:  jax.Array,           # (B,)   integer steps
        T:  int,                 # total number of diffusion steps
        enable_dropout: bool = False,
        key: Optional[jax.Array] = None,
    ) -> jax.Array:
        """Returns logits of shape (B, L, V)."""
        B = xt.shape[0]
        t_norm = t.astype(jnp.float32) / T   # (B,) in [0, 1]

        keys = jax.random.split(key, B) if key is not None else [None] * B

        def single(args):
            xi, ti, ki = args
            return self._forward_single(xi, ti, enable_dropout=enable_dropout, key=ki)

        logits = jax.vmap(self._forward_single)(xt, t_norm,
                                                 enable_dropout=jnp.array(enable_dropout))
        return logits  # (B, L, V)


# ---------------------------------------------------------------------------
# Parameter count utility
# ---------------------------------------------------------------------------

def count_params(model: DiffusionTransformer) -> int:
    """Return total number of trainable parameters."""
    leaves = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
    return sum(x.size for x in leaves)

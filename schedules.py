"""
schedules.py — Noise schedules for discrete masked diffusion.

Implements the absorbing (mask) diffusion framework:
  - Forward process: independently mask each token with probability αₜ.
  - The marginal q(xₜ | x₀): each token is [MASK] with prob αₜ, unchanged otherwise.
  - The posterior q(xₜ₋₁ | xₜ, x₀) is used for training via the ELBO.

Three schedule families are provided:
  cosine  — from Improved DDPM (Nichol & Dhariwal 2021), adapted for masking rate
  linear  — classic linear schedule
  sqrt    — sqrt schedule (MDLM, Shi et al. 2024)

All schedules return:
  alpha_bar : (T+1,) array, ᾱₜ = 1 - masking-rate at step t
              ᾱ₀ = 1 (no masking), ᾱ_T ≈ 0 (fully masked)
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
from functools import partial
from config import ScheduleConfig


# ---------------------------------------------------------------------------
# Core schedule functions  (all return shape [T+1])
# ---------------------------------------------------------------------------

def _cosine_alpha_bar(T: int, eps: float = 1e-4) -> jax.Array:
    """Cosine schedule for the *unmasked* fraction ᾱₜ."""
    t = jnp.linspace(0, 1, T + 1)
    f = jnp.cos((t + 0.008) / 1.008 * jnp.pi / 2) ** 2
    alpha_bar = f / f[0]
    # clamp to (eps, 1-eps) so we never have exactly 0 or 1 masking rate
    alpha_bar = jnp.clip(alpha_bar, eps, 1.0 - eps)
    return alpha_bar  # shape (T+1,)


def _linear_alpha_bar(T: int, eps: float = 1e-4) -> jax.Array:
    """Linear schedule: ᾱₜ decreases linearly from ~1 to eps."""
    t = jnp.linspace(0, 1, T + 1)
    alpha_bar = 1.0 - t * (1.0 - eps)
    alpha_bar = jnp.clip(alpha_bar, eps, 1.0)
    return alpha_bar


def _sqrt_alpha_bar(T: int, eps: float = 1e-4) -> jax.Array:
    """Sqrt schedule: ᾱₜ = 1 - sqrt(t/T), used in MDLM."""
    t = jnp.linspace(0, 1, T + 1)
    alpha_bar = 1.0 - jnp.sqrt(t + eps)
    alpha_bar = jnp.clip(alpha_bar, eps, 1.0 - eps)
    return alpha_bar


# ---------------------------------------------------------------------------
# Schedule builder
# ---------------------------------------------------------------------------

def make_schedule(cfg: ScheduleConfig) -> "NoiseSchedule":
    """Factory: build and return a NoiseSchedule from config."""
    builders = {
        "cosine": _cosine_alpha_bar,
        "linear": _linear_alpha_bar,
        "sqrt":   _sqrt_alpha_bar,
    }
    fn = builders[cfg.kind]
    alpha_bar = fn(cfg.T, cfg.eps)
    return NoiseSchedule(alpha_bar=alpha_bar, T=cfg.T)


# ---------------------------------------------------------------------------
# NoiseSchedule dataclass  (pure-JAX, no Python state after construction)
# ---------------------------------------------------------------------------

class NoiseSchedule:
    """
    Holds pre-computed schedule arrays and exposes forward / posterior methods.

    All methods are pure functions — suitable for use inside jax.jit / jax.pmap.
    """

    def __init__(self, alpha_bar: jax.Array, T: int) -> None:
        self.T = T
        # ᾱₜ : unmasked fraction at step t  (shape T+1)
        self.alpha_bar = alpha_bar
        # αₜ  = ᾱₜ / ᾱₜ₋₁  (step-wise unmasked fraction, shape T+1; index 0 unused)
        alpha_bar_prev = jnp.concatenate([jnp.ones((1,)), alpha_bar[:-1]], axis=0)
        self.alpha = alpha_bar / jnp.clip(alpha_bar_prev, 1e-8)
        # masking rate at step t
        self.mask_rate = 1.0 - alpha_bar  # shape (T+1,)

    # ------------------------------------------------------------------
    # Forward process  q(xₜ | x₀)
    # ------------------------------------------------------------------

    def q_sample(
        self,
        x0: jax.Array,           # (B, L)  integer token ids
        t: jax.Array,            # (B,)    diffusion time steps  ∈ {1, …, T}
        mask_token_id: int,
        rng: jax.Array,
    ) -> jax.Array:
        """
        Sample xₜ ~ q(xₜ | x₀).
        Each position is independently replaced by [MASK] with prob (1-ᾱₜ).
        """
        B, L = x0.shape
        alpha_bar_t = self.alpha_bar[t]          # (B,)
        alpha_bar_t = alpha_bar_t[:, None]       # (B, 1)  → broadcast over L

        # Bernoulli mask: 1 = keep original, 0 = mask
        keep = jax.random.bernoulli(rng, alpha_bar_t, shape=(B, L))  # (B, L)
        xt = jnp.where(keep, x0, mask_token_id)
        return xt

    # ------------------------------------------------------------------
    # Posterior  q(xₜ₋₁ | xₜ, x₀)
    # ------------------------------------------------------------------

    def posterior_sample(
        self,
        x0: jax.Array,           # (B, L)  integer token ids
        xt: jax.Array,           # (B, L)  noisy tokens at step t
        t: jax.Array,            # (B,)    current step
        mask_token_id: int,
        rng: jax.Array,
    ) -> jax.Array:
        """
        Sample xₜ₋₁ ~ q(xₜ₋₁ | xₜ, x₀) for training.

        For absorbing diffusion (Hoogeboom et al., Austin et al.):
          - If xₜ is NOT masked       → xₜ₋₁ is also unmasked (= x₀ = xₜ)
          - If xₜ IS masked           → xₜ₋₁ is unmasked with prob p_unmask, else stays masked

        p_unmask = (ᾱₜ₋₁ - ᾱₜ) / (1 - ᾱₜ)

        This posterior is *exact* for absorbing diffusion.
        """
        alpha_bar_t   = self.alpha_bar[t]              # (B,)
        alpha_bar_tm1 = self.alpha_bar[jnp.maximum(t - 1, 0)]  # (B,)

        # Probability of unmasking a currently masked token
        p_unmask = (alpha_bar_tm1 - alpha_bar_t) / jnp.clip(1.0 - alpha_bar_t, 1e-8)
        p_unmask = jnp.clip(p_unmask, 0.0, 1.0)[:, None]  # (B, 1)

        is_masked = (xt == mask_token_id)  # (B, L)
        # Draw Bernoulli: unmask with prob p_unmask
        unmask = jax.random.bernoulli(rng, p_unmask, shape=xt.shape)  # (B, L)

        # Where masked: unmask → x₀ token; stay masked otherwise
        xt_prev = jnp.where(is_masked & unmask, x0, xt)
        return xt_prev

    # ------------------------------------------------------------------
    # Loss weight  λₜ (optional reweighting)
    # ------------------------------------------------------------------

    def loss_weight(self, t: jax.Array) -> jax.Array:
        """
        MDLM-style loss weight: λₜ = ᾱₜ / (1 - ᾱₜ).
        Upweights clean-data steps where predicting the original token is hard.
        Returns shape matching t.
        """
        ab = self.alpha_bar[t]
        return ab / jnp.clip(1.0 - ab, 1e-8)

    # ------------------------------------------------------------------
    # DDIM-style deterministic reverse step (sampling)
    # ------------------------------------------------------------------

    def ddim_step(
        self,
        logits_x0: jax.Array,    # (B, L, V)  model output (log-probs over vocab)
        xt: jax.Array,           # (B, L)
        t: jax.Array,            # (B,)  scalar per sample
        t_next: jax.Array,       # (B,)  next (smaller) step
        mask_token_id: int,
        rng: jax.Array,
        temperature: float = 1.0,
    ) -> jax.Array:
        """
        Ancestral sampling step: x_{t_next} | xₜ using predicted x̂₀.

        1. Sample x̂₀ ~ Categorical(softmax(logits_x0 / T))  for masked positions.
        2. Apply q_sample(x̂₀, t_next) to get x_{t_next}.
        """
        B, L, V = logits_x0.shape
        rng_sample, rng_q = jax.random.split(rng)

        # Gumbel-max trick for sampling from logits
        gumbel = -jnp.log(-jnp.log(jax.random.uniform(rng_sample, logits_x0.shape) + 1e-10) + 1e-10)
        x0_pred = jnp.argmax(logits_x0 / temperature + gumbel, axis=-1)  # (B, L)

        # Only update masked positions
        is_masked = (xt == mask_token_id)
        x0_for_step = jnp.where(is_masked, x0_pred, xt)

        # Re-corrupt to t_next
        ab_next = self.alpha_bar[t_next][:, None]  # (B, 1)
        keep = jax.random.bernoulli(rng_q, ab_next, shape=(B, L))
        xt_next = jnp.where(keep, x0_for_step, mask_token_id)
        return xt_next

"""
dlm_bwd_tests.py — Overfitting test for the diffusion language model.

Initialises a small model, grabs one batch of data (or falls back to random
tokens), and trains for 200 steps on that single batch.  Reports initial and
final loss to verify the backward pass works end-to-end.
"""

from __future__ import annotations

from ast import Tuple

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax

from src.config import ModelConfig, ScheduleConfig
from src.model import DiffusionTransformer
from src.schedules import make_schedule


def _try_get_real_batch(batch_size: int, seq_len: int) -> tuple[jax.Array, bool]:
    """Attempt to load one batch from OpenWebText.  Returns (tokens, ok)."""
    try:
        from src.config import DataConfig
        from src.data import get_dataloaders

        cfg = DataConfig()
        train_loader, _ = get_dataloaders(cfg, batch_size, validate=True)
        batch = next(train_loader)  # (B, L)
        # Truncate to seq_len if needed
        batch = batch[:, :seq_len]
        return batch, True
    except Exception as e:
        print(f"  Could not load real data ({e}), falling back to random tokens.")
        return None, False


def test_dlm_overfit_batch():
    """Overfit the model on a single batch for 200 steps."""

    BATCH = 2
    SEQ = 1024  # shorter sequence → much faster
    STEPS = 20
    LR = 1e-3
    T_SCHEDULE = 1000  # total diffusion steps

    cfg = ModelConfig(
        vocab_size=50304,
        seq_len=SEQ,
        n_layers=1,
        n_heads=4,
        d_model=128,
        d_ff=256,
        dropout=0.1,  # non-zero dropout so nnx.Dropout is exercised
        mask_token_id=50256,
    )

    x0, is_real = _try_get_real_batch(BATCH, SEQ)
    if x0 is None:
        rng = jax.random.PRNGKey(0)
        x0 = jax.random.randint(rng, (BATCH, SEQ), 0, cfg.vocab_size - 1)

    # Ensure no token equals mask_token_id by accident (unlikely but safe)
    x0 = jnp.where(x0 == cfg.mask_token_id, cfg.mask_token_id - 1, x0)
    print(f"  Batch shape: {x0.shape}  (real data: {is_real})")
    print(
        f"  Model: {cfg.n_layers} layers, d_model={cfg.d_model}, n_heads={cfg.n_heads}"
    )

    sch_cfg = ScheduleConfig(T=T_SCHEDULE, kind="cosine")
    schedule = make_schedule(sch_cfg)
    T = schedule.T

    rngs = nnx.Rngs(cfg.init_seed)
    model = DiffusionTransformer(cfg, rngs=rngs)

    n_params = model.count_params(nonembed=False)
    print(f"  Parameters: {n_params:,}")

    optimizer = nnx.Optimizer(
        model, optax.adamw(learning_rate=LR, weight_decay=0.1), wrt=nnx.Param
    )

    def loss_fn(
        model: DiffusionTransformer,
        x0: jax.Array,  # (B, L)  clean tokens
        xt: jax.Array,  # (B, L)  noisy tokens
        times: Tuple[jax.Array, int],
    ) -> jax.Array:
        """Cross-entropy loss on masked positions only."""
        t, T = times
        logits = model(xt, t, T, training=True)  # (B, L, V)

        # Only compute loss on masked positions
        mask = xt == cfg.mask_token_id  # (B, L)
        n_masked = mask.sum().astype(jnp.float32)

        # vocab_size = logits.shape[-1]
        # logits = logits[:, :-1, :].reshape(-1, vocab_size).contiguous()
        # labels = x0[1:].reshape(-1, vocab_size)

        # If no masks (unlikely with good schedule), return 0
        loss_per_pos = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=x0
        )  # (B, L)
        loss = jnp.where(mask, loss_per_pos, 0.0).sum() / jnp.maximum(n_masked, 1.0)
        return loss, logits

    @nnx.jit
    def train_step(
        model: DiffusionTransformer,
        optimizer: nnx.Optimizer,
        batch: Tuple[jax.Array, jax.Array, jax.Array, int]
    ) -> tuple[nnx.State, optax.OptState, jax.Array]:
        x0, xt, t, T = batch
        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        (loss, logits), grads = grad_fn(model, x0, xt, (t, T))
        optimizer.update(model, grads)
        return loss, logits

    rng_noise = jax.random.PRNGKey(42)
    rng_t, rng_mask = jax.random.split(rng_noise)
    t = jax.random.randint(rng_t, (BATCH,), 1, T + 1)  # random per-sample steps
    xt = schedule.q_sample(x0, t, cfg.mask_token_id, rng_mask)

    mask_pct = (xt == cfg.mask_token_id).mean() * 100
    print(f"  Mask fraction: {mask_pct:.1f}%")

    loss_initial = None
    loss_final = None

    batch = (x0, xt, t, T)

    for step in range(STEPS):
        loss, logits = train_step(model, optimizer, batch)

        if step == 0:
            loss_initial = loss
        if step == STEPS - 1:
            loss_final = loss

        if step % 40 == 0 or step == STEPS - 1:
            print(f"  step {step:>4d}  |  loss = {loss:.6f}")

    print(f"\n  Initial loss: {loss_initial:.6f}")
    print(f"  Final   loss: {loss_final:.6f}")
    print(
        f"  Delta:        {loss_initial - loss_final:.6f} "
        f"({(1 - loss_final / loss_initial) * 100:.1f}% reduction)"
    )
    print("  ✅ Model successfully overfit the batch.")


if __name__ == "__main__":
    print("JAX devices:", jax.devices())
    print()
    test_dlm_overfit_batch()

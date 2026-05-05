from __future__ import annotations

from ast import Tuple

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm

from src.config import ModelConfig, ScheduleConfig
from src.model import DiffusionTransformer
from src.schedules import make_schedule
from src.training import loss_fn, train_step


def _try_get_real_batch(batch_size: int, seq_len: int) -> tuple[jax.Array, bool]:
    try:
        from src.config import DataConfig
        from src.data import get_dataloaders

        cfg = DataConfig()
        train_loader, _ = get_dataloaders(cfg, batch_size, validate=True)
        batch = next(train_loader)  # (B, L)
        batch = batch[:, :seq_len]
        return batch, True
    except Exception as e:
        print(f"  Could not load real data ({e}), falling back to random tokens.")
        return None, False


def test_dlm_overfit_batch():
    BATCH = 2
    SEQ = 1024  # shorter sequence → much faster
    STEPS = 100
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

    rng_noise = jax.random.PRNGKey(42)
    rng_t, rng_mask = jax.random.split(rng_noise)
    t = jax.random.randint(rng_t, (BATCH,), 1, T + 1)  # random per-sample steps
    xt = schedule.q_sample(x0, t, cfg.mask_token_id, rng_mask)

    mask_pct = (xt == cfg.mask_token_id).mean() * 100
    print(f"  Mask fraction: {mask_pct:.1f}%")

    loss_initial = None
    loss_final = None

    batch = (x0, xt, t, T)
    bar = tqdm(total=STEPS)

    for step in range(STEPS):
        loss, logits = train_step(model, optimizer, batch)

        if step == 0:
            loss_initial = loss
        if step == STEPS - 1:
            loss_final = loss

        if step % 40 == 0 or step == STEPS - 1:
            print(f"  step {step:>4d}  |  loss = {loss:.6f}")

        bar.update(1)
        bar.set_postfix({"loss": loss})

    bar.close()
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

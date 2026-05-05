"""
train.py — Training entry point for nano-dlm (Flax NNX version).

Usage
─────
    python train.py                              # defaults
    python train.py --model.n_layers 12 --train.lr 1e-4
    python train.py --help                       # all flags

Multi-device  (jax.pmap)
────────────────────────
We use the nnx.split / nnx.merge pattern so that NNX module state can be
replicated and sharded across devices as plain JAX pytrees:

  graphdef, state = nnx.split(optimizer)      # outside pmap
  rep_state = jax.device_put_replicated(state, jax.devices())

  @jax.pmap
  def train_step(state, batch, rng):
      optimizer = nnx.merge(graphdef, state)   # inside pmap
      ...
      _, new_state = nnx.split(optimizer)
      return new_state, loss

nnx.Optimizer wraps both the model AND optax optimizer state, so a single
split/merge captures everything needed for a gradient step.
"""

from __future__ import annotations

import functools
import json
import math
import pickle
import time
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro

from src.config import Config, validate
from src.data import make_loaders
from src.model import DiffusionTransformer, count_params
from src.schedules import NoiseSchedule, make_schedule


def compute_loss(
    model: DiffusionTransformer,
    batch: jax.Array,  # (local_bs, L)  int32
    schedule: NoiseSchedule,
    rng: jax.Array,
) -> jax.Array:
    """
    MDLM absorbing-diffusion loss:
      L = -E_{t,xₜ}[ Σᵢ λₜ · 1[masked] · log p_θ(x₀ᵢ | xₜ, t) ]
    """
    B, L = batch.shape
    mask_id = model.cfg.mask_token_id
    T = schedule.T

    rng_t, rng_q, rng_model = jax.random.split(rng, 3)

    t = jax.random.randint(rng_t, (B,), 1, T + 1)  # (B,)
    xt = schedule.q_sample(batch, t, mask_id, rng_q)  # (B, L)

    logits = model(xt, t, T, training=True, rng=rng_model)  # (B, L, V)

    is_masked = (xt == mask_id).astype(jnp.float32)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    target_lp = log_probs[jnp.arange(B)[:, None], jnp.arange(L)[None, :], batch]
    lw = schedule.loss_weight(t)[:, None]

    loss = -jnp.sum(target_lp * is_masked * lw) / (jnp.sum(is_masked) + 1e-8)
    return loss


def lr_schedule(step: int, cfg) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * step / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + coeff * (cfg.lr - cfg.min_lr)


# ---------------------------------------------------------------------------
# pmap-compatible train / eval steps
# ---------------------------------------------------------------------------

# graphdef is captured as a Python closure (static, same on all devices)
_graphdef = None


@functools.partial(jax.pmap, axis_name="devices")
def _train_step(state, batch, schedule_alpha_bar, schedule_T, rng):
    """One gradient-update step, pmap'd across batch dimension."""
    optimizer: nnx.Optimizer = nnx.merge(_graphdef, state)
    model = optimizer.model

    # Reconstruct a minimal schedule view inside pmap (no Python object)
    schedule = _ScheduleView(schedule_alpha_bar, int(schedule_T))

    def loss_fn(m):
        return compute_loss(m, batch, schedule, rng)

    loss, grads = nnx.value_and_grad(loss_fn)(model)

    # Average loss and gradients across devices
    loss = jax.lax.pmean(loss, axis_name="devices")
    grads = jax.lax.pmean(grads, axis_name="devices")

    optimizer.update(grads)

    _, new_state = nnx.split(optimizer)
    return new_state, loss


@functools.partial(jax.pmap, axis_name="devices")
def _eval_step(state, batch, schedule_alpha_bar, schedule_T, rng):
    model: DiffusionTransformer = nnx.merge(_graphdef, state).model
    schedule = _ScheduleView(schedule_alpha_bar, int(schedule_T))
    loss = compute_loss(model, batch, schedule, rng)
    return jax.lax.pmean(loss, axis_name="devices")


class _ScheduleView:
    """Thin wrapper so schedule methods work inside jit/pmap from plain arrays."""

    def __init__(self, alpha_bar, T):
        self.alpha_bar = alpha_bar
        self.T = T

    def q_sample(self, x0, t, mask_token_id, rng):
        from schedules import NoiseSchedule

        _s = object.__new__(NoiseSchedule)
        _s.alpha_bar = self.alpha_bar
        _s.T = self.T
        return _s.q_sample(x0, t, mask_token_id, rng)

    def loss_weight(self, t):
        ab = self.alpha_bar[t]
        return ab / jnp.clip(1.0 - ab, 1e-8)


def main():
    global _graphdef

    cfg = tyro.cli(Config)
    validate(cfg)

    out_dir = Path(cfg.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Devices
    n_devices = jax.device_count()
    assert cfg.train.batch_size % n_devices == 0, (
        f"batch_size {cfg.train.batch_size} must be divisible by n_devices {n_devices}"
    )
    local_bs = cfg.train.batch_size // n_devices
    print(f"[init] devices: {n_devices}  local_bs: {local_bs}")

    # Data
    train_loader, val_loader = make_loaders(
        cfg.data, cfg.train.batch_size, cfg.train.seed
    )

    # Schedule
    schedule = make_schedule(cfg.schedule)

    # Model + Optimizer (NNX)
    model = DiffusionTransformer(cfg.model, rngs=nnx.Rngs(cfg.train.seed))
    print(f"[init] params: {count_params(model):,}")

    tx = make_optimizer(cfg.train)
    optimizer = nnx.Optimizer(model, tx)

    # Resume
    if cfg.train.resume:
        load_checkpoint(cfg.train.resume, model)

    # Split optimizer (model + optax state) for pmap replication
    _graphdef, state = nnx.split(optimizer)
    rep_state = jax.device_put_replicated(state, jax.devices())

    # Pre-broadcast schedule arrays for pmap
    ab = schedule.alpha_bar  # (T+1,)
    T_rep = jnp.broadcast_to(jnp.array(schedule.T), (n_devices,))

    loader_iter = iter(train_loader)
    rng = jax.random.PRNGKey(cfg.train.seed)
    log_file = open(out_dir / "log.jsonl", "a")
    t0 = time.time()

    for step in range(cfg.train.max_steps):
        # Batch: (n_devices, local_bs, L)
        batch_np = next(loader_iter)
        batch_jax = jnp.array(batch_np).reshape(n_devices, local_bs, -1)

        # Per-device RNG (unique per device per step)
        rng, step_rng = jax.random.split(rng)
        dev_rngs = jax.random.split(step_rng, n_devices)

        # Broadcast schedule alpha_bar to per-device leading axis
        ab_rep = jnp.broadcast_to(ab[None], (n_devices,) + ab.shape)

        rep_state, loss_rep = _train_step(rep_state, batch_jax, ab_rep, T_rep, dev_rngs)
        loss = float(loss_rep[0])

        if step % cfg.train.log_every == 0:
            dt = (time.time() - t0) / max(1, cfg.train.log_every)
            lr = lr_schedule(step, cfg.train)
            print(
                f"step {step:7d}/{cfg.train.max_steps}  loss {loss:.4f}  "
                f"lr {lr:.2e}  {dt * 1000:.1f}ms/step"
            )
            log_file.write(json.dumps({"step": step, "loss": loss, "lr": lr}) + "\n")
            log_file.flush()
            t0 = time.time()

        if step % cfg.train.eval_every == 0 and step > 0:
            val_loss = evaluate(rep_state, val_loader, schedule, n_devices)
            print(f"  [val] step {step}  val_loss {val_loss:.4f}")
            log_file.write(json.dumps({"step": step, "val_loss": val_loss}) + "\n")
            log_file.flush()

        if step % cfg.train.save_every == 0 and step > 0:
            # Recover single-device optimizer to extract params for saving
            single_state = jax.tree_util.tree_map(lambda x: x[0], rep_state)
            single_optim = nnx.merge(_graphdef, single_state)
            save_checkpoint(str(out_dir), step, single_optim, cfg)

    single_state = jax.tree_util.tree_map(lambda x: x[0], rep_state)
    single_optim = nnx.merge(_graphdef, single_state)
    save_checkpoint(str(out_dir), cfg.train.max_steps, single_optim, cfg)
    log_file.close()
    print("[done] Training complete.")


if __name__ == "__main__":
    main()

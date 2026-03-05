"""
train.py — Training entry point for nano-dlm.

Usage
─────
Single device (CPU/GPU/TPU):
    python train.py

Multi-device (e.g. 8 GPUs / TPU pod):
    python train.py --train.batch_size 512

Override any config field via CLI (powered by tyro):
    python train.py --model.n_layers 12 --model.d_model 768 \\
                    --train.lr 1e-4 --train.max_steps 500000 \\
                    --schedule.kind cosine --data.dataset wikitext103

The training loop
─────────────────
1.  Sample a random diffusion time t ~ Uniform{1, …, T}.
2.  Corrupt x₀ → xₜ via q_sample (absorbing / masking).
3.  Predict clean-token logits via DiffusionTransformer(xₜ, t).
4.  Compute cross-entropy loss *only at masked positions*.
5.  Backprop through Equinox model, clip gradients, AdamW step.

Multi-device strategy: jax.pmap over the batch dimension.
  • Model params are replicated across devices (standard data parallelism).
  • Gradients are summed across devices via pmean inside the pmap'd step.
  • No model parallelism — kept minimal for clarity.
"""

from __future__ import annotations
import os
import math
import time
import json
import pickle
import functools
from pathlib import Path
from typing import Any

import numpy as np
import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import tyro

from config import Config, validate
from schedules import make_schedule, NoiseSchedule
from model import DiffusionTransformer, count_params
from data import make_loaders


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def compute_loss(
    model: DiffusionTransformer,
    batch: jax.Array,        # (local_bs, L)  int32
    schedule: NoiseSchedule,
    rng: jax.Array,
    enable_dropout: bool = True,
) -> jax.Array:
    """
    MDLM / D3PM absorbing-diffusion loss:
        L = -E_{t, xₜ}[ Σ_{i: xₜᵢ=[MASK]} log p_θ(x₀ᵢ | xₜ, t) ]
    Returns scalar loss (mean over batch and masked positions).
    """
    B, L = batch.shape
    mask_id = model.cfg.mask_token_id
    T = schedule.T

    rng_t, rng_q, rng_model = jax.random.split(rng, 3)

    # Sample diffusion times  t ~ Uniform{1, …, T}
    t = jax.random.randint(rng_t, shape=(B,), minval=1, maxval=T + 1)  # (B,)

    # Forward corrupt
    xt = schedule.q_sample(batch, t, mask_id, rng_q)    # (B, L)

    # Model prediction
    logits = model(xt, t, T, enable_dropout=enable_dropout, key=rng_model)  # (B, L, V)

    # Cross-entropy at masked positions only
    is_masked = (xt == mask_id).astype(jnp.float32)     # (B, L)
    log_probs = jax.nn.log_softmax(logits, axis=-1)      # (B, L, V)

    # Gather log-prob of the ground-truth token
    # log_probs[b, l, batch[b, l]]
    target_log_probs = log_probs[
        jnp.arange(B)[:, None],
        jnp.arange(L)[None, :],
        batch,
    ]  # (B, L)

    # Optional MDLM loss re-weighting by λₜ = ᾱₜ / (1 - ᾱₜ)
    lw = schedule.loss_weight(t)[:, None]  # (B, 1)

    loss_per_token      = -target_log_probs * is_masked  # zero at unmasked
    weighted_loss       = loss_per_token * lw

    n_masked = jnp.sum(is_masked) + 1e-8
    return jnp.sum(weighted_loss) / n_masked


# ---------------------------------------------------------------------------
# LR schedule: linear warm-up → cosine decay
# ---------------------------------------------------------------------------

def lr_schedule(step: int, cfg) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * step / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    coeff    = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + coeff * (cfg.lr - cfg.min_lr)


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def make_optimizer(cfg) -> optax.GradientTransformation:
    schedule_fn = optax.join_schedules(
        schedules=[
            optax.linear_schedule(0.0, cfg.lr, transition_steps=cfg.warmup_steps),
            optax.cosine_decay_schedule(cfg.lr, cfg.max_steps - cfg.warmup_steps,
                                        alpha=cfg.min_lr / cfg.lr),
        ],
        boundaries=[cfg.warmup_steps],
    )
    tx = optax.chain(
        optax.clip_by_global_norm(cfg.clip_grad_norm) if cfg.clip_grad_norm > 0
            else optax.identity(),
        optax.adamw(learning_rate=schedule_fn,
                    b1=cfg.beta1, b2=cfg.beta2,
                    weight_decay=cfg.weight_decay),
    )
    return tx


# ---------------------------------------------------------------------------
# pmap-compatible train step
# ---------------------------------------------------------------------------

@functools.partial(jax.pmap, axis_name="devices",
                   in_axes=(0, 0, None, 0),
                   out_axes=(0, 0))
def train_step(
    params_and_model,        # model with replicated params (one shard per device)
    batch: jax.Array,        # (local_bs, L)  per device
    schedule: NoiseSchedule, # static (same on all devices)
    rng: jax.Array,          # (2,) per device — unique per device
):
    """One gradient update step, executed in parallel across devices."""
    model, opt_state = params_and_model

    def loss_fn(m):
        return compute_loss(m, batch, schedule, rng, enable_dropout=True)

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)

    # Sum gradients across devices
    grads = jax.lax.pmean(grads, axis_name="devices")
    loss  = jax.lax.pmean(loss,  axis_name="devices")

    updates, new_opt_state = optimizer.update(
        eqx.filter(grads, eqx.is_array),
        opt_state,
        eqx.filter(model, eqx.is_array),
    )
    new_model = eqx.apply_updates(model, updates)
    return (new_model, new_opt_state), loss


@functools.partial(jax.pmap, axis_name="devices",
                   in_axes=(0, 0, None, 0),
                   out_axes=0)
def eval_step(params_and_model, batch, schedule, rng):
    model, _ = params_and_model
    loss = compute_loss(model, batch, schedule, rng, enable_dropout=False)
    return jax.lax.pmean(loss, axis_name="devices")


# We define optimizer at module level so it can be used inside pmap'd functions.
# It is set in main() before calling train_step.
optimizer: optax.GradientTransformation = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(out_dir: str, step: int, model, opt_state, cfg: Config):
    ckpt_dir = Path(out_dir) / f"step_{step:07d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # Save model params (from first device's replica)
    model_single = jax.tree_util.tree_map(lambda x: x[0] if x.ndim > 0 else x, model)
    with open(ckpt_dir / "model.pkl", "wb") as f:
        pickle.dump(eqx.filter(model_single, eqx.is_array), f)
    with open(ckpt_dir / "config.json", "w") as f:
        # Serialize config (best effort for dataclasses)
        import dataclasses
        json.dump(dataclasses.asdict(cfg), f, indent=2)
    # Save latest pointer
    (Path(out_dir) / "latest").write_text(str(ckpt_dir))
    print(f"[ckpt] Saved checkpoint → {ckpt_dir}")


def load_checkpoint(ckpt_path: str, model, opt_state):
    """Load model weights from a checkpoint directory, returns updated model."""
    ckpt_dir = Path(ckpt_path)
    with open(ckpt_dir / "model.pkl", "rb") as f:
        saved_params = pickle.load(f)
    # Filter & replace arrays in model
    model = eqx.tree_at(
        lambda m: eqx.filter(m, eqx.is_array),
        model,
        saved_params,
    )
    print(f"[ckpt] Loaded checkpoint ← {ckpt_dir}")
    return model, opt_state


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

def evaluate(model_and_opt, val_loader, schedule, n_devices, n_batches=20):
    losses = []
    loader_iter = iter(val_loader)
    for _ in range(n_batches):
        batch_np = next(loader_iter)            # (global_bs, L)
        local_bs = batch_np.shape[0] // n_devices
        batch_jax = jnp.array(batch_np).reshape(n_devices, local_bs, -1)
        rng = jax.random.split(jax.random.PRNGKey(0), n_devices)  # eval: fixed rng
        loss = eval_step(model_and_opt, batch_jax, schedule, rng)
        losses.append(float(loss[0]))
    return float(np.mean(losses))


# ---------------------------------------------------------------------------
# Sampling  (auto-regressive in diffusion time, NOT token order)
# ---------------------------------------------------------------------------

@jax.jit
def sample(
    model: DiffusionTransformer,
    schedule: NoiseSchedule,
    n_samples: int,
    seq_len: int,
    rng: jax.Array,
    temperature: float = 1.0,
    n_steps: int = 50,
) -> jax.Array:
    """
    Generate n_samples sequences via DDIM-style reverse diffusion.
    Starts from fully masked xₜ and iteratively denoises.

    Returns: (n_samples, seq_len) int32 token ids.
    """
    mask_id = model.cfg.mask_token_id
    T = schedule.T

    # Start fully masked
    x = jnp.full((n_samples, seq_len), mask_id, dtype=jnp.int32)

    # Evenly-spaced time steps  T → 0
    steps = jnp.linspace(T, 0, n_steps + 1, dtype=jnp.int32)

    def body(carry, i):
        x, rng = carry
        t_cur  = jnp.full((n_samples,), steps[i],     dtype=jnp.int32)
        t_next = jnp.full((n_samples,), steps[i + 1], dtype=jnp.int32)
        rng, rng_step = jax.random.split(rng)
        logits = model(x, t_cur, T, enable_dropout=False)  # (B, L, V)
        x = schedule.ddim_step(logits, x, t_cur, t_next, mask_id, rng_step, temperature)
        return (x, rng), None

    (x, _), _ = jax.lax.scan(body, (x, rng), jnp.arange(n_steps))
    return x


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    global optimizer  # set before pmap'd train_step is called

    # ── Parse config from CLI ──────────────────────────────────────────────
    cfg = tyro.cli(Config)
    validate(cfg)

    out_dir = Path(cfg.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Devices ────────────────────────────────────────────────────────────
    n_devices = jax.device_count()
    assert cfg.train.batch_size % n_devices == 0, (
        f"batch_size ({cfg.train.batch_size}) must be divisible by n_devices ({n_devices})"
    )
    local_bs = cfg.train.batch_size // n_devices
    print(f"[init] devices: {n_devices}   local batch/device: {local_bs}")

    # ── Data ───────────────────────────────────────────────────────────────
    train_loader, val_loader = make_loaders(cfg.data, cfg.train.batch_size, cfg.train.seed)

    # ── Schedule ───────────────────────────────────────────────────────────
    schedule = make_schedule(cfg.schedule)

    # ── Model ──────────────────────────────────────────────────────────────
    rng = jax.random.PRNGKey(cfg.train.seed)
    rng, model_key = jax.random.split(rng)
    model = DiffusionTransformer(cfg.model, model_key)
    print(f"[init] model params: {count_params(model):,}")

    # ── Optimizer ──────────────────────────────────────────────────────────
    optimizer = make_optimizer(cfg.train)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # ── Resume ─────────────────────────────────────────────────────────────
    start_step = 0
    if cfg.train.resume:
        model, opt_state = load_checkpoint(cfg.train.resume, model, opt_state)

    # ── Replicate across devices ──────────────────────────────────────────
    model_rep    = jax.device_put_replicated(model,     jax.devices())
    opt_state_rep = jax.device_put_replicated(opt_state, jax.devices())
    state = (model_rep, opt_state_rep)

    # ── Training loop ────────────────────────────────────────────────────
    loader_iter = iter(train_loader)
    t0 = time.time()

    log_path = out_dir / "log.jsonl"
    log_file = open(log_path, "a")

    for step in range(start_step, cfg.train.max_steps):

        # ── Batch ─────────────────────────────────────────────────────────
        batch_np = next(loader_iter)                               # (B, L)
        batch_jax = jnp.array(batch_np).reshape(n_devices, local_bs, -1)

        # ── Per-device RNG (unique per device per step) ───────────────────
        rng, step_rng = jax.random.split(rng)
        per_device_rngs = jax.random.split(step_rng, n_devices)  # (n_devices, 2)

        # ── Step ──────────────────────────────────────────────────────────
        state, loss_rep = train_step(state, batch_jax, schedule, per_device_rngs)
        loss = float(loss_rep[0])

        # ── Logging ───────────────────────────────────────────────────────
        if step % cfg.train.log_every == 0:
            dt = (time.time() - t0) / max(1, cfg.train.log_every)
            current_lr = lr_schedule(step, cfg.train)
            print(
                f"step {step:7d}/{cfg.train.max_steps}  "
                f"loss {loss:.4f}  lr {current_lr:.2e}  "
                f"{dt*1000:.1f}ms/step"
            )
            log_file.write(json.dumps({"step": step, "loss": loss, "lr": current_lr}) + "\n")
            log_file.flush()
            t0 = time.time()

        # ── Validation ────────────────────────────────────────────────────
        if step % cfg.train.eval_every == 0 and step > 0:
            val_loss = evaluate(state, val_loader, schedule, n_devices)
            print(f"  [val] step {step}  val_loss {val_loss:.4f}")
            log_file.write(json.dumps({"step": step, "val_loss": val_loss}) + "\n")
            log_file.flush()

        # ── Checkpoint ────────────────────────────────────────────────────
        if step % cfg.train.save_every == 0 and step > 0:
            model_single, opt_single = jax.tree_util.tree_map(
                lambda x: x[0], state
            )
            save_checkpoint(str(out_dir), step, model_single, opt_single, cfg)

    # ── Final checkpoint ──────────────────────────────────────────────────
    model_single, opt_single = jax.tree_util.tree_map(lambda x: x[0], state)
    save_checkpoint(str(out_dir), cfg.train.max_steps, model_single, opt_single, cfg)
    log_file.close()
    print("[done] Training complete.")


if __name__ == "__main__":
    main()

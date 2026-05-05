import typing as tp

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax

from src.model import DiffusionTransformer


def loss_fn(
    model: DiffusionTransformer,
    x0: jax.Array,
    xt: jax.Array,
    times: tp.Tuple[jax.Array, int],
) -> jax.Array:
    t, T = times
    logits = model(xt, t, T, training=True)  # (B, L, V)

    mask = xt == model.cfg.mask_token_id  # (B, L)
    n_masked = mask.sum().astype(jnp.float32)

    loss_per_pos = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=x0
    )
    loss = jnp.where(mask, loss_per_pos, 0.0).sum() / jnp.maximum(n_masked, 1.0)
    return loss, logits


@nnx.jit
def train_step(
    model: DiffusionTransformer,
    optimizer: nnx.Optimizer,
    batch: tp.Tuple[jax.Array, jax.Array, jax.Array, int],
) -> tp.Tuple[jax.Array, jax.Array]:
    x0, xt, t, T = batch
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, x0, xt, (t, T))
    optimizer.update(model, grads)
    return loss, logits


def make_optimizer(cfg) -> optax.GradientTransformation:
    schedule_fn = optax.join_schedules(
        schedules=[
            optax.linear_schedule(0.0, cfg.lr, cfg.warmup_steps),
            optax.cosine_decay_schedule(
                cfg.lr, cfg.max_steps - cfg.warmup_steps, alpha=cfg.min_lr / cfg.lr
            ),
        ],
        boundaries=[cfg.warmup_steps],
    )
    return optax.chain(
        optax.clip_by_global_norm(cfg.clip_grad_norm)
        if cfg.clip_grad_norm > 0
        else optax.identity(),
        optax.adamw(
            learning_rate=schedule_fn,
            b1=cfg.beta1,
            b2=cfg.beta2,
            weight_decay=cfg.weight_decay,
        ),
    )

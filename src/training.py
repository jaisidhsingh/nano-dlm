import typing as tp

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax
from src.model import DiffusionTransformer

from src.config import TrainConfig


def loss_fn(
    model: DiffusionTransformer,
    x0: jax.Array,
    xt: jax.Array,
    times: tp.Tuple[jax.Array, int],
) -> tp.Tuple[jax.Array, jax.Array]:
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


def get_lr_schedule(cfg: TrainConfig) -> optax.Schedule:
    schedule_fn = None
    if cfg.lr_schedule == "none":
        schedule_fn = cfg.lr

    if cfg.lr_schedule == "warmup_cosine":
        schedule_fn = optax.join_schedules(
            schedules=[
                optax.linear_schedule(0.0, cfg.lr, cfg.warmup_steps),
                optax.cosine_decay_schedule(
                    cfg.lr, cfg.max_steps - cfg.warmup_steps, alpha=cfg.min_lr / cfg.lr
                ),
            ],
            boundaries=[cfg.warmup_steps],
        )
    elif cfg.lr_schedule == "wsd":
        schedule_fn = optax.join_schedules(
            schedules=[
                optax.linear_schedule(0.0, cfg.lr, cfg.warmup_steps),
                optax.linear_schedule(
                    cfg.lr, cfg.lr, cfg.cooldown_start_steps - cfg.warmup_steps
                ),
                optax.linear_schedule(
                    cfg.lr, cfg.min_lr, cfg.max_steps - cfg.cooldown_start_steps
                ),
            ],
            boundaries=[cfg.warmup_steps, cfg.cooldown_start_steps],
        )
    elif cfg.lr_schedule == "warmup_constant":
        schedule_fn = optax.join_schedules(
            schedules=[
                optax.linear_schedule(0.0, cfg.lr, cfg.warmup_steps),
                optax.linear_schedule(cfg.lr, cfg.lr, cfg.max_steps - cfg.warmup_steps),
            ],
            boundaries=[cfg.warmup_steps],
        )
    else:
        raise NotImplementedError(
            "The kind of learning schedule specified is not implemented in `src/training.py`"
        )
    return schedule_fn


def get_optimizer_kwargs(cfg: TrainConfig):
    kwargs = {}
    if cfg.optimizer == "adamw":
        kwargs = {
            "b1": cfg.beta1,
            "b2": cfg.beta2,
            "eps": getattr(cfg, "eps", 1e-8),
        }
    else:
        raise NotImplementedError(
            "The kind of optimizer specified is not implemented in `src/training.py`"
        )
    return kwargs


def init_optimizer_alg(
    cfg: TrainConfig,
) -> tp.Union[optax.GradientTransformation, optax.MultiSteps]:
    schedule_fn = get_lr_schedule(cfg)
    optimizer_kwargs = get_optimizer_kwargs(cfg)
    optimizer_ref = (
        getattr(optax, cfg.optimizer)
        if cfg.optimizer != "muon"
        else getattr(optax.contrib, cfg.optimizer)
    )

    base_opt_alg = optax.chain(
        optax.clip_by_global_norm(cfg.clip_grad_norm)
        if cfg.clip_grad_norm > 0
        else optax.identity(),
        optimizer_ref(
            learning_rate=schedule_fn, weight_decay=cfg.weight_decay, **optimizer_kwargs
        ),
    )
    if cfg.grad_acc_steps > 1:
        return optax.MultiSteps(base_opt_alg, every_k_schedule=cfg.grad_acc_steps)
    return base_opt_alg

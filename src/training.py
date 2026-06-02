import typing as tp

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax
from src.config import TrainConfig
from src.model import DiffusionTransformer
from src.schedules import NoiseSchedule


def loss_fn(
    model: DiffusionTransformer,
    schedule: NoiseSchedule,
    inputs: tp.Dict,
    labels: jax.Array,
) -> jax.Array:
    logits = model(**inputs)  # (B, L, V)

    mask = inputs["input_ids"] == model.cfg.mask_token_id  # (B, L)
    n_masked = mask.sum().astype(jnp.float32)

    loss_per_pos = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=labels
    )
    loss_weights = schedule.loss_weight(inputs["timesteps"])[:, None]
    loss = (loss_weights * jnp.where(mask, loss_per_pos, 0.0)).sum() / jnp.maximum(
        n_masked, 1.0
    )
    return loss


@nnx.jit
def get_grad_norm(grads) -> jax.Array:
    grad_norm = jnp.sqrt(sum(jnp.sum(x**2) for x in jax.tree.leaves(grads)))
    return grad_norm


@nnx.jit
def get_parameter_norm(model: DiffusionTransformer) -> jax.Array:
    param_norm = jnp.sqrt(
        sum(jnp.sum(x**2) for x in jax.tree.leaves(nnx.state(model, nnx.Param)))
    )
    return param_norm


@nnx.jit
def train_step(
    model: DiffusionTransformer,
    optimizer: nnx.Optimizer,
    schedule: NoiseSchedule,
    inputs: tp.Dict,
    labels: jax.Array
) -> tp.Tuple[tp.Dict[str, jax.Array], tp.Dict[str, jax.Array]]:
    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grads = grad_fn(model, schedule, inputs, labels)

    grad_norm = get_grad_norm(grads)
    optimizer.update(model, grads)

    parameter_norm = get_parameter_norm(model)
    train_logs = {"loss": loss, "ppl": jnp.exp(loss)}
    param_logs = {"parameter_norm": parameter_norm, "grad_norm": grad_norm}

    return train_logs, param_logs


@nnx.jit
def val_step(
    model: DiffusionTransformer,
    schedule: NoiseSchedule,
    inputs: tp.Dict,
    labels: jax.Array
) -> tp.Dict:
    x0, xt, t = batch
    loss = loss_fn(model, schedule, inputs, labels)
    return {"loss": loss, "ppl": jnp.exp(loss)}


def validation_loop(model: DiffusionTransformer):


def get_lr_schedule(cfg: TrainConfig) -> tp.Union[float, optax.Schedule]:
    schedule_fn = None
    if cfg.lr_schedule == "none":
        schedule_fn = cfg.lr

    elif cfg.lr_schedule == "warmup_cosine":
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

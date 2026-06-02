import json
import os
import typing as tp
from dataclasses import asdict

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import wandb

from src.config import Config, ExperimentConfig
from src.model import DiffusionTransformer
from src.schedules import NoiseSchedule
from src.data import prepare_batch


class MetricLogger:
    def __init__(self, metrics: tp.List = ["train", "val", "params", "info"]):
        self.metrics = metrics
        self.data = {m: {} for m in metrics}
        self.step_number: int = -1

    def step(self, logs: dict, step: int):
        self.step_numer = step

        for m in logs.keys():
            assert m in self.metrics, "Unsupported metric provided"
            self.data[m][step] = {k: round(float(v), 4) for k, v in logs[m].items()}

    def save_logs(self, cfg: ExperimentConfig):
        save_folder = os.path.join(
            cfg.out_dir, cfg.run_name, f"step_{self.step_number}"
        )
        os.makedirs(save_folder, exist_ok=True)

        with open(os.path.join(save_folder, "logs.json"), "w") as f:
            json.dump(self.data, f)

    def log_to_wandb(self, step):
        # always do this after `self.step(logs, step)`
        for m in self.metrics:
            wandb.log(
                {f"{m}/{k}": v for k, v in self.data[m][step].items()},
                step=step,
            )


def save_checkpoint(
    checkpointer: ocp.StandardCheckpointer,
    cfg: Config,
    model: DiffusionTransformer,
    optimizer: nnx.Optimizer,
    metric_logger: MetricLogger,
    step: int,
):
    ckpt_folder = os.path.join(cfg.exp.out_dir, f"step_{step}")
    os.makedirs(ckpt_folder, exist_ok=True)

    model_save_folder = os.path.join(ckpt_folder, "model_state")
    os.makedirs(model_save_folder, exist_ok=True)

    opt_save_folder = os.path.join(ckpt_folder, "optimizer_state")
    os.makedirs(opt_save_folder, exist_ok=True)

    graphdef, model_state = nnx.split(model)
    checkpointer.save(model_save_folder, model_state)

    _, opt_full_state = nnx.split(optimizer)
    checkpointer.save(opt_save_folder, nnx.to_pure_dict(opt_full_state))

    metric_logger.save_logs(cfg.exp)
    with open(os.path.join(ckpt_folder, "config.json"), "w") as f:
        json.dump(asdict(cfg), f)


def load_checkpoint(cfg: Config, optimizer: nnx.Optimizer):
    checkpointer = ocp.StandardCheckpointer()
    ckpt_folder = cfg.exp.resume_folder

    abstract_model = nnx.eval_shape(
        lambda: DiffusionTransformer(cfg.model, rngs=nnx.Rngs(cfg.model.init_seed))
    )
    graphdef, abstract_state = nnx.split(abstract_model)
    model_state = checkpointer.restore(
        os.path.join(ckpt_folder, "model_state"), abstract_state
    )
    model = nnx.merge(graphdef, model_state)

    restored_opt_dict = checkpointer.restore(
        os.path.join(ckpt_folder, "optimizer_state")
    )
    _, opt_state = nnx.split(optimizer)
    nnx.replace_by_pure_dict(opt_state, restored_opt_dict)
    nnx.update(optimizer, opt_state)

    return model, optimizer


def validation_loop(cfg: Config, model: DiffusionTransformer, val_loader: tp.Iterator, noise_schedule: NoiseSchedule, rng_t: jax.random.PRNGKey, rng_mask: jax.random.PRNGKey) -> tp.Dict:
    val_logs = {}

    for raw_tokens in val_loader:
        batch, _ = prepare_batch(raw_tokens, noise_schedule, cfg, rng_t, rng_mask, training=False)
        labels = batch.pop("labels")
        per_val_step_logs = val_step(model, noise_schedule, batch, labels)

        for k in per_val_step_logs.keys():
            if k not in val_logs:
              val_logs[k] = 0
          val_logs[k] += per_val_step_logs[k]

    for k, v in val_logs.items():
        val_logs[k] = v / n

    return val_logs


@nnx.jit
def sample(
    model: DiffusionTransformer,
    schedule: NoiseSchedule,
    n_samples: int,
    seq_len: int,
    rng: jax.Array,
    temperature: float = 1.0,
    n_steps: int = 50,
) -> jax.Array:
    mask_id = model.cfg.mask_token_id
    T = schedule.T

    x = jnp.full((n_samples, seq_len), mask_id, dtype=jnp.int32)
    steps = jnp.linspace(T, 0, n_steps + 1, dtype=jnp.int32)

    def body(carry, i):
        x, rng = carry
        t_cur = jnp.full((n_samples,), steps[i], dtype=jnp.int32)
        t_next = jnp.full((n_samples,), steps[i + 1], dtype=jnp.int32)
        rng, rng_step = jax.random.split(rng)
        logits = model(x, t_cur, training=False)
        x = schedule.ddim_step(logits, x, t_cur, t_next, mask_id, rng_step, temperature)
        return (x, rng), None

    (x, _), _ = jax.lax.scan(body, (x, rng), jnp.arange(n_steps))
    return x

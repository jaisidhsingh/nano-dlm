from __future__ import annotations

import os
from dataclasses import asdict

import flax.nnx as nnx
import jax
import orbax.checkpoint as ocp
import tyro
import wandb
from tqdm import tqdm

from src.config import Config
from src.data import get_dataloaders
from src.model import DiffusionTransformer
from src.schedules import make_schedule
from src.training import init_optimizer_alg, train_step, val_step
from src.utils import MetricLogger, load_checkpoint, save_checkpoint


def main(cfg: Config):
    # create model and optimizer
    rngs = nnx.Rngs(cfg.model.init_seed)
    model = DiffusionTransformer(cfg.model, rngs=rngs)
    n_params = model.count_params(nonembed=False)
    print(f"Initialized diffusion language model with no. of parameters = {n_params:,}")

    opt_alg = init_optimizer_alg(cfg.train)
    optimizer = nnx.Optimizer(model, opt_alg, wrt=nnx.Param)

    if cfg.exp.resume and os.path.exists(cfg.exp.resume_folder):
        model, optimizer = load_checkpoint(cfg, optimizer)

    # load data
    train_loader, val_loader = get_dataloaders(
        cfg.data, cfg.train.batch_size, validate=True
    )

    # noise (token masking) schedule
    noise_schedule = make_schedule(cfg.schedule)
    rng_noise = jax.random.PRNGKey(42)
    T = noise_schedule.T

    # initialize wandb run (only if set)
    if cfg.exp.use_wandb:
        wandb.init(
            name=cfg.exp.run_name, project=cfg.exp.project_name, config=asdict(cfg)
        )

    bar = tqdm(total=cfg.train.max_steps)
    metric_logger = MetricLogger()
    token_count = 0

    if cfg.exp.saving:
        checkpointer = ocp.StandardCheckpointer()

    for step in range(cfg.train.max_steps):
        x0 = next(train_loader)
        token_count += int(x0.shape[0] * x0.shape[1])

        # advance the rng key, the basis of which we sample timesteps and noise
        # if we don't do `...split(rng_noise, 3)`, then we would have the same timesteps
        # and corruptions for every batch
        rng_noise, rng_t, rng_mask = jax.random.split(rng_noise, 3)
        t = jax.random.randint(rng_t, (cfg.train.batch_size,), 1, T + 1)

        # random per-sample steps
        xt = noise_schedule.q_sample(x0, t, cfg.model.mask_token_id, rng_mask)
        mask_pct = (xt == cfg.model.mask_token_id).mean() * 100

        batch = (x0, xt, t) if cfg.model.time_conditioning else (x0, xt, None)
        train_logs, param_logs = train_step(model, optimizer, noise_schedule, batch)

        if step % cfg.exp.eval_every == 0:
            val_logs = val_step(model, noise_schedule, batch)
        else:
            val_logs = {}

        if step % cfg.exp.log_every == 0:
            info = {"tokens": token_count, "mask_pct": mask_pct}
            metric_logger.step(
                {
                    "train": train_logs,
                    "params": param_logs,
                    "val": val_logs,
                    "info": info,
                },
                step=step,
            )

            if cfg.exp.use_wandb:
                metric_logger.log_to_wandb(step)

        if cfg.exp.saving and step % cfg.exp.save_every == 0:
            save_checkpoint(checkpointer, cfg, model, optimizer, metric_logger, step)

        bar.set_postfix(
            {
                "mask_pct": float(mask_pct),
                "train_loss": train_logs["loss"],
                "train_ppl": train_logs["ppl"],
            }
        )
        bar.update(1)

    bar.close()
    metric_logger.save_logs(cfg.exp)


if __name__ == "__main__":
    cfg = tyro.cli(Config, default=Config())
    main(cfg)

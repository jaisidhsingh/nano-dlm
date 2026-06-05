from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import asdict

import flax.nnx as nnx
import jax
import modal
import orbax.checkpoint as ocp
import tyro
import wandb
from tqdm import tqdm

from src.config import Config
from src.data import get_dataloaders
from src.model import DiffusionTransformer, init_model
from src.schedules import make_schedule
from src.training import init_optimizer_alg, train_step
from src.utils import (
  MetricLogger,
  load_checkpoint,
  prepare_batch,
  save_checkpoint,
  validation_loop,
)

app = modal.App('nano-dlm-2xh100-training')


@app.function()
def train_dlm(cfg: Config):
  # initialize our mesh of devices
  mesh = jax.make_mesh((jax.device_count(),), ('data',))
  nnx.use_eager_sharding(True)

  # put everything under the operable mesh
  with jax.set_mesh(mesh):
    # create model and optimizer
    model = init_model(cfg.model)(nnx.Rngs(cfg.model.init_seed))
    # model = DiffusionTransformer(cfg.model)
    n_params = model.count_params(nonembed=False)
    print(f'Initialized diffusion language model with no. of parameters = {n_params:,}')

    opt_alg = init_optimizer_alg(cfg.train)
    optimizer = nnx.Optimizer(model, opt_alg, wrt=nnx.Param)

    if cfg.exp.resume and os.path.exists(cfg.exp.resume_folder):
      model, optimizer = load_checkpoint(cfg, optimizer)

    # load data
    train_loader, val_dataset = get_dataloaders(cfg.data, cfg.train.batch_size, validate=True)

    # noise (token masking) schedule
    noise_schedule = make_schedule(cfg.schedule)
    rng_noise = jax.random.PRNGKey(cfg.schedule.noise_seed_train)
    rng_noise_val = jax.random.PRNGKey(cfg.schedule.noise_seed_val)

    # initialize wandb run (only if set)
    if cfg.exp.use_wandb:
      wandb.init(name=cfg.exp.run_name, project=cfg.exp.project_name, config=asdict(cfg))

    bar = tqdm(total=cfg.train.max_steps)
    metric_logger = MetricLogger()
    token_count = 0
    checkpointer = ocp.StandardCheckpointer()

    # training loop
    step = 0
    last_val_loss = -1.0
    for step in range(1, cfg.train.max_steps + 1):
      raw_tokens = next(train_loader)

      # advance the rng key, the basis of which we sample timesteps and noise
      # if we don't do `...split(rng_noise, 3)`, then we would have the same timesteps
      # and corruptions for every batch
      rng_noise, rng_t, rng_mask = jax.random.split(rng_noise, 3)

      # get training data
      batch, batch_info = prepare_batch(raw_tokens, noise_schedule, cfg, rng_t, rng_mask, training=True)
      labels = batch.pop('labels')
      token_count += batch_info['batch_tokens']
      batch_info['token_count'] = token_count

      # take a backward step (gradient accumulation taken care of)
      train_logs, param_logs = train_step(model, optimizer, noise_schedule, batch, labels)

      # validate
      val_logs = None
      if step % cfg.exp.eval_every == 0:
        rng_noise, rng_t_val, rng_mask_val = jax.random.split(rng_noise, 3)
        val_iter = iter(val_dataset)
        val_logs = validation_loop(cfg, model, val_iter, noise_schedule, rng_noise_val)
        last_val_loss = val_logs['loss']
      else:
        val_logs = {}

      # logging
      if step % cfg.exp.log_every == 0:
        metric_logger.step(
          {
            'train': train_logs,
            'params': param_logs,
            'val': val_logs,
            'info': batch_info,
          },
          step=step,
        )

        if cfg.exp.use_wandb:
          metric_logger.log_to_wandb(step)

      # save checkpoint
      if cfg.exp.saving and step % cfg.exp.save_every == 0:
        save_checkpoint(checkpointer, cfg, model, optimizer, metric_logger, step)

      bar_logs = {'train_loss': float(train_logs['loss']), 'last_val_loss': last_val_loss, 'tokens': f'{round(token_count / 1e6, 2)}M'}
      bar.set_postfix(bar_logs)
      bar.update(1)

    bar.close()
    if cfg.exp.save_last:
      save_checkpoint(checkpointer, cfg, model, optimizer, metric_logger, step)


@app.local_entrypoint()
def main():
  cfg = tyro.cli(Config, default=Config())
  train_dlm.remote(cfg)

from __future__ import annotations

import json

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import tyro
from tqdm import tqdm

from src.config import Config
from src.data import get_dataloaders
from src.model import DiffusionTransformer
from src.schedules import make_schedule
from src.training import init_optimizer_alg, train_step, val_step


def main(cfg: Config):
    # create model and optimizer
    rngs = nnx.Rngs(cfg.model.init_seed)
    model = DiffusionTransformer(cfg.model, rngs=rngs)
    n_params = model.count_params(nonembed=False)
    print(f"Initialized diffusion language model with no. of parameters = {n_params:,}")

    opt_alg = init_optimizer_alg(cfg.train)
    optimizer = nnx.Optimizer(model, opt_alg, wrt=nnx.Param)

    # load data
    train_loader, val_loader = get_dataloaders(
        cfg.data, cfg.train.batch_size, validate=True
    )

    # noise (token masking) schedule
    noise_schedule = make_schedule(cfg.schedule)
    rng_noise = jax.random.PRNGKey(42)
    T = noise_schedule.T

    bar = tqdm(total=cfg.train.max_steps)
    logs = {"train": {}, "val": {}}

    for step in range(cfg.train.max_steps):
        x0 = next(train_loader)

        rng_t, rng_mask = jax.random.split(rng_noise)
        t = jax.random.randint(
            rng_t, (cfg.train.batch_size,), 1, T + 1
        )  # random per-sample steps
        xt = noise_schedule.q_sample(x0, t, cfg.model.mask_token_id, rng_mask)
        mask_pct = (xt == cfg.model.mask_token_id).mean() * 100

        batch = (x0, xt, t) if cfg.model.time_conditioning else (x0, xt, None)
        train_loss, train_logits = train_step(model, optimizer, batch)

        logs["train"][step] = {
            "loss": train_loss.item(),
            "ppl": jnp.exp(train_loss).item(),
        }

        if step % cfg.exp.eval_every == 0:
            val_loss, val_logits = val_step(model, batch)
            logs["val"][step] = {
                "loss": val_loss.item(),
                "ppl": jnp.exp(val_loss).item(),
            }

        bar.update(1)
        bar.set_postfix({"train_loss": train_loss})

    bar.close()
    with open("/fast/jsingh/nano-dlm_single_gpu_42M_test_run.json", "w") as f:
        json.dump(logs, f)


if __name__ == "__main__":
    cfg = tyro.cli(Config, default=Config())
    main(cfg)

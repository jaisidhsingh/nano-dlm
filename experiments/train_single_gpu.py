from __future__ import annotations

import typing as tp

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax
import tyro
from tqdm import tqdm

from src.config import Config, ModelConfig, ScheduleConfig
from src.data import get_dataloaders
from src.model import DiffusionTransformer
from src.schedules import make_schedule
from src.training import loss_fn, train_step

BATCH = 2
SEQ = 1024  # shorter sequence → much faster
STEPS = 100
LR = 1e-3
T_SCHEDULE = 1000  # total diffusion steps

model_cfg = ModelConfig(
    vocab_size=50304,
    seq_len=SEQ,
    n_layers=1,
    n_heads=4,
    d_model=128,
    d_ff=256,
    dropout=0.1,  # non-zero dropout so nnx.Dropout is exercised
    mask_token_id=50256,
)
sch_cfg = ScheduleConfig(T=T_SCHEDULE, kind="cosine")

cfg = Config(model=model_cfg, schedule=sch_cfg)


def main(cfg: Config):
    rngs = nnx.Rngs(cfg.model.init_seed)
    model = DiffusionTransformer(cfg.model, rngs=rngs)

    n_params = model.count_params(nonembed=False)
    print(f"Initialized diffusion language model with no. of parameters = {n_params:,}")

    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=LR,
        warmup_steps=cfg.train.warmup_steps,
        decay_steps=STEPS,
        end_value=1e-5,
    )

    optimizer = nnx.Optimizer(
        model,
        optax.adamw(learning_rate=lr_schedule, weight_decay=0.1),
        wrt=nnx.Param,
    )

    train_loader, val_loader = get_dataloaders(
        cfg.data, cfg.train.batch_size, validate=True
    )
    noise_schedule = make_schedule(cfg.schedule)
    rng_noise = jax.random.PRNGKey(42)
    T = noise_schedule.T

    bar = tqdm(total=cfg.train.max_steps)
    for step in range(cfg.train.max_steps):
        x0 = next(train_loader)

        rng_t, rng_mask = jax.random.split(rng_noise)
        t = jax.random.randint(rng_t, (BATCH,), 1, T + 1)  # random per-sample steps
        xt = noise_schedule.q_sample(x0, t, cfg.mask_token_id, rng_mask)

        mask_pct = (xt == cfg.model.mask_token_id).mean() * 100

        batch = (x0, xt, t, T)
        loss, logits = train_step(model, optimizer, batch)

        bar.update(1)
        bar.set_postfix({"loss": loss})

    bar.close()


if __name__ == "__main__":
    cfg = tyro.cli(Config, default=Config())
    main(cfg)

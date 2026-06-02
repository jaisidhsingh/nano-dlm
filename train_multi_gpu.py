from __future__ import annotations

import json
from functools import partial

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import tyro
from jax.sharding import PartitionSpec as P
from src.config import Config
from src.data import get_dataloaders
from src.model import DiffusionTransformer
from src.schedules import make_schedule
from src.training import init_optimizer_alg, train_step, val_step
from tqdm import tqdm


@partial(jax.jit, static_argnums=(0,))
def init_model(model_cfg):
    return DiffusionTransformer(model_cfg, rngs=nnx.Rngs(model_cfg.init_seed))


def main(cfg: Config):
    # single axis of sharding: data parallel only
    mesh = jax.make_mesh((jax.device_count(),), ("data",))
    nnx.use_eager_sharding(True)

    # now do everything under the mesh we made for ddp
    with jax.set_mesh(mesh):
        # load model and optimizer
        model = init_model(cfg.model)


if __name__ == "__main__":
    cfg = tyro.cli(Config, default=Config())
    main(cfg)

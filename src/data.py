from __future__ import annotations

import typing as tp

import jax
import jax.numpy as jnp
import tiktoken
from jax.sharding import PartitionSpec as P

from src.config import Config, DataConfig
from src.schedules import NoiseSchedule


class Tokenizer:
  def __init__(self, tokenizer_id: str = 'gpt2'):
    self._tok = tiktoken.get_encoding(tokenizer_id)

  def encode(self, text: str) -> jax.Array:
    return jnp.array(self._tok.encode(text), dtype=jnp.int32)

  def decode(self, tokens: jax.Array) -> str:
    return self._tok.decode(tokens.tolist())

  def __call__(self, text: str) -> jax.Array:
    return self.encode(text)

  def __len__(self):
    return self._tok.n_vocab


def get_dataloaders(cfg: DataConfig, batch_size: int, validate: bool = True) -> tp.Union[tp.Tuple, tp.Iterable]:
  from datasets import load_from_disk
  from grain import MapDataset

  train_source = load_from_disk(cfg.trainset_path)
  if validate:
    val_source = load_from_disk(cfg.validset_path)

  collate_fn = None

  def owt_nn_gpt2_collate_fn(sample: dict) -> jax.Array:
    return jnp.array(sample['tokens'], dtype=jnp.int64)

  if 'gpt2' in cfg.trainset_path:
    collate_fn = owt_nn_gpt2_collate_fn

  train_dataset = MapDataset.source(train_source).shuffle(cfg.shuffle_seed).map(collate_fn).to_iter_dataset().batch(batch_size)

  if validate:
    val_dataset = MapDataset.source(val_source).shuffle(cfg.shuffle_seed).map(collate_fn).to_iter_dataset().batch(batch_size)

  train_loader = iter(train_dataset)
  if validate:
    return train_loader, val_dataset
  return train_loader


def prepare_batch(
  raw_tokens: jax.Array,
  noise_schedule: NoiseSchedule,
  cfg: Config,
  rng_t: jax.Array,
  rng_mask: jax.Array,
  training: bool = True,
) -> tp.Tuple:
  x0 = jax.device_put(raw_tokens, P('data', None))
  t = jax.random.randint(rng_t, (raw_tokens.shape[0],), 1, cfg.model.T + 1)
  xt = noise_schedule.q_sample(raw_tokens, t, cfg.model.mask_token_id, rng_mask)
  t = jax.device_put(
    t,
    P(),
  )
  xt = jax.device_put(
    xt,
    P(),
  )
  batch_info = {}
  batch = {
    'input_ids': xt,
    'timesteps': t,
    'labels': x0,
    'training': training,
  }

  if training:
    batch_tokens = x0.shape[0] * x0.shape[1]
    mask_pct = xt == cfg.model.mask_token_id

    batch_info['batch_tokens'] = batch_tokens
    batch_info['mask_pct_mean'] = round(float(mask_pct.mean()), 2)
    batch_info['mask_pct_max'] = round(float(mask_pct.mean(axis=-1).max()), 2)
    batch_info['mask_pct_min'] = round(float(mask_pct.mean(axis=-1).min()), 2)
    batch_info['mask_pct_std'] = round(float(mask_pct.std()), 2)

  return batch, batch_info

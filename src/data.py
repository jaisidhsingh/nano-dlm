from __future__ import annotations

import typing as tp

import jax.numpy as jnp

from src.config import DataConfig


class Tokenizer:
    def __init__(self, tokenizer_type: str):
        import tiktoken

        self._tok = tiktoken.get_encoding("gtp2")

    def encode(self, text: str) -> jnp.ndarray:
        return jnp.array(self._tok(text), dtype=jnp.int64)

    def decode(self, tokens: jnp.ndarray) -> str:
        return self._tok.decode(tokens)

    def __call__(self, text: str) -> jnp.ndarray:
        return self.encode(text)


def get_dataloaders(
    cfg: DataConfig, batch_size: int, validate: bool = True
) -> tp.Union[tp.Tuple, tp.Iterable]:
    from datasets import load_from_disk
    from grain import MapDataset

    train_source = load_from_disk(cfg.trainset_path)
    if validate:
        val_source = load_from_disk(cfg.validset_path)

    collate_fn = None

    def owt_nn_gpt2_collate_fn(sample: dict) -> jax.Array:
        return jnp.array(sample["tokens"], dtype=jnp.int64)

    if "gpt2" in cfg.trainset_path:
        collate_fn = owt_nn_gpt2_collate_fn

    train_dataset = (
        MapDataset.source(train_source)
        .shuffle(cfg.shuffle_seed)
        .map(collate_fn)
        .to_iter_dataset()
        .batch(batch_size)
    )

    if validate:
        val_dataset = (
            MapDataset.source(val_source)
            .shuffle(cfg.shuffle_seed)
            .map(collate_fn)
            .to_iter_dataset()
            .batch(batch_size)
        )

    train_loader = iter(train_dataset)
    if validate:
        val_loader = iter(val_dataset)
        return train_loader, val_loader
    return train_loader

"""
data.py — Dataset and batching utilities for nano-dlm.

Supports three dataset modes:
  wikitext103  — auto-downloaded via HuggingFace datasets
  text8        — raw character-level text
  custom       — any plain .txt file you supply via --data.data_path

All text is tokenised with the GPT-2 BPE tokenizer (tiktoken), packed into
non-overlapping windows of length seq_len, and served as numpy arrays.

The DataLoader is a simple iterator that:
  1. Shuffles shard order each epoch.
  2. Yields batches as numpy arrays (host side); JAX handles device placement.
  3. Supports deterministic seeding for reproducibility.

Multi-device note
─────────────────
The DataLoader strips off batches of size  (n_devices × local_batch_size).
The train loop is responsible for reshaping to (n_devices, local_bs, L) before
calling pmap.  This keeps data.py free of device-specific logic.
"""

from __future__ import annotations
import os
import math
import numpy as np
from pathlib import Path
from typing import Generator, Optional
from config import DataConfig


# ---------------------------------------------------------------------------
# Tokeniser helpers (tiktoken, GPT-2 BPE)
# ---------------------------------------------------------------------------

def get_tokenizer():
    """Return a tiktoken GPT-2 encoder (cached after first call)."""
    import tiktoken
    return tiktoken.get_encoding("gpt2")


def encode(text: str, enc=None) -> np.ndarray:
    """Encode a string to a 1-D numpy array of int32 token ids."""
    if enc is None:
        enc = get_tokenizer()
    return np.array(enc.encode(text), dtype=np.int32)


# ---------------------------------------------------------------------------
# Dataset loaders  (return list of 1-D np.ndarray shards)
# ---------------------------------------------------------------------------

def _load_wikitext103(cache_dir: str) -> str:
    """Load WikiText-103 via HuggingFace datasets.  Returns raw text."""
    from datasets import load_dataset
    cache = Path(cache_dir) / "wikitext103"
    raw_file = cache / "all.txt"
    if raw_file.exists():
        return raw_file.read_text()
    cache.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    text = "\n".join(ds["text"])
    raw_file.write_text(text)
    return text


def _load_text8(cache_dir: str) -> str:
    """Load text8. Expects text8.zip in cache_dir or downloads it."""
    import urllib.request, zipfile
    cache = Path(cache_dir) / "text8"
    raw_file = cache / "text8"
    if raw_file.exists():
        return raw_file.read_text()
    cache.mkdir(parents=True, exist_ok=True)
    url = "http://mattmahoney.net/dc/text8.zip"
    zip_path = cache / "text8.zip"
    print(f"Downloading text8 from {url} …")
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(cache)
    return raw_file.read_text()


def _load_custom(data_path: str) -> str:
    p = Path(data_path)
    if not p.exists():
        raise FileNotFoundError(f"Custom data file not found: {data_path}")
    return p.read_text()


# ---------------------------------------------------------------------------
# Packing  (chunk text into non-overlapping windows)
# ---------------------------------------------------------------------------

def pack_tokens(tokens: np.ndarray, seq_len: int) -> np.ndarray:
    """
    Truncate `tokens` to a multiple of seq_len, then reshape to
    (N, seq_len).  Returns an np.ndarray of dtype int32.
    """
    n_full = len(tokens) // seq_len
    tokens = tokens[: n_full * seq_len]
    return tokens.reshape(n_full, seq_len)


# ---------------------------------------------------------------------------
# Top-level function: build packed token dataset
# ---------------------------------------------------------------------------

def build_dataset(cfg: DataConfig, split: str = "train") -> np.ndarray:
    """
    Returns a 2-D array of shape (N, seq_len) with packed token ids.

    split: "train" | "val"  (val uses the last 5% of examples)
    """
    enc = get_tokenizer()

    if cfg.dataset == "wikitext103":
        text = _load_wikitext103(cfg.cache_dir)
    elif cfg.dataset == "text8":
        text = _load_text8(cfg.cache_dir)
    elif cfg.dataset == "custom":
        text = _load_custom(cfg.data_path)
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")

    # Encode once; cache to disk as a memory-mapped array for large corpora
    cache_path = Path(cfg.cache_dir) / f"{cfg.dataset}_tokens_{cfg.seq_len}.npy"
    if not cache_path.exists():
        print("Tokenising dataset (this may take a while)…")
        tokens = encode(text, enc)
        packed = pack_tokens(tokens, cfg.seq_len)
        np.save(cache_path, packed)
    else:
        packed = np.load(cache_path, mmap_mode="r")  # type: ignore[assignment]

    # Train / validation split  (90/10)
    n_val = max(1, int(len(packed) * 0.1))
    if split == "val":
        return packed[-n_val:]
    return packed[:-n_val]


# ---------------------------------------------------------------------------
# DataLoader  (simple numpy iterator; no torch dependency)
# ---------------------------------------------------------------------------

class DataLoader:
    """
    Infinite iterator that yields batches of shape (batch_size, seq_len).

    Parameters
    ----------
    dataset   : 2-D np.ndarray of shape (N, seq_len)
    batch_size: global batch size (will be split across devices by train.py)
    seed      : RNG seed for shuffling
    """

    def __init__(
        self,
        dataset: np.ndarray,
        batch_size: int,
        seed: int = 0,
        drop_last: bool = True,
    ) -> None:
        self.dataset    = dataset
        self.batch_size = batch_size
        self.drop_last  = drop_last
        self.rng        = np.random.default_rng(seed)
        self._n         = len(dataset)
        self._idx       = self._new_order()

    def _new_order(self) -> np.ndarray:
        return self.rng.permutation(self._n)

    def __iter__(self) -> Generator[np.ndarray, None, None]:
        pos = 0
        while True:
            if pos + self.batch_size > len(self._idx):
                # Reshuffle and wrap
                self._idx = self._new_order()
                pos = 0
            idx_batch = self._idx[pos : pos + self.batch_size]
            pos += self.batch_size
            yield self.dataset[idx_batch]   # (batch_size, seq_len)

    def __next__(self) -> np.ndarray:
        return next(iter(self))


# ---------------------------------------------------------------------------
# Convenience function used by train.py
# ---------------------------------------------------------------------------

def make_loaders(cfg: DataConfig, batch_size: int, seed: int = 0):
    """Build train and validation DataLoaders."""
    train_ds = build_dataset(cfg, split="train")
    val_ds   = build_dataset(cfg, split="val")
    print(f"[data] train examples: {len(train_ds):,}   val examples: {len(val_ds):,}")
    train_loader = DataLoader(train_ds, batch_size, seed=seed)
    val_loader   = DataLoader(val_ds,   batch_size, seed=seed + 1)
    return train_loader, val_loader

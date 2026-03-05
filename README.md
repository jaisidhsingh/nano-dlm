# nano-dlm 🧬

> A minimalist, extensible JAX implementation of **diffusion language models** —
> the spiritual successor to [nanoGPT](https://github.com/karpathy/nanoGPT) for the diffusion era.

Implements **absorbing / masked diffusion** (MDLM / D3PM framework):
each training step randomly masks tokens according to a noise schedule and trains
a bidirectional Transformer to predict the original tokens.

---

## File Map  (exactly 5 source files)

| File | Role |
|---|---|
| [`config.py`](config.py) | All dataclass configs + `tyro` CLI wiring |
| [`schedules.py`](schedules.py) | Cosine / linear / sqrt noise schedules; forward & reverse diffusion math |
| [`model.py`](model.py) | Bidirectional Transformer denoiser (Equinox); AdaLN, RoPE, predict-x₀ |
| [`data.py`](data.py) | tiktoken BPE tokeniser, packing into fixed windows, infinite `DataLoader` |
| [`train.py`](train.py) | Training loop, `jax.pmap` multi-device, AdamW, checkpointing, DDIM sampler |

---

## Installation

```bash
pip install jax jaxlib flax optax tyro tiktoken datasets numpy
```

For GPU / TPU JAX builds follow the [official JAX install guide](https://github.com/google/jax#installation).

---

## Quick Start

```bash
# Single device — train on WikiText-103 with defaults (6-layer, 512-dim model)
python train.py

# See every available flag
python train.py --help

# Medium model on WikiText-103
python train.py \
  --model.n_layers 12 --model.d_model 768 --model.n_heads 12 \
  --train.lr 1e-4 --train.max_steps 500000 --train.batch_size 256 \
  --schedule.kind cosine \
  --data.dataset wikitext103 \
  --train.out_dir runs/mdlm-medium

# Custom plain-text corpus
python train.py \
  --data.dataset custom --data.data_path /path/to/corpus.txt \
  --train.out_dir runs/custom

# text8 character dataset
python train.py --data.dataset text8 --model.vocab_size 27
```

Training logs are written as JSONL to `<out_dir>/log.jsonl`:
```json
{"step": 0,    "loss": 5.4321, "lr": 0.0}
{"step": 50,   "loss": 4.2107, "lr": 7.5e-06}
{"step": 1000, "val_loss": 3.9842}
```

---

## Architecture

```
xₜ (noisy tokens, B × L)   t (step integer, B)
        │                           │
        ▼                           ▼
  TokenEmbed(V→D)         SinEmbed(D) → Linear → SiLU → Linear  ← time_emb (B × 4D)
        │                           │
        └───────────┬───────────────┘
                    ▼
        ┌───────────────────────┐
        │  × N TransformerBlock │
        │  ┌─────────────────┐  │
        │  │ AdaLN(x, t_emb) │  │   ← adaptive scale+shift from t
        │  │ MHA  + RoPE     │  │   ← bidirectional (no causal mask)
        │  │ AdaLN(x, t_emb) │  │
        │  │ FFN  (GELU)     │  │
        │  └─────────────────┘  │
        └───────────────────────┘
                    │
              LayerNorm
              LM Head (D→V)
                    │
          logits  p_θ(x₀ | xₜ, t)          shape (B, L, V)
```

**Parameterisation:** the model predicts **x₀ directly** (not the noise).
Loss = weighted cross-entropy at masked positions only.

---

## How Diffusion Works Here

1. **Forward process** `q(xₜ | x₀)` — each token is replaced by `[MASK]` independently
   with probability `1 - ᾱₜ`, where `ᾱₜ` follows the chosen schedule.
2. **Training** — given `(xₜ, t)`, the model predicts logits for the original tokens.
   Loss is MDLM-weighted cross-entropy over masked positions:
   `L = -E[λₜ · Σᵢ 1[xₜᵢ=[M]] · log p_θ(x₀ᵢ | xₜ, t)]`
3. **Sampling** — start fully masked `xₜ`, iteratively denoise via DDIM-style
   ancestral steps using the predicted `x̂₀`.

---

## Noise Schedules

| Flag value | Formula | Notes |
|---|---|---|
| `cosine` | `cos²((t/T + 0.008) / 1.008 · π/2)` | Smooth, well-tested (Nichol & Dhariwal 2021) |
| `linear` | `1 − t/T` | Simplest baseline |
| `sqrt`   | `1 − √(t/T)` | Recommended by MDLM (Shi et al. 2024) |

```bash
python train.py --schedule.kind sqrt --schedule.T 1000
```

---

## Full Config Reference

### `--model.*`

| Flag | Default | Description |
|---|---|---|
| `vocab_size` | `50257` | GPT-2 BPE vocabulary size |
| `seq_len` | `128` | Sequence length |
| `n_layers` | `6` | Number of Transformer blocks |
| `n_heads` | `8` | Attention heads |
| `d_model` | `512` | Embedding / hidden dimension |
| `d_ff` | `2048` | FFN hidden dim (0 → 4×d_model) |
| `dropout` | `0.1` | Dropout in attention & FFN |
| `mask_token_id` | `50256` | Token ID used as `[MASK]` |

### `--schedule.*`

| Flag | Default | Description |
|---|---|---|
| `kind` | `cosine` | Schedule family: `cosine`, `linear`, `sqrt` |
| `T` | `1000` | Total diffusion steps |
| `eps` | `1e-4` | Clamp margin to avoid degenerate rates |

### `--data.*`

| Flag | Default | Description |
|---|---|---|
| `dataset` | `wikitext103` | `wikitext103`, `text8`, or `custom` |
| `data_path` | `""` | Path to `.txt` file when `dataset=custom` |
| `seq_len` | `128` | Must match `model.seq_len` |
| `num_workers` | `4` | CPU workers for data loading |
| `cache_dir` | `.cache` | Where to store tokenised shards |

### `--train.*`

| Flag | Default | Description |
|---|---|---|
| `seed` | `42` | Global PRNG seed |
| `batch_size` | `256` | Global batch (split across devices) |
| `grad_accum_steps` | `1` | Gradient accumulation |
| `max_steps` | `100000` | Total optimiser steps |
| `warmup_steps` | `2000` | Linear LR warm-up steps |
| `lr` | `3e-4` | Peak learning rate |
| `min_lr` | `1e-5` | Cosine decay floor |
| `weight_decay` | `0.1` | AdamW weight decay |
| `beta1` / `beta2` | `0.9` / `0.98` | AdamW moments |
| `clip_grad_norm` | `1.0` | Gradient norm clipping (0 = off) |
| `log_every` | `50` | Steps between log lines |
| `eval_every` | `1000` | Steps between validation runs |
| `save_every` | `5000` | Steps between checkpoints |
| `out_dir` | `runs/default` | Output directory |
| `resume` | `""` | Checkpoint dir to resume from |

---

## Multi-device Training

`jax.pmap` shards the batch across all visible devices automatically.
No code changes required — just make sure `batch_size` is divisible by device count:

```bash
# 8 GPUs / 8 TPUs
python train.py --train.batch_size 512   # 64 per device

# Check detected devices
python -c "import jax; print(jax.devices())"
```

Gradients are averaged across devices with `jax.lax.pmean` inside the pmap'd step.

---

## Checkpointing & Resuming

Checkpoints are saved to `<out_dir>/step_XXXXXXX/` automatically every `--train.save_every` steps.
The most recent checkpoint path is written to `<out_dir>/latest`.

```bash
# Resume a run from the last saved checkpoint
python train.py \
  --train.resume runs/mdlm-medium/step_0050000 \
  --train.out_dir runs/mdlm-medium
```

Each checkpoint directory contains:
```
step_0050000/
  model.pkl       # serialised Equinox parameter arrays
  config.json     # full config snapshot for reproducibility
```

---

## Sampling / Generation

```python
import pickle, jax, tiktoken
import equinox as eqx
from config import Config
from model import DiffusionTransformer
from schedules import make_schedule
from train import sample

# Build model with the same config used for training
cfg   = Config()   # or load from checkpoint's config.json
sched = make_schedule(cfg.schedule)
model = DiffusionTransformer(cfg.model, jax.random.PRNGKey(0))

# Load trained weights
with open("runs/mdlm-medium/step_0050000/model.pkl", "rb") as f:
    params = pickle.load(f)
model = eqx.tree_at(lambda m: eqx.filter(m, eqx.is_array), model, params)

# Generate — starts fully masked, runs 50 DDIM reverse steps
tokens = sample(
    model, sched,
    n_samples=8, seq_len=128,
    rng=jax.random.PRNGKey(42),
    n_steps=50,
    temperature=1.0,
)  # shape (8, 128) int32

# Decode with tiktoken
enc = tiktoken.get_encoding("gpt2")
for row in tokens:
    print(enc.decode(row.tolist()))
```

**Temperature:** `< 1.0` = sharper / more repetitive, `> 1.0` = more diverse/noisy.

---

## Training Tips

| Goal | Suggestion |
|---|---|
| Faster convergence | Use `--schedule.kind sqrt` (MDLM recommendation) |
| Larger model | Increase `n_layers`, `d_model`; lower `lr` to `1e-4` |
| Reduce overfitting | Increase `dropout`, add more data |
| Long sequences | Increase `seq_len` and `batch_size` proportionally |
| Stable training | Keep `beta2=0.98`, `clip_grad_norm=1.0` |
| Debugging NaNs | Run with `--model.dropout 0.0` and `--train.lr 1e-5` |

---

## Extending nano-dlm

- **New schedule** → add a function in `schedules.py` returning `(T+1,)` `alpha_bar`, register it in `make_schedule`.
- **New architecture** → swap the `DiffusionTransformer` in `model.py`; keep the same `(xt, t, T) → logits` signature.
- **New dataset** → add a loader in `data.py` returning a raw string, handle it in `build_dataset`.
- **Different loss** → modify `compute_loss` in `train.py` (e.g. remove MDLM re-weighting, add auxiliary losses).
- **Classifier-free guidance** → condition `token_emb` on a class label and randomly drop it during training.

---

## References

- [D3PM: Structured Denoising Diffusion Models in Discrete State-Spaces](https://arxiv.org/abs/2107.03006) — Austin et al. 2021
- [Improved Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672) — Nichol & Dhariwal 2021
- [MDLM: Masked Diffusion Language Models](https://arxiv.org/abs/2406.07524) — Shi et al. 2024
- [nanoGPT](https://github.com/karpathy/nanoGPT) — Karpathy (inspiration)

---

## Dependencies

```
jax / jaxlib   — array library + JIT/pmap
equinox        — Pytree-based neural net modules
optax          — gradient transformations & schedules
tyro           — dataclass → CLI
tiktoken       — GPT-2 BPE tokenizer
datasets       — HuggingFace (for wikitext-103; optional)
numpy          — host-side data loading
```

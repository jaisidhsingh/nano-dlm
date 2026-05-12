# nano-dlm 🧬

> A minimalist, extensible JAX implementation of **diffusion language models** —
> the spiritual successor to [nanoGPT](https://github.com/karpathy/nanoGPT) for the diffusion era.

Implements **masked diffusion** (MDLM) process where each training step randomly masks tokens according to a noise schedule. A bidirectional Transformer learns to predict the original tokens, i.e., the model unmasks tokens at each timestep.

## Installation

```bash
pip install jax jaxlib flax optax tyro tiktoken datasets orbax
```

## Quick Start

```bash
# Single device — train on WikiText-103 with defaults (6-layer, 512-dim model)
python train.py

# See every available flag
python train.py --help

python train.py \
# Control every setting via heirarchical cli args
  --model.n_layers 12 --model.d_model 768 --model.n_heads 12 \
  --train.lr 1e-4 --train.max_steps 500000 --train.batch_size 256 \
  --schedule.kind cosine \
  --train.out_dir runs/mdlm-medium
```

## Architecture

**Parameterisation:** the model predicts **x₀ directly** (not the noise).
Loss = weighted cross-entropy at masked positions only.

1. **Forward process** `q(xₜ | x₀)` — each token is replaced by `[MASK]` independently
   with probability `1 - ᾱₜ`, where `ᾱₜ` follows the chosen schedule.
2. **Training** — given `(xₜ, t)`, the model predicts logits for the original tokens.
   Loss is MDLM-weighted cross-entropy over masked positions:
   `L = -E[λₜ · Σᵢ 1[xₜᵢ=[M]] · log p_θ(x₀ᵢ | xₜ, t)]`
3. **Sampling** — start fully masked `xₜ`, iteratively denoise via DDIM-style
   ancestral steps using the predicted `x̂₀`.


## Noise Schedules

| Flag value | Formula | Notes |
|---|---|---|
| `cosine` | `cos²((t/T + 0.008) / 1.008 · π/2)` | Smooth, well-tested (Nichol & Dhariwal 2021) |
| `linear` | `1 − t/T` | Simplest baseline |
| `sqrt`   | `1 − √(t/T)` | Recommended by MDLM (Shi et al. 2024) |

```bash
python train.py --schedule.kind sqrt --schedule.T 1000
```

## Multi-device Training

We create a DDP (FSDP) configuration over however many GPUs (single node) you have available via `jax.make_mesh((jax.device_count(), ), ("data",))` that shards the batch across all visible devices automatically.
No code changes required — just make sure `batch_size` is divisible by device count:

```bash
python -c "import jax; print(jax.devices())";
python -m experiments.train_multi_gpu
```

## Checkpointing & Resuming

Every few steps, controllable via the `--exp.save_every` cli arg, we use `orbax` to checkpoint the model and optimizer states. Alongside, the logs upto that step and the full config is saved in `logs.json` and `config.json` respectively.

Each checkpoint directory contains:
```
step_0050000/
  logs.json       # logs upto step 0050000
  model.pkl       # serialised Equinox parameter arrays
  config.json     # full config snapshot for reproducibility
```

## References

- [D3PM: Structured Denoising Diffusion Models in Discrete State-Spaces](https://arxiv.org/abs/2107.03006) — Austin et al. 2021
- [Improved Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672) — Nichol & Dhariwal 2021
- [MDLM: Masked Diffusion Language Models](https://arxiv.org/abs/2406.07524) — Shi et al. 2024
- [nanoGPT](https://github.com/karpathy/nanoGPT) — Karpathy (inspiration)

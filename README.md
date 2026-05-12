<div align="center">
# `nano-dlm` 🧬
</div>

> A minimalist, extensible JAX implementation of **diffusion language models** —
> the spiritual successor to [nanoGPT](https://github.com/karpathy/nanoGPT) for the diffusion era.

Implements **masked diffusion** (MDLM) process where each training step randomly masks tokens according to a noise schedule. A bidirectional Transformer learns to predict the original tokens, i.e., the model unmasks tokens at each timestep.


## Installation

```bash
pip install jax jaxlib flax optax tyro tiktoken datasets orbax
```

## Data
We train the diffusion language model on a pre-tokenized subset of OpenWebText, very conveniently provided by Neel Nanda on huggingface. You can download and use the dataset easily by

```python
from datasets import load_dataset, load_from_disk

dataset = load_dataset("NeelNanda/openwebtext-tokenized-9b", split="train")
dataset.save_to_disk("your/save/path") # if you want to save to a specific location

# then load it back in from the saved path
dataset = load_from_disk("your/save/path")
```

You can execute the download, saving, and splitting (train-val) of this dataset very easily by running

```bash
bash scripts/prepare_data.sh
```

Remember to specify the output path (where the datasets will be saved) in the script, and in `DataConfig` in `src/config.py`. Once that's done, you're all set!


## Quick Start: Single GPU Training
Launch a training run by simply running

```bash
# Single device training with defaults (6-layer, 512-dim model)
python -m experiments.train_single_gpu

# Control every setting via heirarchical cli args
python -m experiments.train_single_gpu \
  --model.init_seed 123 \
  --model.n_layers 6 \
  --model.d_model 512 \
  --model.n_heads 8 \
  --data.shuffle_seed 123 \
  --train.seed 123 \
  --train.lr 1e-3 \
  --train.weight_decay 0.1 \
  --train.max_steps 10000 \
  --train.batch_size 32 \
  --train.grad_acc_steps 8 \
  --schedule.kind cosine \
  --exp.run_name "single_gpu_dlm_run" \
  --exp.use_wandb True \
  --exp.project_name "nano-dlm"

# See every available flag
python -m experiments.train_single_gpu --help
```


## Multi-GPU Training
We create a DDP (FSDP) configuration over however many GPUs (single node) you have available via `jax.make_mesh((jax.device_count(), ), ("data",))` that shards the batch across all visible devices automatically.
No code changes required — just make sure `batch_size` is divisible by device count:

```bash
python -c "import jax; print(jax.devices())";
python -m experiments.train_multi_gpu # same cli-args as `experiments.train_single_gpu.py`
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

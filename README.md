<div align="center">

# `nano-dlm` 🧬

<a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT%202.0-green.svg?style=for-the-badge" alt="License"></a>

> A clean & extensible **_JAX_** implementation of **diffusion language models** —
> [nanoGPT](https://github.com/karpathy/nanoGPT) for the diffusion era.

</div>

Implements **masked/absorbing diffusion** (MDLM) process where each training step randomly masks tokens according to a noise schedule. A bidirectional Transformer learns to predict the original tokens, i.e., the model unmasks tokens at each timestep. Uniform diffusion is coming soon!

## 💡 Why JAX

1. JAX's "pure function" nature composes elegantly with `q_sample` for diffusion training.
2. `jax.jit` and `nnx.jit` compile the entire computation graph via XLA, which means better kernel fusion and more predictable performance. This makes them _much_ stronger than `torch.compile`, which is still only a tracing-based partial compiler.
3. Explicit `PRNG` splitting for randomness management => better reproducibility

We stick purely to the JAX ecosystem, even for data fetching and loading, via the new [`grain`](https://google-grain.readthedocs.io/en/latest/) package.

## 🛠️ Installation

We need the following packages for this repository, that we recommend be installed in a dedicated `conda` environment.

```bash
conda create -n nano-dlm python=3.12

# installs latest package versions available for your system
pip install jax jaxlib flax optax tyro tiktoken datasets orbax grain

# for version specificity
pip install -r requirements.txt
```

On the other hand, you can also use `uv`

```bash
uv init .
source .venv/bin/activate
uv add jax jaxlib flax optax tyro tiktoken datasets orbax
uv sync
```

## 📀 Data

We train the diffusion language model on a pre-tokenized subset of `OpenWebText`, very conveniently provided by Neel Nanda on huggingface. You can download and use the dataset easily by

```python
from datasets import load_dataset, load_from_disk

dataset = load_dataset("NeelNanda/openwebtext-tokenized-9b", split="train")
dataset.save_to_disk("your/save/path") # if you want to save to a specific location

# then load it back in from the saved path
dataset = load_from_disk("your/save/path")
```

Specific information on how the dataset is used can be found in `src/data.py/` and `src/config.py`. Remember to split the dataset into `train` and `val` splits. In our experiments, we use 1M tokens for validation.

## ⚡️ Quick Start: Training

Single-GPU and multi-GPU use the same script. JAX auto-discovers all visible devices and shards the batch accordingly — no launcher, no code changes.

```bash
# See available devices
python -c "import jax; print(jax.devices())"

# Single-GPU or multi-GPU — same command
python train.py

# Control every setting via hierarchical CLI args
python train.py \
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
  --exp.run_name "dlm_run" \
  --exp.use_wandb True \
  --exp.project_name "nano-dlm"

# See every available flag
python train.py --help
```

On multi-GPU the batch is sharded along the data axis via `jax.make_mesh` (pure data parallelism). On a single GPU the mesh degrades to one device with zero overhead.

## 🔍 Architecture

We provide the option for timestep-conditioning, although the default configuration has it switched off, following the modern implementations of diffusion language model. A brief overview of the architecture and diffusion process is given as follows.

**Parameterisation:** the model predicts **x₀ directly** (not the noise).
Loss = weighted cross-entropy at masked positions only.

1. **Forward process** `q(xₜ | x₀)` — each token is replaced by `[MASK]` independently
   with probability `1 - ᾱₜ`, where `ᾱₜ` follows the chosen schedule.
2. **Training** — given `(xₜ, t)`, the model predicts logits for the original tokens.
   Loss is MDLM-weighted cross-entropy over masked positions:
   `L = -E[λₜ · Σᵢ 1[xₜᵢ=[M]] · log p_θ(x₀ᵢ | xₜ, t)]`
3. **Sampling** — start fully masked `xₜ`, iteratively denoise via DDIM-style
   ancestral steps using the predicted `x̂₀`.

## ⏳ Noise Schedules

| Flag value | Formula                             | Notes                                        |
| ---------- | ----------------------------------- | -------------------------------------------- |
| `cosine`   | `cos²((t/T + 0.008) / 1.008 · π/2)` | Smooth, well-tested (Nichol & Dhariwal 2021) |
| `linear`   | `1 − t/T`                           | Simplest baseline                            |
| `sqrt`     | `1 − √(t/T)`                        | Recommended by MDLM (Shi et al. 2024)        |

```bash
python train.py --schedule.kind sqrt --schedule.T 1000
```

## 🔄 Checkpointing

Every few steps, controllable via the `--exp.save_every` cli arg, we use `orbax` to checkpoint the model and optimizer states. Alongside, the logs upto that step and the full config is saved in `logs.json` and `config.json` respectively. To resume from say step 100 from the example checkpoint below, set `--exp.resume=True` and provide the folder path `nano-dlm-checkpoints/step_100` to `--exp.resume_path`.

```plaintext
nano-dlm-checkpoints/
└── step_100/
    ├── model_state/
    ├── optimizer_state/
    ├── logs.json
    └── config.json
```

## 🎓 Citation

If you found this work useful, please cite it as follows.

```bibtex
@software{singh2026nanodlm,
  author       = {Singh, Jaisidh},
  title        = {nano-dlm: A Minimal JAX Implementation of Diffusion Language Models},
  year         = {2026},
  publisher    = {GitHub},
  url          = {https://github.com/jaisidhsingh/nano-dlm}
}
```

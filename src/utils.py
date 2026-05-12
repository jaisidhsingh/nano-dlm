import os
import pickle

import flax.nnx as nnx
import wandb

from src.config import ExperimentConfig


def log_metrics(cfg: ExperimentConfig, per_step_logs: dict, step: int) -> None:
    if cfg.use_wandb:
        """
        We want to log
        - step
        - micro_step
        - lr
        - tokens
        - train_loss
        - train_loss_batch_avged
        - train_ppl
        - val_loss
        - val_ppl
        - avg_tokens_masked_in_batch
        """
        wandb.log(per_step_logs, step=step)


def save_checkpoint(out_dir: str, step: int, optimizer: nnx.Optimizer, cfg: Config):
    ckpt_dir = os.path.join(out_dir, f"step_{step:07d}")
    os.makedirs(ckpt_dir, exist_ok=True)
    param_state = nnx.state(optimizer.model, nnx.Param)
    with open(os.path.join(ckpt_dir, "model.pkl"), "wb") as f:
        pickle.dump(param_state, f)
    import dataclasses

    with open(os.path.join("config.json"), "w") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=2)
    with open(os.path.join(out_dir, "latest"), "w") as f:
        f.write(str(ckpt_dir))
    print(f"[ckpt] Saved → {ckpt_dir}")


def load_checkpoint(ckpt_path: str, model: DiffusionTransformer):
    with open(os.path.join(ckpt_path, "model.pkl"), "rb") as f:
        saved = pickle.load(f)
    nnx.update(model, saved)
    print(f"[ckpt] Loaded ← {ckpt_path}")


@nnx.jit
def sample(
    model: DiffusionTransformer,
    schedule: NoiseSchedule,
    n_samples: int,
    seq_len: int,
    rng: jax.Array,
    temperature: float = 1.0,
    n_steps: int = 50,
) -> jax.Array:
    """Generate n_samples sequences via iterative denoising.  Returns (N, L) int32."""
    mask_id = model.cfg.mask_token_id
    T = schedule.T

    x = jnp.full((n_samples, seq_len), mask_id, dtype=jnp.int32)
    steps = jnp.linspace(T, 0, n_steps + 1, dtype=jnp.int32)

    def body(carry, i):
        x, rng = carry
        t_cur = jnp.full((n_samples,), steps[i], dtype=jnp.int32)
        t_next = jnp.full((n_samples,), steps[i + 1], dtype=jnp.int32)
        rng, rng_step = jax.random.split(rng)
        logits = model(x, t_cur, T, training=False)
        x = schedule.ddim_step(logits, x, t_cur, t_next, mask_id, rng_step, temperature)
        return (x, rng), None

    (x, _), _ = jax.lax.scan(body, (x, rng), jnp.arange(n_steps))
    return x


# def evaluate(rep_state, val_loader, schedule, n_devices, n_batches=20):
#     losses = []
#     ab = schedule.alpha_bar
#     T_arr = jnp.array(schedule.T)
#     for _, batch_np in zip(range(n_batches), val_loader):
#         local_bs = batch_np.shape[0] // n_devices
#         batch_jax = jnp.array(batch_np).reshape(n_devices, local_bs, -1)
#         rng = jax.random.split(jax.random.PRNGKey(0), n_devices)
#         loss = _eval_step(
#             rep_state,
#             batch_jax,
#             jnp.broadcast_to(ab, (n_devices,) + ab.shape),
#             jnp.broadcast_to(T_arr, (n_devices,)),
#             rng,
#         )
#         losses.append(float(loss[0]))
#     return float(np.mean(losses))

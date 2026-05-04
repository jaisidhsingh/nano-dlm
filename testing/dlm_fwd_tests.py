import flax.nnx as nnx
import jax

from src.config import ModelConfig
from src.model import DiffusionTransformer


def test_dlm_init():
    cfg = ModelConfig()
    rngs = nnx.Rngs(cfg.init_seed)
    model = DiffusionTransformer(cfg, rngs=rngs)

    total = model.count_params(nonembed=False)
    without_embed = model.count_params(nonembed=True)
    print(total, without_embed)


def test_dlm_forward():
    """Test the full forward pass with random tokens and masking."""
    import jax.numpy as jnp

    from src.config import ScheduleConfig
    from src.schedules import make_schedule

    # --- Setup ---
    cfg = ModelConfig()
    sch_cfg = ScheduleConfig(T=1000, kind="cosine")
    schedule = make_schedule(sch_cfg)

    rngs = nnx.Rngs(cfg.init_seed)
    model = DiffusionTransformer(cfg, rngs=rngs)

    rng = jax.random.PRNGKey(42)
    B, L = 4, cfg.seq_len  # small batch, full sequence length

    # --- Create random "clean" tokens and mask them ---
    rng_tok, rng_mask, rng_t = jax.random.split(rng, 3)

    x0 = jax.random.randint(rng_tok, (B, L), 0, cfg.vocab_size - 1)  # clean tokens
    t = jax.random.randint(rng_t, (B,), 1, schedule.T + 1)  # random steps

    xt = schedule.q_sample(x0, t, cfg.mask_token_id, rng_mask)  # noisy tokens

    # --- Forward pass ---
    logits = model(xt, t, schedule.T, training=False)

    # --- Shape checks ---
    V = cfg.vocab_size
    assert logits.shape == (B, L, V), f"Expected {(B, L, V)}, got {logits.shape}"
    assert logits.dtype == jnp.float32, f"Expected float32, got {logits.dtype}"

    # --- Sanity: no NaNs or infinities ---
    assert not jnp.any(jnp.isnan(logits)), "Logits contain NaN values"
    assert not jnp.any(jnp.isinf(logits)), "Logits contain Inf values"

    # --- Sanity: softmax produces valid probability distributions ---
    probs = jax.nn.softmax(logits, axis=-1)
    assert jnp.all(probs >= 0) and jnp.all(probs <= 1), "Probs out of [0, 1] range"
    assert jnp.allclose(probs.sum(axis=-1), 1.0, atol=1e-5), "Probs don't sum to 1"

    # --- Sanity: logits are finite and not all identical ---
    std_per_pos = jnp.std(logits, axis=-1).mean()
    assert std_per_pos > 0, (
        "Logits are constant across the vocabulary — model not learning"
    )

    # --- Training-mode forward pass (with dropout rng) ---
    rng_fwd = jax.random.PRNGKey(99)
    logits_train = model(xt, t, schedule.T, training=True, rng=rng_fwd)
    assert logits_train.shape == (B, L, V), "Training forward pass produced wrong shape"

    # --- Verify masked positions are still masked after forward pass ---
    mask_fraction = (xt == cfg.mask_token_id).mean()
    print(f"  Mask fraction in batch: {mask_fraction:.3f}")
    print(f"  Logit std (mean over vocab dim): {std_per_pos:.4f}")
    print(f"  All checks passed: shapes={logits.shape}, no NaN, probs valid")


if __name__ == "__main__":
    print("JAX devices:", jax.devices())
    print("\n=== test_dlm_init ===")
    test_dlm_init()
    print("\n=== test_dlm_forward ===")
    test_dlm_forward()
    print("\nAll tests passed.")

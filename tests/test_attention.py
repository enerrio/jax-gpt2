import jax
import equinox as eqx
import jax.numpy as jnp
from gpt.model import (
    MultiHeadedAttention,
    MLP,
    TransformerBlock,
)


def test_multi_headed_attention(cfg, x, key):
    batch_size = 10

    # Initialize custom model
    custom_mha = MultiHeadedAttention(cfg, key=key)

    # Initialize Equinox model
    equiv_mha = eqx.nn.MultiheadAttention(
        num_heads=cfg["n_heads"],
        query_size=cfg["emb_dim"],
        output_size=cfg["emb_dim"],
        use_query_bias=cfg["qkv_bias"],
        use_key_bias=cfg["qkv_bias"],
        use_value_bias=cfg["qkv_bias"],
        dropout_p=cfg["drop_rate"],
        use_output_bias=True,
        key=key,
    )

    assert jnp.allclose(
        equiv_mha.query_proj.weight, custom_mha.W_q.weight
    ), "weights not equal"
    assert jnp.allclose(
        equiv_mha.query_proj.bias, custom_mha.W_q.bias
    ), "weights not equal"
    assert jnp.allclose(
        equiv_mha.key_proj.weight, custom_mha.W_k.weight
    ), "weights not equal"
    assert jnp.allclose(
        equiv_mha.key_proj.bias, custom_mha.W_k.bias
    ), "weights not equal"
    assert jnp.allclose(
        equiv_mha.value_proj.weight, custom_mha.W_v.weight
    ), "weights not equal"
    assert jnp.allclose(
        equiv_mha.value_proj.bias, custom_mha.W_v.bias
    ), "weights not equal"
    assert jnp.allclose(
        equiv_mha.output_proj.weight, custom_mha.out_proj.weight
    ), "weights not equal"
    assert jnp.allclose(
        equiv_mha.output_proj.bias, custom_mha.out_proj.bias
    ), "weights not equal"

    # Create causal mask
    mask = jnp.tril(
        jnp.ones(
            (batch_size, cfg["context_length"], cfg["context_length"]), dtype=jnp.bool
        )
    )
    # Run custom model
    custom_output = jax.vmap(custom_mha, in_axes=(0, None, None))(x, True, None)
    # Run Equinox model
    equiv_output = jax.vmap(equiv_mha)(x, x, x, mask)

    # Compare outputs
    assert jnp.allclose(custom_output, equiv_output), "Outputs are not close"
    print("Test passed: Custom MHA and Equinox MHA produce identical outputs.")


def test_mlp(cfg, x, key):
    # Initialize custom MLP
    custom_mlp = MLP(cfg, key)

    # Initialize Equinox's built-in MLP
    eqx_mlp = eqx.nn.MLP(
        in_size=cfg["emb_dim"],
        out_size=cfg["emb_dim"],
        width_size=cfg["emb_dim"] * 4,
        depth=1,
        activation=jax.nn.gelu,
        key=key,
    )

    # Verify weights and biases are the same
    assert jnp.allclose(
        eqx_mlp.layers[0].weight, custom_mlp.layers[0].weight
    ), "fc1 weights do not match"
    assert jnp.allclose(
        eqx_mlp.layers[0].bias, custom_mlp.layers[0].bias
    ), "fc1 biases do not match"
    assert jnp.allclose(
        eqx_mlp.layers[1].weight, custom_mlp.layers[2].weight
    ), "fc2 weights do not match"
    assert jnp.allclose(
        eqx_mlp.layers[1].bias, custom_mlp.layers[2].bias
    ), "fc2 biases do not match"

    # Apply custom MLP
    custom_output = jax.vmap(custom_mlp)(x)
    eqx_output = jax.vmap(jax.vmap(eqx_mlp))(x)  # vectorized over batch & seq dims

    # Compare outputs
    assert jnp.allclose(custom_output, eqx_output), "MLP outputs do not match"
    print("Test passed: Custom MLP and Equinox's MLP produce identical outputs.")

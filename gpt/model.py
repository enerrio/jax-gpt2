from typing import Optional
import einops
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PyTree, Int, Float, Bool, Key, PRNGKeyArray


def key_split_allowing_none(
    key: Optional[PRNGKeyArray],
) -> Key[Array, "2"] | tuple[None, None]:
    """Split key, if passed None then return original key."""
    if key is None:
        return key, None
    else:
        return jr.split(key)


def normal_init(
    key: PRNGKeyArray,
    shape: tuple[int, int],
    dtype: jnp.dtype,
    mean: float = 0.0,
    std: float = 0.02,
) -> Array:
    return mean + std * jr.normal(key, shape=shape, dtype=dtype)


def reinit_model_params(
    model: eqx.Module, dtype: jnp.dtype, key: PRNGKeyArray
) -> eqx.Module:
    """Custom weight initialization."""
    # Split keys for each category
    key_tok, key_pos, key_linear = jr.split(key, 3)

    # Token embedding: normal(0,0.02)
    model = eqx.tree_at(
        lambda m: m.shared.pytree[0].weight,
        model,
        normal_init(
            key=key_tok,
            shape=model.shared.pytree[0].weight.shape,
            dtype=dtype,
            mean=0.0,
            std=0.02,
        ),
    )

    # Positional embedding: normal(0,0.01)
    model = eqx.tree_at(
        lambda m: m.pos_embed,
        model,
        normal_init(
            key=key_pos, shape=model.pos_embed.shape, dtype=dtype, mean=0.0, std=0.01
        ),
    )

    # Linear layer weights: normal(0,0.02)
    is_linear = lambda x: isinstance(x, eqx.nn.Linear)
    linear_layers = [
        x for x in jax.tree.leaves(model, is_leaf=is_linear) if is_linear(x)
    ]
    w_shapes = [layer.weight.shape for layer in linear_layers[1:]]  # ignore shared node
    b_shapes = [layer.bias.shape for layer in linear_layers if layer.bias is not None]
    w_keys = jr.split(key_linear, len(w_shapes))
    new_weights = [
        normal_init(k, s, dtype, 0.0, 0.02) for k, s in zip(w_keys, w_shapes)
    ]
    # bias should be 0
    new_biases = [
        jnp.zeros(s, dtype=dtype) for s in b_shapes
    ]

    model = eqx.tree_at(
        lambda m: [
            x.weight for x in jax.tree.leaves(m, is_leaf=is_linear) if is_linear(x)
        ][
            1:
        ],  # ignore shared node
        model,
        new_weights,
    )
    model = eqx.tree_at(
        lambda m: [
            x.bias
            for x in jax.tree.leaves(m, is_leaf=is_linear)
            if is_linear(x) and x.bias is not None
        ],
        model,
        new_biases,
    )
    return model


class MultiHeadedAttention(eqx.Module):
    W_q: eqx.nn.Linear
    W_k: eqx.nn.Linear
    W_v: eqx.nn.Linear
    out_proj: eqx.nn.Linear
    drop: eqx.nn.Dropout
    n_heads: int

    def __init__(self, cfg: dict[str, int | float | bool], key: PRNGKeyArray):
        key1, key2, key3, key4 = jr.split(key, 4)
        self.n_heads = cfg["n_heads"]
        assert (
            cfg["emb_dim"] % self.n_heads
        ) == 0, "Embedding dimension must be divisible by n_heads"
        self.W_q = eqx.nn.Linear(
            cfg["emb_dim"], cfg["emb_dim"], use_bias=cfg["qkv_bias"], key=key1
        )
        self.W_k = eqx.nn.Linear(
            cfg["emb_dim"], cfg["emb_dim"], use_bias=cfg["qkv_bias"], key=key2
        )
        self.W_v = eqx.nn.Linear(
            cfg["emb_dim"], cfg["emb_dim"], use_bias=cfg["qkv_bias"], key=key3
        )
        self.out_proj = eqx.nn.Linear(cfg["emb_dim"], cfg["emb_dim"], key=key4)
        self.drop = eqx.nn.Dropout(cfg["drop_rate"])

    def _create_causal_mask(self, seq_length: int) -> Bool[Array, "seq_len seq_len"]:
        """Creates a (seq_length, seq_length) boolean mask."""
        mask = jnp.tril(jnp.ones((seq_length, seq_length), dtype=jnp.bool))
        return mask

    def __call__(
        self,
        x: Float[Array, "seq_len emb_dim"],
        inference: bool = False,
        key: PRNGKeyArray = None,
    ) -> Float[Array, "seq_len emb_dim"]:
        queries = jax.vmap(self.W_q)(x)
        keys = jax.vmap(self.W_k)(x)
        values = jax.vmap(self.W_v)(x)
        queries = einops.rearrange(
            queries,
            "seq_len (heads head_dim) -> heads seq_len head_dim",
            heads=self.n_heads,
        )
        keys = einops.rearrange(
            keys,
            "seq_len (heads head_dim) -> heads seq_len head_dim",
            heads=self.n_heads,
        )
        values = einops.rearrange(
            values,
            "seq_len (heads head_dim) -> heads seq_len head_dim",
            heads=self.n_heads,
        )
        queries = queries / jnp.sqrt(keys.shape[-1])
        attention_scores = queries @ einops.rearrange(
            keys, "heads seq_len head_dim -> heads head_dim seq_len"
        )
        mask = self._create_causal_mask(x.shape[0])
        attention_scores = jnp.where(mask, attention_scores, -jnp.inf)
        attention_weights = jax.nn.softmax(attention_scores, axis=-1)
        attention_weights = self.drop(attention_weights, inference=inference, key=key)
        context_weights = attention_weights @ values
        context_weights = einops.rearrange(
            context_weights,
            "heads seq_len head_dim -> seq_len (heads head_dim)",
        )
        out_proj = jax.vmap(self.out_proj)(context_weights)
        return out_proj


class MLP(eqx.Module):
    layers: list

    def __init__(self, cfg: dict[str, int | float | bool], key: PRNGKeyArray):
        key1, key2 = jr.split(key)
        self.layers = [
            eqx.nn.Linear(cfg["emb_dim"], cfg["emb_dim"] * 4, key=key1),
            jax.nn.gelu,
            eqx.nn.Linear(cfg["emb_dim"] * 4, cfg["emb_dim"], key=key2),
        ]

    def __call__(
        self, x: Float[Array, "seq_len emb_dim"]
    ) -> Float[Array, "seq_len emb_dim"]:
        for layer in self.layers:
            x = jax.vmap(layer)(x)
        return x


class TransformerBlock(eqx.Module):
    attn: MultiHeadedAttention
    mlp: MLP
    ln1: eqx.nn.LayerNorm
    ln2: eqx.nn.LayerNorm
    drop: eqx.nn.Dropout

    def __init__(self, cfg: dict[str, int | float | bool], dtype, key: PRNGKeyArray):
        key1, key2 = jr.split(key)
        self.attn = MultiHeadedAttention(cfg, key1)
        self.mlp = MLP(cfg, key2)
        self.ln1 = eqx.nn.LayerNorm(cfg["emb_dim"], dtype=dtype)
        self.ln2 = eqx.nn.LayerNorm(cfg["emb_dim"], dtype=dtype)
        self.drop = eqx.nn.Dropout(cfg["drop_rate"])

    def __call__(
        self,
        x: Float[Array, "seq_len emb_dim"],
        *,
        inference: bool = False,
        key: PRNGKeyArray = None,
    ) -> Float[Array, "seq_len emb_dim"]:
        if key is not None:
            key_attn, key_drop1, key_drop2 = jr.split(key, 3)
        else:
            key_attn = key_drop1 = key_drop2 = None
        shortcut = x
        x = jax.vmap(self.ln1)(x)
        x = self.attn(x, inference=inference, key=key_attn)
        x = self.drop(x, inference=inference, key=key_drop1) + shortcut

        shortcut = x
        x = jax.vmap(self.ln2)(x)
        x = self.mlp(x)
        return self.drop(x, inference=inference, key=key_drop2) + shortcut


class GPTModel(eqx.Module):
    shared: eqx.nn.Shared
    pos_embed: Float[Array, "seq_len emb_dim"]
    drop_emb: eqx.nn.Dropout
    trf_blocks: list[TransformerBlock]
    final_norm: eqx.nn.LayerNorm

    def __init__(
        self, cfg: dict[str, int | float | bool], dtype: jnp.dtype, key: PRNGKeyArray
    ):
        key1, key2, key3, key4 = jr.split(key, 4)
        tok_embed = eqx.nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], key=key1)
        self.pos_embed = eqx.nn.Embedding(
            cfg["seq_len"], cfg["emb_dim"], key=key2
        ).weight
        self.drop_emb = eqx.nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = [
            TransformerBlock(cfg, dtype, keyn)
            for keyn in jr.split(key3, cfg["n_layers"])
        ]
        self.final_norm = eqx.nn.LayerNorm(cfg["emb_dim"], dtype=dtype)
        out_head = eqx.nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], use_bias=False, key=key4
        )
        # Tie token embedding layer and output layer together
        self.shared = eqx.nn.Shared(
            (tok_embed, out_head),
            where=lambda embed_and_lin: embed_and_lin[1].weight,
            get=lambda embed_and_lin: embed_and_lin[0].weight,
        )

    def __call__(
        self,
        x: Int[Array, " seq_len"],
        inference: bool = False,
        key: PRNGKeyArray = None,
    ) -> Float[Array, "seq_len vocab_size"]:
        if key is not None:
            key_drop, key_trf = jr.split(key, 2)
        else:
            key_drop = key_trf = None
        tok_embed, out_head = self.shared()
        assert tok_embed.weight is out_head.weight, "Error with weight tying"
        seq_len = x.shape[0]
        tok_embeds = jax.vmap(tok_embed)(x)
        x = tok_embeds + self.pos_embed[:seq_len, :]
        x = self.drop_emb(x, inference=inference, key=key_drop)
        for block in self.trf_blocks:
            key_trf, subkey_trf = key_split_allowing_none(key_trf)
            x = block(x, inference=inference, key=subkey_trf)
        x = jax.vmap(self.final_norm)(x)
        return jax.vmap(out_head)(x)


if __name__ == "__main__":
    import equinox as eqx
    import jax.random as jr
    import jax
    from rich import print
    from config import GPT_CONFIG
    from gpt.model import GPTModel, reinit_model_params

    key = jr.key(21)
    a = eqx.nn.Linear(40, 22, key=key)
    print(a.weight.mean(), a.weight.std())
    cfg = GPT_CONFIG["small"]
    cfg.update({"seq_len": 256})
    model = GPTModel(cfg, key)
    # Token embedding: normal(0,0.02)
    print(model.shared.pytree[0].weight.mean(), model.shared.pytree[0].weight.std())
    # Positional embedding: normal(0,0.01)
    print(model.pos_embed.mean(), model.pos_embed.std())
    print(
        model.trf_blocks[0].mlp.layers[0].weight.mean(),
        model.trf_blocks[0].mlp.layers[0].weight.std(),
    )
    # Linear layer weights: normal(0,0.02)
    print(
        model.trf_blocks[0].mlp.layers[0].bias.mean(),
        model.trf_blocks[0].mlp.layers[0].bias.std(),
    )
    print(
        model.trf_blocks[-1].mlp.layers[0].weight.mean(),
        model.trf_blocks[-1].mlp.layers[0].weight.std(),
    )
    print(
        model.trf_blocks[-1].mlp.layers[0].bias.mean(),
        model.trf_blocks[-1].mlp.layers[0].bias.std(),
    )
    print(model.final_norm.weight.mean(), model.final_norm.weight.std())

    model = reinit_model_params(model, key)
    # Token embedding: normal(0,0.02)
    print(model.shared.pytree[0].weight.mean(), model.shared.pytree[0].weight.std())
    # Positional embedding: normal(0,0.01)
    print(model.pos_embed.mean(), model.pos_embed.std())
    # Linear layer weights: normal(0,0.02)
    print(
        model.trf_blocks[0].mlp.layers[0].weight.mean(),
        model.trf_blocks[0].mlp.layers[0].weight.std(),
    )
    print(
        model.trf_blocks[0].mlp.layers[0].bias.mean(),
        model.trf_blocks[0].mlp.layers[0].bias.std(),
    )
    print(
        model.trf_blocks[-1].mlp.layers[0].weight.mean(),
        model.trf_blocks[-1].mlp.layers[0].weight.std(),
    )
    print(
        model.trf_blocks[-1].mlp.layers[0].bias.mean(),
        model.trf_blocks[-1].mlp.layers[0].bias.std(),
    )
    print(model.final_norm.weight.mean(), model.final_norm.weight.std())

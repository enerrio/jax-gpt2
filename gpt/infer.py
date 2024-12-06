from typing import Optional
import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random as jr
import tiktoken
from jaxtyping import Array, Int, PRNGKeyArray
from config import GPT_CONFIG
from gpt.model import GPTModel
from gpt.utils import load


def text_to_token_ids(text: str, tokenizer: tiktoken.Encoding):
    """Convert text to array of token IDs."""
    return jnp.array(tokenizer.encode(text, allowed_special={"<|endoftext|>"}))


def token_ids_to_text(token_ids: Int[Array, " seq_len"], tokenizer: tiktoken.Encoding):
    """Convert sequence of token IDs to text."""
    return tokenizer.decode(token_ids)


def generate_simple_text(
    inference_model: eqx.Module,
    context: Int[Array, " seq_len"],
    max_new_tokens: int,
    context_size: int,
) -> Int[Array, " out_seq_len"]:
    """Run inference on some context using greedy decoding strategy."""
    for _ in range(max_new_tokens):
        idx_cond = context[-context_size:]
        logits = inference_model(idx_cond, inference=True)
        logits = logits[-1, :]
        probs = jax.nn.softmax(logits)
        idx_next = jnp.argmax(probs, keepdims=True)
        context = jnp.concatenate((context, idx_next))
    return context


def generate_text(
    inference_model: eqx.Module,
    context: Int[Array, " seq_len"],
    max_new_tokens: int,
    context_size: int,
    key: PRNGKeyArray,
    temperature: float = 0.0,
    top_k: Optional[int] = None,
) -> Int[Array, " out_seq_len"]:
    """Run inference on some context using temperature scaling and top-k decoding strategies."""
    for _ in range(max_new_tokens):
        key, subkey = jr.split(key)
        idx_cond = context[-context_size:]
        logits = inference_model(idx_cond, inference=True)
        logits = logits[-1, :]
        # Apply top k filtering
        if top_k is not None:
            top_logits, _ = jax.lax.top_k(logits, top_k)
            min_val = top_logits[-1]
            logits = jnp.where(
                logits < min_val,
                jnp.full_like(logits, -jnp.inf),
                logits,
            )
        if temperature > 0.0:
            # Apply temperature scaling
            scaled_logits = logits / temperature
            idx_next = jr.categorical(subkey, scaled_logits, shape=(1,))
        else:
            # Apply greedy decoding
            idx_next = jnp.argmax(logits, keepdims=True)
        context = jnp.concatenate((context, idx_next))
    return context


if __name__ == "__main__":
    model_key = jr.key(21)
    skeleton = GPTModel(GPT_CONFIG["small"], model_key)
    model = load("gpt2.eqx", skeleton)
    model = eqx.nn.inference_mode(model)

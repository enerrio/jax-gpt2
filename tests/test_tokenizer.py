import pytest
import jax.numpy as jnp
from gpt.infer import (
    text_to_token_ids,
    token_ids_to_text,
    generate_simple_text,
    generate_text,
)


@pytest.mark.parametrize(
    "text",
    [
        "every effort moves you",
        "Just when the idea occurred to her",
        "Hello <|endoftext|>",
    ],
)
def test_text_to_token_ids(tokenizer, text):
    token_ids = text_to_token_ids(text, tokenizer)
    expected_token_ids = jnp.array(
        tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    )
    assert jnp.allclose(token_ids, expected_token_ids)


@pytest.mark.parametrize(
    "token_ids",
    [
        jnp.array([16833, 3626, 6100, 345]),
        jnp.array([5703, 618, 262, 2126, 5091, 284, 607]),
        jnp.array([15496, 220, 50256]),
    ],
)
def test_token_ids_to_text(tokenizer, token_ids):
    text = token_ids_to_text(token_ids, tokenizer)
    expected_text = tokenizer.decode(token_ids)
    assert text == expected_text


@pytest.mark.parametrize(
    "context,max_new_tokens",
    [
        (jnp.array([16833, 3626]), 6),
        (jnp.array([16833, 3626, 6100, 345]), 6),
        (jnp.array([5703, 618, 262, 2126, 5091, 284, 607]), 10),
    ],
)
def test_generate_simple_text(inference_model, tokenizer, context, max_new_tokens):
    token_ids = generate_simple_text(
        inference_model=inference_model,
        context=context,
        max_new_tokens=max_new_tokens,
        context_size=inference_model.pos_embed.shape[0],
    )
    text = token_ids_to_text(token_ids, tokenizer)
    print(text)
    assert len(token_ids) == context.shape[0] + max_new_tokens


@pytest.mark.parametrize(
    "context,max_new_tokens,temperature,top_k",
    [
        (jnp.array([16833, 3626, 6100, 345]), 6, 0.1, 10),
        (jnp.array([16833, 3626, 6100, 345]), 6, 0.1, 5),
        (jnp.array([16833, 3626, 6100, 345]), 6, 0.1, None),
        (jnp.array([16833, 3626, 6100, 345]), 6, 1.5, 10),
        (jnp.array([16833, 3626, 6100, 345]), 6, 1.5, 5),
        (jnp.array([16833, 3626, 6100, 345]), 6, 1.5, None),
    ],
)
def test_generate_text(
    inference_model, tokenizer, context, max_new_tokens, temperature, top_k, key
):
    token_ids = generate_text(
        inference_model=inference_model,
        context=context,
        max_new_tokens=max_new_tokens,
        context_size=inference_model.pos_embed.shape[0],
        key=key,
        temperature=temperature,
        top_k=top_k,
    )
    text = token_ids_to_text(token_ids, tokenizer)
    print(text)
    assert len(token_ids) == context.shape[0] + max_new_tokens

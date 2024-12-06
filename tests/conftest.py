import pytest
import tiktoken
import equinox as eqx
import jax.random as jr
from config import GPT_CONFIG
from gpt.model import GPTModel


@pytest.fixture
def key():
    return jr.key(21)


@pytest.fixture
def cfg():
    return {
        "context_length": 5,
        "emb_dim": 32,
        "n_heads": 4,
        "qkv_bias": True,
        "depth": 1,  # MLP depth (number of hidden layers)
        "drop_rate": 0.0,  # Set drop_rate to zero for testing
    }


@pytest.fixture
def x(key, cfg):
    batch_size = 10
    return jr.normal(key, (batch_size, cfg["context_length"], cfg["emb_dim"]))


@pytest.fixture
def tokenizer():
    return tiktoken.get_encoding("gpt2")


@pytest.fixture
def inference_model():
    cfg = GPT_CONFIG["small"]
    cfg.update({"seq_len": 256})
    gpt = GPTModel(cfg, jr.key(21))
    gpt = eqx.nn.inference_mode(gpt)
    return gpt

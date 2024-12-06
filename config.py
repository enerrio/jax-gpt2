GPT_CONFIG = {
    # 124M
    "small": {
        "vocab_size": 50257,  # vocabulary size
        "emb_dim": 768,  # Embedding dimension
        "n_heads": 12,  # Number of attention heads
        "n_layers": 12,  # Number of layers
        "drop_rate": 0.1,  # Dropout rate
        "qkv_bias": True,  # Query-Key-Value bias
    },
    # 355M
    "medium": {
        "vocab_size": 50257,
        "emb_dim": 1024,
        "n_heads": 16,
        "n_layers": 24,
        "drop_rate": 0.1,
        "qkv_bias": True,
    },
    # 774M
    "large": {
        "vocab_size": 50257,
        "emb_dim": 1280,
        "n_heads": 20,
        "n_layers": 36,
        "drop_rate": 0.1,
        "qkv_bias": True,
    },
    # 1558M
    "xlarge": {
        "vocab_size": 50257,
        "emb_dim": 1600,
        "n_heads": 25,
        "n_layers": 48,
        "drop_rate": 0.1,
        "qkv_bias": True,
    },
}

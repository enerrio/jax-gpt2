import argparse
import equinox as eqx
from jax import random as jr
import tiktoken
from rich import print as rprint
from config import GPT_CONFIG
from gpt.model import GPTModel
from gpt.utils import load
from gpt.infer import (
    text_to_token_ids,
    token_ids_to_text,
    generate_text,
)


def main(args=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_size",
        type=str,
        required=True,
        choices=["small", "medium", "large", "xlarge"],
    )
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="My name is chat and I am")
    parser.add_argument("--max_new_tokens", type=int, required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--seed", type=int, default=21)
    args = parser.parse_args(args)

    model_config = GPT_CONFIG[args.model_size]
    model_config["seq_len"] = args.seq_len

    model_key = jr.key(21)
    skeleton = GPTModel(model_config, model_key)
    rprint("Loading model...")
    model = load(f"{args.exp_name}/{args.model_name}", skeleton)
    rprint("Model loaded!")
    model = eqx.nn.inference_mode(model)
    tokenizer = tiktoken.get_encoding("gpt2")
    prompt_tokens = text_to_token_ids(args.prompt, tokenizer)
    # Run inference
    rprint("Generating...")
    token_ids = generate_text(
        inference_model=model,
        context=prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        context_size=model.pos_embed.shape[0],
        key=jr.key(args.seed),
        temperature=args.temperature,
        top_k=args.topk,
    )
    text = token_ids_to_text(token_ids, tokenizer)
    rprint(f"Model output:\n{text}")


if __name__ == "__main__":
    main()

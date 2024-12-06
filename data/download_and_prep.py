import os
import sys
import multiprocessing as mp
import argparse
import tiktoken
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from rich import print


def tokenize_gpt2(doc):
    """Tokenizes a single document and returns a numpy array of uint16 tokens."""
    tokenizer = tiktoken.get_encoding("gpt2")
    encode = lambda s: tokenizer.encode_ordinary(s)
    eot = tokenizer._special_tokens["<|endoftext|>"]  # end of text token
    tokens = [eot]  # the special <|endoftext|> token delimits all documents
    tokens.extend(encode(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (
        tokens_np < 2**16
    ).all(), "token dictionary too large for uint16"
    tokens_np_uint = tokens_np.astype(np.uint16)
    return tokens_np_uint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        choices=["the-verdict", "tinystories", "fineweb", "hellaswag"],
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        default=10**8,
        help="Size of each data shard in the output .npy files, in tokens",
    )
    args = parser.parse_args()
    name = args.data
    # Special case for very small dataset
    if name == "the-verdict":
        # Identical preprocessing as from my original jax gpt2 repo
        print("Processing the verdict dataset...")
        with open("data/the-verdict.txt", "r", encoding="utf-8") as f:
            raw_text = f.read()
        split_idx = int(0.9 * len(raw_text))
        train_data = {"text": raw_text[:split_idx]}
        val_data = {"text": raw_text[split_idx:]}
        train_tokens = tokenize_gpt2(train_data)
        val_tokens = tokenize_gpt2(val_data)
        # drop eot token
        train_tokens = train_tokens[1:]
        val_tokens = val_tokens[1:]
        os.makedirs("data/the-verdict", exist_ok=True)
        np.save("data/the-verdict/the-verdict_train_000000.npy", train_tokens)
        np.save("data/the-verdict/the-verdict_val_000000.npy", val_tokens)
        print("Done.")
        sys.exit(0)
    dataset_name, dataset_config = {
        "tinystories": ("roneneldan/TinyStories", "default"),
        "fineweb": ("HuggingFaceFW/fineweb", "sample-10BT"),
    }[name]
    data_cache_dir = os.path.join("data", name)
    os.makedirs(data_cache_dir, exist_ok=True)
    print(f"Downloading {dataset_name} to ~/.cache/huggingface/datasets")
    hfdataset = load_dataset(dataset_name, name=dataset_config, split="train")
    print(f"Number of rows in the dataset: {hfdataset.num_rows}")
    print(f"Shard size (in tokens): {args.shard_size:,}")
    print(f"Converting to numpy arrays and saving to: {data_cache_dir}")

    # tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
    nprocs = max(1, os.cpu_count() - 2)  # don't hog the entire system
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        # preallocate buffer to hold current shard
        all_tokens_np = np.empty((args.shard_size,), dtype=np.uint16)
        token_count = 0
        progress_bar = None

        for tokens in pool.imap(tokenize_gpt2, hfdataset, chunksize=16):
            # is there enough space in the current shard for the new tokens?
            if token_count + len(tokens) < args.shard_size:
                # simply append tokens to current shard
                all_tokens_np[token_count : token_count + len(tokens)] = tokens
                token_count += len(tokens)
                # update progress bar
                if progress_bar is None:
                    progress_bar = tqdm(
                        total=args.shard_size,
                        unit="tokens",
                        desc=f"Shard {shard_index}",
                    )
                progress_bar.update(len(tokens))
            else:
                # write the current shard and start a new one
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(
                    data_cache_dir, f"{name}_{split}_{shard_index:06d}.npy"
                )
                # split the document into whatever fits in this shard; the remainder goes to next one
                remainder = args.shard_size - token_count
                progress_bar.update(remainder)
                all_tokens_np[token_count : token_count + remainder] = tokens[
                    :remainder
                ]
                np.save(filename, all_tokens_np)
                shard_index += 1
                progress_bar = None
                # populate the next shard with the leftovers of the current doc
                all_tokens_np[0 : len(tokens) - remainder] = tokens[remainder:]
                token_count = len(tokens) - remainder

        # write any remaining tokens as the last shard
        if token_count != 0:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(
                data_cache_dir, f"{name}_{split}_{shard_index:06d}.npy"
            )
            np.save(filename, all_tokens_np[:token_count])
    print("Done.")


if __name__ == "__main__":
    main()

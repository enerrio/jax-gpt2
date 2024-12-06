import os
from typing import Any
import numpy as np
import tiktoken
from rich import print
from rich.progress import track
from datasets import load_dataset, Dataset


def process_hellaswag(dataset: Dataset, tokenizer: tiktoken.Encoding) -> dict[str, Any]:
    """Processes the HellaSwag dataset by tokenizing contexts and options."""
    # processed_data = []
    prompt_tokens_list = []
    options_tokens_list = []
    labels_list = []
    prompt_lengths = []
    option_lengths = []
    for i in track(range(len(dataset)), description="Processing Hellaswag..."):
        context = dataset["ctx"][i]
        label = dataset["label"][i]
        endings = dataset["endings"][i]

        context_tokens = tokenizer.encode(context)
        # Tokenize each ending, prepending a space to ensure proper tokenization
        options_tokens = [tokenizer.encode(" " + end) for end in endings]

        pair_lengths = [len(context_tokens) + len(opt) for opt in options_tokens]
        max_pair_length = max(pair_lengths)

        # Pad each prompt-ending pair to the maximum length within this example
        padded_options_tokens = []
        option_length_list = []
        for opt in options_tokens:
            pair = context_tokens + opt
            padding_length = max_pair_length - len(pair)
            if padding_length > 0:
                pair_padded = pair + [0] * padding_length
            else:
                pair_padded = pair  # No padding needed
            padded_options_tokens.append(pair_padded)
            option_length_list.append(len(opt))  # Track original option length

        prompt_tokens_list.append(context_tokens)
        options_tokens_list.append(padded_options_tokens)
        labels_list.append(label)
        prompt_lengths.append(len(context_tokens))
        option_lengths.append(option_length_list)
    return {
        "prompt_tokens": prompt_tokens_list,  # List[List[int]]
        "options_tokens": options_tokens_list,  # List[List[List[int]]]
        "labels": labels_list,  # List[int]
        "prompt_lengths": prompt_lengths,  # List[int]
        "option_lengths": option_lengths,  # List[List[int]]
    }


def main() -> None:
    print("Processing the HellaSwag dataset...")
    dataset_name = "Rowan/hellaswag"
    data_cache_dir = os.path.join("data", "hellaswag")
    os.makedirs(data_cache_dir, exist_ok=True)

    # Load the dataset splits
    print("Downloading Hellaswag to ~/.cache/huggingface/datasets")
    hfdataset_val = load_dataset(dataset_name, split="validation")
    print(f"Loaded HellaSwag validation split with {len(hfdataset_val):,} examples.")

    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    print("Processing validation split...")
    processed_val = process_hellaswag(hfdataset_val, tokenizer)
    np.savez_compressed(
        os.path.join(data_cache_dir, "hellaswag_val.npz"),
        prompt_tokens=np.array(processed_val["prompt_tokens"], dtype=object),
        options_tokens=np.array(processed_val["options_tokens"], dtype=object),
        labels=np.array(processed_val["labels"], dtype=np.uint8),
        prompt_lengths=np.array(processed_val["prompt_lengths"], dtype=np.int32),
        option_lengths=np.array(processed_val["option_lengths"], dtype=object),
    )
    print(processed_val["prompt_tokens"][:2])
    print(processed_val["options_tokens"][:2])
    print(processed_val["labels"][:2])

    # a = np.load(os.path.join(data_cache_dir, "hellaswag_val.npz"), allow_pickle=True)
    # b = np.array(a["prompt_tokens"][0], dtype=np.uint16)
    # c = a["labels"][0]
    # print('test')
    # print(a["prompt_tokens"][0])
    # print(type(b), b.dtype)
    # print(b)
    # print(c)
    print("HellaSwag processing complete.")


if __name__ == "__main__":
    main()

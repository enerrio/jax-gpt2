import os
import numpy as np
import jax.numpy as jnp
from jaxtyping import Array, Int
from torch.utils.data import DataLoader, Dataset

# def set_seed(seed: int):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)

# def worker_init_fn(worker_id):
#     worker_seed = 42  # Or any fixed seed
#     np.random.seed(worker_seed)
#     random.seed(worker_seed)

# # Set a fixed seed for reproducibility
# set_seed(42)


class HellaSwagDataset(Dataset):
    def __init__(self, data_dir: str) -> None:
        data = np.load(data_dir, allow_pickle=True)
        self.prompt_tokens = data["prompt_tokens"]  # List[List[int]]
        self.options_tokens = data["options_tokens"]  # List[List[List[int]]]
        self.labels = data["labels"]  # List[int]
        self.prompt_lengths = data["prompt_lengths"]  # List[int]
        self.option_lengths = data["option_lengths"]  # List[List[int]]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[list[int], list[list[int]], int]:
        return (
            self.prompt_tokens[idx],  # List[int]
            self.options_tokens[idx],  # List[List[int]] (4 options)
            self.labels[idx],  # int
            self.prompt_lengths[idx],  # int
            self.option_lengths[idx],  # List[int]
        )


def collate_hellaswag(
    batch: list[tuple[list[int], list[list[int]], int]]
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Convert to Jax arrays."""
    prompts, options, labels, prompt_lengths, option_lengths = zip(*batch)
    # print(f"prompts len: {len(prompts)} | content: {prompts[0]}")
    # print(f"options len: {len(options[0])} | content: {options[0]}")
    # print(f"prompt_lengths len: {len(prompt_lengths)} | content: {prompt_lengths[0]}")
    # print(f"option_lengths len: {len(option_lengths)} | content: {option_lengths[0]}")
    assert (
        len(labels) == 1
    ), f"Batch size must be 1 for Hellaswag dataset {len(labels)} != 1"
    prompt_len = len(prompts[0])
    # Determine maximum lengths across the options
    max_option_len = max(option_lengths[0])

    # Initialize batched arrays with padding
    batched_completions = jnp.zeros(
        (1, 4, prompt_len + max_option_len), dtype=jnp.uint16
    )
    batched_labels = jnp.array(labels, dtype=jnp.uint16)
    # Mask indicating completion tokens: 1 where the token is part of the prompt+option, 0 otherwise
    masked_completion = jnp.zeros((1, 4, prompt_len + max_option_len), dtype=jnp.uint16)

    prompt_len = prompt_lengths[0]
    for j in range(4):
        option = options[0][j]
        option_len = option_lengths[0][j]
        batched_completions = batched_completions.at[0, j, : len(option)].set(
            jnp.array(option, dtype=jnp.uint16)
        )
        # Create mask: 1 for completion tokens, 0 for prompt tokens and padding
        masked_completion = masked_completion.at[0, j, : prompt_len + option_len].set(1)

    return (
        batched_completions[0],  # (1, 4, L+M)
        masked_completion[0],  # (1, 4, L+M)
        batched_labels[0],  # (1,)
    )


class GPTDataset(Dataset):
    def __init__(self, data_dir: str, split: str, seq_len: int, stride: int) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.stride = stride
        self.file_paths = self._get_file_paths(data_dir, split)

        # Memory map all files
        self.memmaps = [
            np.memmap(fp, dtype=np.uint16, mode="r") for fp in self.file_paths
        ]

        self.file_sizes = [fp.shape[0] for fp in self.memmaps]
        # self.cumulative_sizes = np.cumsum(self.file_sizes)
        self.sequences_per_file = [
            (size - seq_len) // stride + 1 for size in self.file_sizes
        ]
        self.cumulative_sequences = np.cumsum(self.sequences_per_file)
        self.total_sequences = self.cumulative_sequences[-1]
        # self.total_sequences = sum(
        #     (size - seq_len) // stride + 1 for size in self.file_sizes
        # )

    def _get_file_paths(self, data_dir: str, split: str) -> list[str]:
        files = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".npy") and f"_{split}_" in f
        ]
        return files

    def __len__(self) -> int:
        return self.total_sequences

    def __getitem__(self, idx: int):
        # Find which file the idx falls into
        file_idx = np.searchsorted(self.cumulative_sequences, idx, side="right")
        if file_idx == 0:
            seq_idx = idx
        else:
            seq_idx = idx - self.cumulative_sequences[file_idx - 1]

        # Calculate the starting token index in the file
        start_token = seq_idx * self.stride
        end_token = start_token + self.seq_len

        # Handle cases where the last sequence might exceed file size
        if end_token + 1 > self.file_sizes[file_idx]:
            raise IndexError("Sequence exceeds file size.")

        # Fetch the sequence
        sequence = self.memmaps[file_idx][start_token : end_token + 1]

        # For language modeling, targets are inputs shifted by one token
        input_ids = sequence[:-1]
        target_ids = sequence[1:]

        return input_ids, target_ids


def collate_fn(
    batch: list[tuple[list[int], list[int]]]
) -> tuple[Int[Array, "batch seq_len"], Int[Array, "batch seq_len"]]:
    """Convert tensors to Jax arrays."""
    input_batch, target_batch = zip(*batch)
    input_array = jnp.array(input_batch)
    target_array = jnp.array(target_batch)
    return input_array, target_array


def create_dataloader(
    data_dir: str,
    split: str,
    seq_len: int = 256,
    stride: int = 128,
    batch_size: int = 4,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Instantiate our custom Dataset and dataloader."""
    dataset = GPTDataset(data_dir=data_dir, split=split, seq_len=seq_len, stride=stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        collate_fn=collate_fn,
        # worker_init_fn=worker_init_fn,
    )
    return dataloader


def load_hellaswag_data(
    data_dir: str,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    dataset = HellaSwagDataset(data_dir=data_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_hellaswag,
    )
    return dataloader


def load_data(
    data_dir: str,
    split: str,
    seq_len: int,
    stride: int,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    """Load data, tokenize, and create dataloaders."""
    dataloader = create_dataloader(
        data_dir,
        split,
        seq_len=seq_len,
        stride=stride,
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=0,
    )
    return dataloader


if __name__ == "__main__":
    # Path to the padded validation data
    npy_file = "data/hellaswag/hellaswag_val.npz"

    # Create the DataLoader
    hellaswag_dataloader = load_hellaswag_data(npy_file, batch_size=1)

    # Iterate through the DataLoader
    for i, batch in enumerate(hellaswag_dataloader):
        completions, masked_completions, label = batch
        print(
            "Completions shape:", completions.shape
        )  # (1, 4, prompt + max_option_len)
        print(
            "Masked completions shape:", masked_completions.shape
        )  # (4, prompt + max_option_len)
        print("Label:", label)
        print(type(completions), type(masked_completions), type(label))
        print(completions.dtype, masked_completions.dtype)
        print(completions)
        print(masked_completions)
        print(label)
        print("-" * 50)
        if i > 4:
            break  # Remove this break to iterate through the entire dataset

    # train = load_data("data/the-verdict", "train", 10, 10, 2)
    # val = load_data("data/the-verdict", "val", 10, 10, 2)
    # print(len(train))
    # x, y = next(iter(train))
    # print(type(x), type(y))
    # print(x.shape, y.shape, x.dtype, y.dtype)
    # print(x[0])
    # print(y[0])

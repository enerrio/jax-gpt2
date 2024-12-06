import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from torch.utils.data import DataLoader
from rich.progress import Progress
from tqdm import tqdm


@eqx.filter_jit
def hellaswag_step(inference_model: eqx.Module, completions, masked_completions) -> int:
    logits = jax.vmap(inference_model, in_axes=(0, None, None))(completions, True, None)
    # Shift logits and completions by one
    # print("logits:", logits.shape)
    logits = logits[..., :-1, :]
    # print("logits:", logits.shape)
    completions = completions[..., 1:]
    # print("completions:", completions.shape)
    masked_completions = masked_completions[..., 1:]
    # print("masked_completions:", masked_completions.shape)
    loss = optax.losses.softmax_cross_entropy_with_integer_labels(logits, completions)
    # print(f"loss shape: {loss.shape}")

    # Zero out losses according to our mask
    masked_loss = loss * masked_completions
    # print(f"masked_loss: {masked_loss.shape}\n{masked_loss}")
    # Sum losses over all completion tokens
    summed_loss = jnp.sum(masked_loss, axis=1)
    # print(f"summed_loss: {summed_loss.shape}\n{summed_loss}")
    # print(f"masked completion: {masked_completions}")
    # print(f"Summed mask: {jnp.sum(masked_completions, axis=1)}")
    summed_loss = summed_loss / jnp.sum(masked_completions, axis=1)
    # print(f"summed_loss: {summed_loss.shape}\n{summed_loss}")

    prediction = jnp.argmin(summed_loss)
    # print(f"Prediction: {prediction}")
    return prediction


def hellaswag(
    inference_model: eqx.Module,
    dataloader: DataLoader,
    main_pbar: Progress,
    hellaswag_task: int,
) -> float:
    """Evaluate the model on the HellaSwag dataset."""
    correct = 0

    for batch in dataloader:
        completions, masked_completions, label = batch
        # Make predictions on all prompt + options
        # print(f"\ncompletions shape: {completions.shape}")
        # print(f"masked_completions shape: {masked_completions.shape}")
        prediction = hellaswag_step(inference_model, completions, masked_completions)

        # Compare with true labels
        correct += prediction == label
        # print(correct)
        main_pbar.update(hellaswag_task, advance=1)

    # Calculate accuracy
    accuracy = correct / len(dataloader)
    return accuracy


if __name__ == "__main__":
    from data import load_hellaswag_data
    from model import GPTModel
    from utils import configure_pbar
    from rich.panel import Panel
    from rich.console import Group
    from rich.live import Live
    import jax.random as jr

    cfg = {
        "vocab_size": 50257,  # vocabulary size
        "emb_dim": 768,  # Embedding dimension
        "n_heads": 12,  # Number of attention heads
        "n_layers": 12,  # Number of layers
        "drop_rate": 0.1,  # Dropout rate
        "qkv_bias": True,  # Query-Key-Value bias
    }
    cfg.update({"seq_len": 256})
    gpt = GPTModel(cfg, jr.key(21))
    inference_model = eqx.nn.inference_mode(gpt)
    dataloader = load_hellaswag_data(
        data_dir="data/hellaswag/hellaswag_val.npz",
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    metadata_pbar, main_pbar = configure_pbar()
    panel = Panel(
        Group(metadata_pbar, main_pbar),
        title="Training GPT-2",
        style="gold1",
    )
    print(f"size of dataloader: {len(dataloader)}")
    with Live(panel):
        hellaswag_task = main_pbar.add_task(
            "[magenta1]Evaluating model on Hellaswag...",
            total=len(dataloader),
            visible=True,
        )
        acc = hellaswag(inference_model, dataloader, main_pbar, hellaswag_task)
        print("done")
    print(f"Metric results: {acc}")

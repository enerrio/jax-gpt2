import time
import logging
import tiktoken
from typing import Iterator
from rich.live import Live
from rich.panel import Panel
from rich.console import Group
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from jaxtyping import Array, Int, Key, PyTree, Scalar
from torch.utils.data import DataLoader
from gpt.utils import configure_pbar, save
from gpt.infer import generate_text, text_to_token_ids, token_ids_to_text

# Get the logger instance
logger = logging.getLogger("train")
tokenizer = tiktoken.get_encoding("gpt2")
prompt = "Once upon a time"
prompt_tokens = text_to_token_ids(prompt, tokenizer)


def infinite_dataloader(
    dataloader: DataLoader,
) -> Iterator[tuple[Int[Array, "batch seq_len"], Int[Array, "batch seq_len"]]]:
    while True:
        for x_batch, y_batch in dataloader:
            yield x_batch, y_batch


@eqx.filter_value_and_grad
def loss_fn(
    model: eqx.Module,
    x: Int[Array, "batch seq_len"],
    y: Int[Array, "batch seq_len"],
    keys: Key[Array, " batch"],
) -> Scalar:
    """Forward pass of model and compute loss."""
    logits = jax.vmap(model, in_axes=(0, None, 0))(x, False, keys)
    loss = optax.losses.softmax_cross_entropy_with_integer_labels(logits, y)
    return loss.mean()


@eqx.filter_jit
def train_step(
    model: eqx.Module,
    optim: optax.GradientTransformation,
    opt_state: PyTree,
    x: Int[Array, "batch seq_len"],
    y: Int[Array, "batch seq_len"],
    keys: Key[Array, " batch"],
) -> tuple[eqx.Module, PyTree, Scalar, Scalar]:
    """Single training step for a batch of data. Forward pass, compute loss/grads, update weights."""
    loss, grads = loss_fn(model, x, y, keys)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    grad_norm = optax.tree_utils.tree_l2_norm(grads)
    return model, opt_state, loss, grad_norm


@eqx.filter_jit
def validate_step(
    inference_model: eqx.Module,
    x: Int[Array, "batch seq_len"],
    y: Int[Array, "batch seq_len"],
) -> Scalar:
    def validation_loss_fn(
        model: eqx.Module,
        x: Int[Array, "batch seq_len"],
        y: Int[Array, "batch seq_len"],
    ) -> Scalar:
        logits = jax.vmap(model, in_axes=(0, None, None))(x, True, None)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, y)
        return loss.mean()

    loss = validation_loss_fn(inference_model, x, y)
    return loss


def train(
    model: eqx.Module,
    optim: optax.GradientTransformation,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    key: Key[Array, ""],
    num_steps: int,
    eval_freq: int,
    checkpoint_freq: int,
    checkpoint_name: str,
) -> eqx.Module:
    """Train the model."""
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    tokens_seen = 0
    metadata_pbar, main_pbar = configure_pbar()
    panel = Panel(
        Group(metadata_pbar, main_pbar),
        title="Training GPT-2",
        style="gold1",
    )

    train_iter = infinite_dataloader(train_dataloader)

    # with progress:
    with Live(panel):
        main_task = main_pbar.add_task("[red]Training model...", total=num_steps)
        val_task = main_pbar.add_task(
            "[magenta1]Evaluating model on validation set...", total=len(val_dataloader), visible=False
        )
        metadata_task = metadata_pbar.add_task(
            "Metadata", total=num_steps, tokens_seen=tokens_seen
        )
        for step in range(1, num_steps + 1):
            lr = opt_state.hyperparams["learning_rate"]
            start = time.time()
            # train phase
            x_batch, y_batch = next(train_iter)
            key, *subkeys = jr.split(key, len(x_batch) + 1)
            subkeys = jnp.array(subkeys)
            model, opt_state, loss, grad_norm = train_step(
                model, optim, opt_state, x_batch, y_batch, subkeys
            )

            step_time = (time.time() - start) * 1e3
            loss = loss.item()
            grad_norm = grad_norm.item()
            lr = lr.item()
            tokens_seen += x_batch.size
            throughput = x_batch.size / step_time

            logger.info(
                f"Step [{step:07d}/{num_steps:07d}] | Train Loss: {loss:.4f} | lr: {lr:.6f} | Grad Norm: {grad_norm:.3f} | Step Time: {step_time:04.0f}ms | Throughput: {throughput:02.2f} toks/s",
                extra={
                    "mode": "training",
                    "step": step,
                    "train_loss": round(loss, 4),
                    "val_loss": None,
                    "hellaswag_acc": None,
                    "learning_rate": round(lr, 6),
                    "grad_norm": round(grad_norm, 3),
                    "step_time": round(step_time, 4),
                    "throughput": round(throughput, 1),
                    "tokens_seen": tokens_seen,
                },
            )

            main_pbar.update(main_task, advance=1)
            metadata_pbar.update(metadata_task, advance=1, tokens_seen=tokens_seen)

            if (step % eval_freq) == 0:
                # validation phase
                main_pbar.update(val_task, visible=True)
                val_loss = 0.0
                inference_model = eqx.nn.inference_mode(model)
                for x_val, y_val in val_dataloader:
                    val_loss += validate_step(inference_model, x_val, y_val)
                    main_pbar.update(val_task, advance=1)
                main_pbar.reset(val_task, visible=False)

                # Average and store loss
                val_loss /= len(val_dataloader)
                val_loss = val_loss.item()

                logger.info("Generating text...")
                generated_text_ids = generate_text(
                    inference_model=inference_model,
                    context=prompt_tokens,
                    max_new_tokens=50,
                    context_size=x_batch.shape[1],
                    key=key,
                )
                generated_text = token_ids_to_text(generated_text_ids, tokenizer)
                logger.info(
                    f"Step [{step:07d}/{num_steps:07d}] | Val Loss: {val_loss:.4f} | Generated Text: {generated_text}",
                    extra={
                        "mode": "validation",
                        "step": step,
                        "train_loss": None,
                        "val_loss": round(val_loss, 4),
                        "hellaswag_acc": None,
                        "learning_rate": None,
                        "grad_norm": None,
                        "step_time": None,
                        "throughput": None,
                        "tokens_seen": None,
                        "generated_text": generated_text,
                    },
                )

            if (step % checkpoint_freq) == 0:
                ckpt_name = f"{checkpoint_name}-{step}-chkpt.eqx"
                logger.info(f"Checkpointing model to disk: {ckpt_name}")
                save(ckpt_name, model)
    return model

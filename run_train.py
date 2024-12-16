import os
import time
import argparse
from functools import partial
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
from config import GPT_CONFIG
from gpt.logger import setup_logger
from gpt.model import GPTModel, reinit_model_params
from gpt.train import train
from gpt.utils import save
from gpt.data import load_data


@partial(optax.inject_hyperparams, static_args="wd")
def create_optimizer(learning_rate, wd):
    return optax.chain(
        optax.scale_by_adam(),
        optax.add_decayed_weights(weight_decay=wd),
        optax.scale_by_learning_rate(learning_rate=learning_rate),
    )


def main(args=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_size",
        type=str,
        required=True,
        choices=["small", "medium", "large", "xlarge"],
    )
    parser.add_argument(
        "--data", type=str, required=True, choices=["the-verdict", "tinystories"]
    )
    parser.add_argument(
        "--nb_steps", type=int, required=True, help="Total number of optimization steps"
    )
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=4e-4, help="Max learning rate")
    parser.add_argument("--wd", type=float, default=0.1, help="Optimizer weight decay")
    parser.add_argument(
        "--warmup",
        type=int,
        default=50,
        help="Number of steps to warm up learning rate to --lr value",
    )
    parser.add_argument(
        "--decay_percentage",
        type=float,
        default=0.1,
        help="Fraction of max learning rate to decay to after warmup; 0 means decay to zero, 1 means no decay",
    )
    parser.add_argument(
        "--dtype",
        type=jnp.dtype,
        default="float32",
        choices=["float16", "bfloat16", "float32"],
        help="Data type for model weights",
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=100,
        help="How often to evaluate the model on validation set",
    )
    parser.add_argument(
        "--checkpoint_freq",
        type=int,
        default=100,
        help="How often to save the model to disk",
    )
    parser.add_argument("--exp_name", type=str, required=True)
    args = parser.parse_args(args)

    resultsdir = os.path.join("results", args.exp_name)
    os.makedirs(resultsdir, exist_ok=True)
    logfile = os.path.join(resultsdir, f"train_log_{args.exp_name}.jsonl")
    logger = setup_logger(log_file=logfile)
    logger.info(f"Logging training info to {logfile}")

    model_config = GPT_CONFIG[args.model_size]
    model_config["seq_len"] = args.seq_len
    data_dir = os.path.join(f"data/{args.data}")
    logger.info(f"Data directory: {data_dir}")

    ckpt_dir = os.path.join(resultsdir, f"gpt2-{args.model_size}-{args.exp_name}")
    logger.info(f"Saving model checkpoints to {resultsdir}")

    train_dataloader = load_data(
        data_dir, "train", args.seq_len, args.seq_len, args.batch_size
    )
    val_dataloader = load_data(
        data_dir, "val", args.seq_len, args.seq_len, args.batch_size
    )
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Number of batches in train dataloader: {len(train_dataloader)}")
    logger.info(f"Number of batches in val dataloader: {len(val_dataloader)}")

    key = jr.key(21)
    logger.info(f"Creating model with dype: {args.dtype}")
    model_key, train_key = jr.split(key)
    model = GPTModel(model_config, args.dtype, model_key)
    model = reinit_model_params(model, args.dtype, model_key)

    end_value = args.lr * args.decay_percentage
    lr_scheduler = optax.schedules.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=args.lr,
        warmup_steps=args.warmup,
        decay_steps=args.nb_steps,
        end_value=end_value,
    )
    optim = create_optimizer(lr_scheduler, args.wd)

    leaves, _ = jax.tree.flatten(model)
    num_params = sum([leaf.size for leaf in leaves if eqx.is_array(leaf)])
    logger.info(f"Total number of model parameters ({args.model_size}): {num_params:,}")
    # model_str = eqx.tree_pformat(model)
    # logger.info(model_str)

    ## Test out what initial loss should look like
    # import sys
    # initial_loss = -jnp.log(1.0 / model_config["vocab_size"])
    # logger.info(f"Initial loss should be around: {initial_loss:.3f}")
    # key, *sample_keys = jr.split(train_key, train_dataloader.batch_size + 1)
    # sample_keys = jnp.array(sample_keys)
    # x_sample, y_sample = next(iter(train_dataloader))
    # logits = jax.vmap(model, in_axes=(0, None, 0))(x_sample, False, sample_keys)
    # loss = optax.losses.softmax_cross_entropy_with_integer_labels(
    #     logits, y_sample
    # ).mean().item()
    # logits_mean = jnp.mean(logits).item()
    # logits_std = jnp.std(logits).item()
    # logger.info(
    #     f"Logits mean: {logits_mean:.3f}, std: {logits_std:.3f}, dtype: {logits.dtype}"
    # )
    # logger.info(f"Actual initial loss is: {loss:.3f}")
    # min_label = jnp.min(y_sample)
    # max_label = jnp.max(y_sample)
    # logger.info(f"Label range: {min_label} to {max_label}")
    # assert (
    #     min_label >= 0 and max_label < model_config["vocab_size"]
    # ), "Labels out of range."
    # logger.info(f"> tok_embed dtype: {model.shared.pytree[0].weight.dtype}")
    # logger.info(f"> pos_embed dtype: {model.pos_embed.dtype}")
    # logger.info(f"> trf_block ln weight dtype: {model.trf_blocks[0].ln1.weight.dtype}")
    # logger.info(f"> trf_block ln bias dtype: {model.trf_blocks[0].ln1.bias.dtype}")
    # logger.info(
    #     f"> mlp linear weight dtype: {model.trf_blocks[0].mlp.layers[0].weight.dtype}"
    # )
    # logger.info(
    #     f"> mlp linear bias dtype: {model.trf_blocks[0].mlp.layers[0].bias.dtype}"
    # )
    # logger.info(f"> attn W_q dtype: {model.trf_blocks[0].attn.W_q.weight.dtype}")
    # logger.info(
    #     f"> attn out_proj dtype: {model.trf_blocks[0].attn.out_proj.weight.dtype}"
    # )
    # logger.info(f"> final_norm dtype: {model.final_norm.weight.dtype}")
    # sys.exit(0)

    logger.info("Training...")
    start = time.time()
    model = train(
        model=model,
        optim=optim,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        key=train_key,
        num_steps=args.nb_steps,
        eval_freq=args.eval_freq,
        checkpoint_freq=args.checkpoint_freq,
        checkpoint_name=ckpt_dir,
    )
    logger.info(f"Total training time: {time.time()-start:.2f} seconds.")
    logger.info("Complete!")
    save(f"{ckpt_dir}-{args.nb_steps}-final.eqx", model)
    logger.info(f"Final model saved to disk: {ckpt_dir}-{args.nb_steps}-final.eqx")


if __name__ == "__main__":
    main()

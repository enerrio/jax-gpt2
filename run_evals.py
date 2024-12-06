import os
import argparse
from glob import glob
import equinox as eqx
from jax import random as jr
from rich.live import Live
from rich.panel import Panel
from rich.console import Group
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TaskProgressColumn,
    Column,
)
from config import GPT_CONFIG
from gpt.model import GPTModel
from gpt.data import load_hellaswag_data
from gpt.eval import hellaswag
from gpt.utils import load
from gpt.logger import setup_logger


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
    args = parser.parse_args(args)

    logfile = os.path.join(args.exp_name, f"train_log_{args.exp_name}.jsonl")
    logger = setup_logger(log_file=logfile)
    logger.info(f"Logging training info to {logfile}")

    model_config = GPT_CONFIG[args.model_size]
    model_config["seq_len"] = args.seq_len

    model_key = jr.key(21)
    skeleton = GPTModel(model_config, model_key)

    hellaswag_file = "data/hellaswag/hellaswag_val.npz"
    hellaswag_dataloader = load_hellaswag_data(hellaswag_file, batch_size=1)
    logger.info(
        f"HellaSwag DataLoader created. Number of samples: {len(hellaswag_dataloader):,}"
    )

    # Run inference on each model checkpoint
    model_checkpoints = glob(f"{args.exp_name}/*.eqx")
    # Serial
    for model_ckpt in model_checkpoints:
        logger.info(f"Loading model {model_ckpt}")
        model = load(model_ckpt, skeleton)
        inference_model = eqx.nn.inference_mode(model)
        step = model_ckpt.split("-")[-2]

        logger.info("Evaluating on HellaSwag...")
        main_pbar = Progress(
            TextColumn(
                "[progress.description]{task.description}", table_column=Column(ratio=1)
            ),
            TextColumn("•"),
            BarColumn(bar_width=None, table_column=Column(ratio=2)),
            TaskProgressColumn(
                text_format="[progress.percentage]{task.percentage:>3.1f}%"
            ),
            TimeElapsedColumn(),
            expand=True,
        )
        panel = Panel(
            Group(main_pbar),
            title="Running Evals",
            style="gold1",
        )
        with Live(panel):
            hellaswag_task = main_pbar.add_task(
                "[magenta1]Evaluating model on Hellaswag...",
                total=len(hellaswag_dataloader),
                visible=False,
            )
            main_pbar.update(hellaswag_task, visible=True)
            hellaswag_acc = hellaswag(
                inference_model, hellaswag_dataloader, main_pbar, hellaswag_task
            )
            main_pbar.reset(hellaswag_task, visible=False)
        hellaswag_acc = round(hellaswag_acc, 2)
        logger.info(
            f"Hellaswag accuracy: {hellaswag_acc:.2f}",
            extra={
                "mode": "evaluation",
                "step": step,
                "train_loss": None,
                "val_loss": None,
                "hellaswag_acc": hellaswag_acc,
                "learning_rate": None,
                "grad_norm": None,
                "step_time": None,
                "throughput": None,
                "tokens_seen": None,
            },
        )


if __name__ == "__main__":
    main()

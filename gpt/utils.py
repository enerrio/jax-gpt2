import json
from typing import Optional
import pandas as pd
import numpy as np
import equinox as eqx
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TaskProgressColumn,
    Column,
)
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

sz = "124M"
loss_baseline = {
    "124M": 3.424958,
    "350M": 3.083089,
    "774M": 3.000580,
    "1558M": 2.831273,
}[sz]
hella2_baseline = {  # for GPT-2
    "124M": 0.294463,
    "350M": 0.375224,
    "774M": 0.431986,
    "1558M": 0.488946,
}[sz]
hella3_baseline = {  # for GPT-3
    "124M": 0.337,
    "350M": 0.436,
    "774M": 0.510,
    "1558M": 0.547,
}[sz]


def save(filename: str, model: eqx.Module):
    """Save model to disk."""
    with open(filename, "wb") as f:
        eqx.tree_serialise_leaves(f, model)


def load(filename: str, model: eqx.Module):
    """Load saved model."""
    with open(filename, "rb") as f:
        return eqx.tree_deserialise_leaves(f, model)


def configure_pbar() -> tuple[Progress, Progress]:
    """Setup rich progress bar for monitoring training."""
    metadata_pbar = Progress(
        # TextColumn("{task.fields[batches_completed]:,} of [underline]{task.total:,}[/underline] batches completed"),
        TextColumn(
            "{task.completed:,} of [underline]{task.total:,}[/underline] optimization steps taken"
        ),
        TextColumn("🔥"),
        TextColumn("{task.fields[tokens_seen]:,} tokens seen"),
    )
    main_pbar = Progress(
        TextColumn(
            "[progress.description]{task.description}", table_column=Column(ratio=1)
        ),
        TextColumn("•"),
        BarColumn(bar_width=None, table_column=Column(ratio=2)),
        TaskProgressColumn(text_format="[progress.percentage]{task.percentage:>3.1f}%"),
        TimeElapsedColumn(),
        expand=True,
    )
    return metadata_pbar, main_pbar


def read_log_file(logfile: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reads a JSONL log file and returns DataFrames with training and val metrics."""
    records = []
    with open(logfile, "r") as f:
        for line in f:
            record = json.loads(line)
            records.append(record)
    df = pd.DataFrame(records)

    # Separate training and validation
    train_df = df[df["mode"] == "training"].sort_values("step").reset_index(drop=True)
    val_df = df[df["mode"] == "validation"].sort_values("step").reset_index(drop=True)
    return train_df, val_df


def plot_stats(logfile: str, plot_name: str) -> None:
    """Plots training stats Plotly."""
    train_df, val_df = read_log_file(logfile)
    fig = make_subplots(
        3,
        1,
        shared_xaxes=True,
        subplot_titles=("Training & Validation Loss", "Learning Rate"),
    )

    # Plot Training Loss
    fig.add_trace(
        go.Scatter(
            x=train_df["step"],
            y=train_df["train_loss"],
            name="Train Loss",
            line=dict(color="blue"),
        ),
        row=1,
        col=1,
    )

    # Plot Validation Loss
    fig.add_trace(
        go.Scatter(
            x=val_df["step"],
            y=val_df["val_loss"],
            name="Validation Loss",
            line=dict(color="orange"),
        ),
        row=1,
        col=1,
    )

    # Plot learning rate
    fig.add_trace(
        go.Scatter(
            x=train_df["step"],
            y=train_df["learning_rate"],
            name="Learning Rate",
            line=dict(color="blue"),
        ),
        row=2,
        col=1,
    )

    # Configure secondary x-axis for tokens_seen
    fig.update_layout(
        title="Training Stats",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    # Set x-axis title
    fig.update_xaxes(
        title_text="Step", showticklabels=True, tick0=0, dtick=10, row=1, col=1
    )
    # fig.update_xaxes(
    #     overlaying="x",
    #     side="top",
    #     showticklabels=True,
    #     # tick0=0,
    #     # dtick=1000,
    #     tickvals=train_df["step"],
    #     ticktext=train_df["tokens_seen"].astype(str),
    #     title="Tokens Seen",
    #     row=1, col=1
    # )
    fig.update_xaxes(
        title_text="Step", showticklabels=True, tick0=0, dtick=10, row=2, col=1
    )

    # Set y-axes titles
    fig.update_yaxes(title_text="<b>Loss</b>", row=1, col=1)
    fig.update_yaxes(title_text="<b>Learning Rate</b>", tickformat=".5f", row=2, col=1)
    fig.write_html(plot_name)

    fig.show()

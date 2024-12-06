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


def compute_statistics(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Computes mean and confidence intervals across multiple DataFrames."""
    # Assume all DataFrames have the same steps
    steps = dfs[0]["step"]
    tokens_seen = dfs[0]["tokens_seen"]

    train_losses = np.array([df["train_loss_train"].values for df in dfs])
    val_losses = np.array([df["val_loss_val"].values for df in dfs])

    # Compute mean and confidence intervals
    mean_train = np.nanmean(train_losses, axis=0)
    mean_val = np.nanmean(val_losses, axis=0)

    sem_train = stats.sem(train_losses, axis=0, nan_policy="omit")
    sem_val = stats.sem(val_losses, axis=0, nan_policy="omit")

    confidence = 0.95
    h_train = sem_train * stats.t.ppf((1 + confidence) / 2.0, len(dfs) - 1)
    h_val = sem_val * stats.t.ppf((1 + confidence) / 2.0, len(dfs) - 1)

    stats_df = pd.DataFrame(
        {
            "step": steps,
            "tokens_seen": tokens_seen,
            "mean_train_loss": mean_train,
            "mean_val_loss": mean_val,
            "train_loss_lower": mean_train - h_train,
            "train_loss_upper": mean_train + h_train,
            "val_loss_lower": mean_val - h_val,
            "val_loss_upper": mean_val + h_val,
        }
    )

    return stats_df


def plot_stats(
    logfiles: list[str],
    plot_name: Optional[str] = None,
    show_individual_runs: bool = False,
) -> None:
    """
    Plots training and validation loss using Plotly.

    Parameters:
    - logfiles: List of paths to JSONL log files.
    - plot_name: Optional. If provided, the plot will be saved to this file.
    - show_individual_runs: If True, overlays individual runs on the mean plot.
    """
    dfs = [read_log_file(logfile) for logfile in logfiles]

    if len(dfs) == 0:
        raise ValueError("No log files provided.")

    if len(dfs) == 1:
        # Single run
        train_df, val_df = dfs[0]
        fig = make_subplots(
            3,
            1,
            shared_xaxes=True,
            subplot_titles=("Training & Validation Loss", "Learning Rate"),
            specs=[
                [{"secondary_y": True}],  # First subplot with secondary y-axis
                [{}],  # Second subplot
                [{}],  # Third subplot (if needed)
            ],
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
            secondary_y=False,
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
            secondary_y=False,
        )

        # Plot tokens seen
        fig.add_trace(
            go.Scatter(
                x=train_df["step"],
                y=train_df["tokens_seen"],
                name="Tokens seen",
                line=dict(color="green"),
            ),
            row=1,
            col=1,
            secondary_y=True,
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
    else:
        # Multiple runs: compute statistics
        stats_df = compute_statistics(dfs)
        fig = make_subplots(specs=[[{"secondary_y": False}]])

        # Plot Mean Training Loss
        fig.add_trace(
            go.Scatter(
                x=stats_df["step"],
                y=stats_df["mean_train_loss"],
                mode="lines",
                name="Mean Train Loss",
                line=dict(color="blue"),
            )
        )

        # Confidence Interval for Training Loss
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([stats_df["step"], stats_df["step"][::-1]]),
                y=np.concatenate(
                    [stats_df["train_loss_upper"], stats_df["train_loss_lower"][::-1]]
                ),
                fill="toself",
                fillcolor="rgba(0, 0, 255, 0.1)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=True,
                name="Train Loss 95% CI",
            )
        )

        # Plot Mean Validation Loss
        fig.add_trace(
            go.Scatter(
                x=stats_df["step"],
                y=stats_df["mean_val_loss"],
                mode="lines",
                name="Mean Val Loss",
                line=dict(color="orange"),
            )
        )

        # Confidence Interval for Validation Loss
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([stats_df["step"], stats_df["step"][::-1]]),
                y=np.concatenate(
                    [stats_df["val_loss_upper"], stats_df["val_loss_lower"][::-1]]
                ),
                fill="toself",
                fillcolor="rgba(255, 165, 0, 0.1)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=True,
                name="Val Loss 95% CI",
            )
        )

        if show_individual_runs:
            for idx, df in enumerate(dfs):
                fig.add_trace(
                    go.Scatter(
                        x=df["step"],
                        y=df["train_loss"],
                        mode="lines",
                        name=f"Train Loss Run {idx+1}",
                        line=dict(color="blue", opacity=0.2),
                        showlegend=False,
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=df["step"],
                        y=df["val_loss"],
                        mode="lines",
                        name=f"Val Loss Run {idx+1}",
                        line=dict(color="orange", opacity=0.2),
                        showlegend=False,
                    )
                )

    # Configure secondary x-axis for tokens_seen
    fig.update_layout(
        title="Training Stats",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    # Set x-axis title
    fig.update_xaxes(title_text="Step")

    # Set y-axes titles
    fig.update_yaxes(title_text="<b>Loss</b>", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="<b>Tokens Seen</b>", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="<b>Learning Rate</b>", row=2, col=1)
    fig.write_html(plot_name)

    fig.show()


# def plot_stats(train_stats: dict[str, list[float | int]], plot_name: str) -> None:
#     # def plot_stats(logfile: str, plot_name: str) -> None:
#     """Plot training & validation loss."""
#     _, ax = plt.subplots()
#     ax.plot(train_stats["train_loss"], label="Train Loss")
#     ax.plot(train_stats["val_loss"], linestyle="-.", label="Validation Loss")
#     ax.set_title("Loss curve")
#     ax.set_xlabel("Epoch")
#     ax.set_ylabel("Loss")
#     ax.grid(True)
#     ax.legend()
#     ax.xaxis.set_major_locator(MaxNLocator(integer=True))
#     ax_top = ax.twiny()
#     ax_top.plot(train_stats["tokens_seen"], train_stats["train_loss"], alpha=0)
#     ax_top.set_xlabel("Tokens seen")
#     plt.tight_layout()
#     plt.savefig(plot_name)

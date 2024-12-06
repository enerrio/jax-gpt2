import pytest
import jax
import jax.numpy as jnp
import equinox as eqx
from unittest.mock import MagicMock
from torch.utils.data import DataLoader
from typing import List, Tuple
from gpt.eval import hellaswag
from gpt.data import load_hellaswag_data, HellaSwagDataset


@pytest.fixture
def hellaswag_dataloader():
    """Fixture to load the actual HellaSwag DataLoader for testing."""
    return load_hellaswag_data(
        data_dir="data/hellaswag/hellaswag_val.npz",
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )


def test_hellaswag_model_accuracy(inference_model, hellaswag_dataloader):
    """Test hellaswag function with the actual model and data."""
    metrics = hellaswag(inference_model, hellaswag_dataloader)
    accuracy = metrics["accuracy"]

    # Define expected accuracy based on your model's performance
    # For a well-trained model, expect high accuracy, e.g., > 0.7
    assert accuracy > 0.7, f"Expected accuracy > 0.7, got {accuracy}"


# def test_hellaswag_partial_correctness(inference_model, hellaswag_dataloader):
#     """Test hellaswag function for partial correctness scenarios."""
#     metrics = hellaswag(inference_model, hellaswag_dataloader)
#     accuracy = metrics["accuracy"]

#     # Define expected accuracy range
#     assert (
#         0.5 <= accuracy <= 1.0
#     ), f"Expected accuracy between 0.5 and 1.0, got {accuracy}"

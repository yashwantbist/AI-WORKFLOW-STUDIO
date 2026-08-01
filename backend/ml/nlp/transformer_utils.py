"""Small utilities for decoder-only Transformer experiments."""

import torch
from torch import Tensor, nn


def prepare_next_token_batch(
    token_ids: Tensor,
    padding_id: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Shift token sequences into language-model inputs and targets.

    Example:

    ``[the, cat, sat, <pad>]`` becomes:

    - input: ``[the, cat, sat]``
    - target: ``[cat, sat, <pad>]``
    """
    if token_ids.ndim != 2:
        raise ValueError("token_ids must have shape [batch, sequence]")
    if token_ids.size(1) < 2:
        raise ValueError("At least two tokens are required for prediction")

    input_ids = token_ids[:, :-1]
    target_ids = token_ids[:, 1:]
    input_mask = input_ids.ne(padding_id)
    target_mask = target_ids.ne(padding_id)
    return input_ids, target_ids, input_mask, target_mask


def calculate_next_token_loss(
    logits: Tensor,
    target_ids: Tensor,
    target_mask: Tensor,
) -> Tensor:
    """Calculate cross-entropy only at positions containing real targets."""
    if logits.ndim != 3:
        raise ValueError(
            "logits must have shape [batch, sequence, vocabulary]"
        )
    if target_ids.shape != logits.shape[:2]:
        raise ValueError("target_ids must match logits' first two dimensions")
    if target_mask.shape != target_ids.shape:
        raise ValueError("target_mask must have the same shape as target_ids")

    target_mask = target_mask.bool()
    if not target_mask.any():
        raise ValueError("At least one real target token is required")

    losses = nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target_ids.reshape(-1),
        reduction="none",
    ).reshape_as(target_ids)
    return losses[target_mask].mean()


def future_attention_values(attention_weights: Tensor) -> Tensor:
    """Return only probabilities above the main diagonal."""
    if attention_weights.ndim < 2:
        raise ValueError("attention_weights must have at least 2 dimensions")
    if attention_weights.size(-1) != attention_weights.size(-2):
        raise ValueError("The last two attention dimensions must be square")

    return torch.triu(attention_weights, diagonal=1)


def has_no_future_attention(
    attention_weights: Tensor,
    tolerance: float = 1e-7,
) -> bool:
    """Return whether every future-token probability is effectively zero."""
    if tolerance < 0:
        raise ValueError("tolerance must not be negative")

    future_values = future_attention_values(attention_weights)
    return bool((future_values.abs() <= tolerance).all().item())


def assert_no_future_attention(
    attention_weights: Tensor,
    tolerance: float = 1e-7,
) -> None:
    """Raise an informative error if any token looks to its right."""
    if not has_no_future_attention(attention_weights, tolerance):
        largest_value = future_attention_values(
            attention_weights
        ).abs().max().item()
        raise AssertionError(
            "Future-token attention was not zero. "
            f"Largest blocked probability: {largest_value:.8f}"
        )


def format_boolean_matrix(attention_mask: Tensor) -> str:
    """Format a 2D boolean matrix as beginner-friendly ones and zeroes."""
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must be a 2D tensor")

    return "\n".join(
        " ".join("1" if allowed else "0" for allowed in row.tolist())
        for row in attention_mask.bool().cpu()
    )

"""Boolean attention-permission utilities for causal language models."""

import torch
from torch import Tensor


def create_causal_attention_mask(
    sequence_length: int,
    device: torch.device | str | None = None,
) -> Tensor:
    """Return a lower-triangular causal permission matrix.

    Rows represent query positions. Columns represent key positions.
    ``True`` means attention is allowed. Position ``i`` may use positions
    ``0`` through ``i``, but it may not use any position greater than ``i``.
    """
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")

    return torch.ones(
        sequence_length,
        sequence_length,
        dtype=torch.bool,
        device=device,
    ).tril()


def combine_causal_and_padding_masks(
    padding_mask: Tensor,
) -> Tensor:
    """Combine autoregressive and padding rules for a whole batch.

    Args:
        padding_mask: ``[batch, sequence]`` tensor where ``True`` identifies
            a real token and ``False`` identifies padding.

    Returns:
        A boolean tensor shaped ``[batch, query_sequence, key_sequence]``.
        It can be passed directly to Day 10's ``MultiHeadAttention``.
    """
    if padding_mask.ndim != 2:
        raise ValueError(
            "padding_mask must have shape [batch, sequence]"
        )
    if padding_mask.size(0) < 1 or padding_mask.size(1) < 1:
        raise ValueError("padding_mask dimensions must be positive")

    padding_mask = padding_mask.bool()
    sequence_length = padding_mask.size(1)
    causal_mask = create_causal_attention_mask(
        sequence_length=sequence_length,
        device=padding_mask.device,
    ).unsqueeze(0)

    # A valid attention connection needs three permissions:
    # 1. the key is not in the future;
    # 2. the query is a real token;
    # 3. the key is a real token.
    real_queries = padding_mask.unsqueeze(2)
    real_keys = padding_mask.unsqueeze(1)
    return causal_mask & real_queries & real_keys

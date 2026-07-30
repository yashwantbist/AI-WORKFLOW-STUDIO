"""Multi-head self-attention built from beginner-friendly PyTorch operations."""

import math

import torch
from torch import Tensor, nn


def _expand_attention_mask(
    attention_mask: Tensor,
    scores: Tensor,
) -> Tensor:
    """Convert a token mask into a mask broadcastable over attention scores.

    Accepted mask shapes:

    - ``[batch, key_sequence]``
    - ``[batch, query_sequence, key_sequence]``
    - ``[batch, heads, query_sequence, key_sequence]``

    The project convention is ``True`` for a real token and ``False`` for
    padding. This is the opposite of PyTorch's ``key_padding_mask`` convention.
    """
    if attention_mask.ndim == 2:
        expanded_mask = attention_mask[:, None, None, :]
    elif attention_mask.ndim == 3:
        expanded_mask = attention_mask[:, None, :, :]
    elif attention_mask.ndim == 4:
        expanded_mask = attention_mask
    else:
        raise ValueError(
            "attention_mask must have 2, 3, or 4 dimensions"
        )

    expanded_mask = expanded_mask.to(
        device=scores.device,
        dtype=torch.bool,
    )

    try:
        torch.broadcast_shapes(scores.shape, expanded_mask.shape)
    except RuntimeError as error:
        raise ValueError(
            "attention_mask cannot be broadcast to attention scores: "
            f"scores={tuple(scores.shape)}, "
            f"mask={tuple(expanded_mask.shape)}"
        ) from error

    return expanded_mask


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Tensor | None = None,
    dropout: nn.Module | None = None,
) -> tuple[Tensor, Tensor]:
    """Apply scaled dot-product attention.

    Args:
        query: Tensor shaped ``[batch, heads, query_sequence, head_dimension]``.
        key: Tensor shaped ``[batch, heads, key_sequence, head_dimension]``.
        value: Tensor shaped
            ``[batch, heads, key_sequence, value_dimension]``.
        attention_mask: Optional mask where ``True`` means a key may be used.
        dropout: Optional dropout module applied to the attention probabilities.

    Returns:
        A pair containing the attended values and attention probabilities.
    """
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must all be 4D tensors")
    if query.shape[:2] != key.shape[:2]:
        raise ValueError("query and key must have the same batch and head sizes")
    if key.shape[:3] != value.shape[:3]:
        raise ValueError(
            "key and value must have the same batch, head, and sequence sizes"
        )
    if query.size(-1) != key.size(-1):
        raise ValueError("query and key must have the same head dimension")

    head_dimension = query.size(-1)
    if head_dimension < 1:
        raise ValueError("head dimension must be positive")

    attention_scores = torch.matmul(
        query,
        key.transpose(-2, -1),
    ) / math.sqrt(head_dimension)

    expanded_mask: Tensor | None = None
    if attention_mask is not None:
        expanded_mask = _expand_attention_mask(
            attention_mask,
            attention_scores,
        )
        attention_scores = attention_scores.masked_fill(
            ~expanded_mask,
            torch.finfo(attention_scores.dtype).min,
        )

    attention_weights = torch.softmax(attention_scores, dim=-1)

    if expanded_mask is not None:
        # Softmax over a completely masked row would otherwise return a
        # uniform distribution. Zeroing and renormalizing makes that case safe.
        attention_weights = attention_weights.masked_fill(
            ~expanded_mask,
            0.0,
        )
        weight_sums = attention_weights.sum(dim=-1, keepdim=True)
        attention_weights = torch.where(
            weight_sums > 0,
            attention_weights / weight_sums.clamp_min(
                torch.finfo(attention_weights.dtype).eps
            ),
            torch.zeros_like(attention_weights),
        )

    probabilities_for_output = (
        dropout(attention_weights)
        if dropout is not None
        else attention_weights
    )
    attended_values = torch.matmul(probabilities_for_output, value)
    return attended_values, attention_weights


class MultiHeadAttention(nn.Module):
    """Run self-attention in parallel across several representation subspaces."""

    def __init__(
        self,
        model_dimension: int,
        number_of_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if model_dimension < 1:
            raise ValueError("model_dimension must be positive")
        if number_of_heads < 1:
            raise ValueError("number_of_heads must be positive")
        if model_dimension % number_of_heads != 0:
            raise ValueError(
                "model_dimension must be divisible by number_of_heads"
            )
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be at least 0 and less than 1")

        self.model_dimension = model_dimension
        self.number_of_heads = number_of_heads
        self.head_dimension = model_dimension // number_of_heads

        self.query_projection = nn.Linear(model_dimension, model_dimension)
        self.key_projection = nn.Linear(model_dimension, model_dimension)
        self.value_projection = nn.Linear(model_dimension, model_dimension)
        self.output_projection = nn.Linear(model_dimension, model_dimension)
        self.attention_dropout = nn.Dropout(dropout)

    def _split_heads(self, values: Tensor) -> Tensor:
        """Change a batch-first tensor into separate attention heads."""
        batch_size, sequence_length, _ = values.shape
        return (
            values.reshape(
                batch_size,
                sequence_length,
                self.number_of_heads,
                self.head_dimension,
            )
            .transpose(1, 2)
            .contiguous()
        )

    def _combine_heads(self, values: Tensor) -> Tensor:
        """Reverse ``_split_heads`` after every head has attended."""
        batch_size, _, sequence_length, _ = values.shape
        return (
            values.transpose(1, 2)
            .contiguous()
            .reshape(batch_size, sequence_length, self.model_dimension)
        )

    def forward(
        self,
        token_embeddings: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return contextualized embeddings and per-head attention weights."""
        if token_embeddings.ndim != 3:
            raise ValueError(
                "token_embeddings must have shape [batch, sequence, model]"
            )
        if token_embeddings.size(-1) != self.model_dimension:
            raise ValueError(
                f"Expected model dimension {self.model_dimension}, "
                f"received {token_embeddings.size(-1)}"
            )
        if (
            attention_mask is not None
            and attention_mask.ndim == 2
            and attention_mask.shape != token_embeddings.shape[:2]
        ):
            raise ValueError(
                "A 2D attention_mask must match [batch, sequence]"
            )

        query = self._split_heads(
            self.query_projection(token_embeddings)
        )
        key = self._split_heads(
            self.key_projection(token_embeddings)
        )
        value = self._split_heads(
            self.value_projection(token_embeddings)
        )

        attended_heads, attention_weights = scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attention_mask=attention_mask,
            dropout=self.attention_dropout,
        )
        combined_heads = self._combine_heads(attended_heads)
        return self.output_projection(combined_heads), attention_weights

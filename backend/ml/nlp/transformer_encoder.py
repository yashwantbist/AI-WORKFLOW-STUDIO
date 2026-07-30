"""A Transformer encoder assembled from the custom attention implementation."""

import math

import torch
from torch import Tensor, nn

try:
    # Supports: python -m backend.ml.nlp.visualize_attention
    from .attention import MultiHeadAttention
    from .transformer_model import SinusoidalPositionalEncoding
except ImportError:
    # Supports direct execution from backend/ml/nlp.
    from attention import MultiHeadAttention
    from transformer_model import SinusoidalPositionalEncoding


class FeedForwardNetwork(nn.Module):
    """Transform each token independently after tokens exchange information."""

    def __init__(
        self,
        model_dimension: int,
        feed_forward_dimension: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if model_dimension < 1:
            raise ValueError("model_dimension must be positive")
        if feed_forward_dimension < 1:
            raise ValueError("feed_forward_dimension must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be at least 0 and less than 1")

        self.network = nn.Sequential(
            nn.Linear(model_dimension, feed_forward_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feed_forward_dimension, model_dimension),
        )

    def forward(self, token_embeddings: Tensor) -> Tensor:
        return self.network(token_embeddings)


class EncoderBlock(nn.Module):
    """One post-LayerNorm Transformer encoder block.

    The two residual paths are:

    ``LayerNorm(x + MultiHeadAttention(x))``

    ``LayerNorm(x + FeedForward(x))``
    """

    def __init__(
        self,
        model_dimension: int,
        number_of_heads: int,
        feed_forward_dimension: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.self_attention = MultiHeadAttention(
            model_dimension=model_dimension,
            number_of_heads=number_of_heads,
            dropout=dropout,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.attention_layer_norm = nn.LayerNorm(model_dimension)

        self.feed_forward = FeedForwardNetwork(
            model_dimension=model_dimension,
            feed_forward_dimension=feed_forward_dimension,
            dropout=dropout,
        )
        self.feed_forward_dropout = nn.Dropout(dropout)
        self.feed_forward_layer_norm = nn.LayerNorm(model_dimension)

    def forward(
        self,
        token_embeddings: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return encoded tokens and this block's per-head attention weights."""
        attention_output, attention_weights = self.self_attention(
            token_embeddings,
            attention_mask,
        )
        token_embeddings = self.attention_layer_norm(
            token_embeddings + self.attention_dropout(attention_output)
        )

        feed_forward_output = self.feed_forward(token_embeddings)
        encoded_tokens = self.feed_forward_layer_norm(
            token_embeddings
            + self.feed_forward_dropout(feed_forward_output)
        )
        return encoded_tokens, attention_weights


class TransformerEncoder(nn.Module):
    """Stack several encoder blocks and expose each layer's attention."""

    def __init__(
        self,
        model_dimension: int,
        number_of_heads: int,
        number_of_layers: int,
        feed_forward_dimension: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if number_of_layers < 1:
            raise ValueError("number_of_layers must be positive")

        self.layers = nn.ModuleList(
            [
                EncoderBlock(
                    model_dimension=model_dimension,
                    number_of_heads=number_of_heads,
                    feed_forward_dimension=feed_forward_dimension,
                    dropout=dropout,
                )
                for _ in range(number_of_layers)
            ]
        )

    def forward(
        self,
        token_embeddings: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, tuple[Tensor, ...]]:
        """Pass tokens through every block and retain visualization weights."""
        attention_by_layer: list[Tensor] = []

        for layer in self.layers:
            token_embeddings, attention_weights = layer(
                token_embeddings,
                attention_mask,
            )
            attention_by_layer.append(attention_weights)

        return token_embeddings, tuple(attention_by_layer)


class AttentionTextClassifier(nn.Module):
    """Classify text with the hand-built Transformer encoder."""

    def __init__(
        self,
        vocabulary_size: int,
        number_of_classes: int,
        padding_id: int,
        model_dimension: int = 32,
        number_of_heads: int = 4,
        number_of_layers: int = 2,
        feed_forward_dimension: int = 64,
        dropout: float = 0.1,
        maximum_sequence_length: int = 128,
    ) -> None:
        super().__init__()

        if vocabulary_size < 2:
            raise ValueError("vocabulary_size must be at least 2")
        if number_of_classes < 2:
            raise ValueError("number_of_classes must be at least 2")

        self.embedding_scale = math.sqrt(model_dimension)
        self.embedding = nn.Embedding(
            vocabulary_size,
            model_dimension,
            padding_idx=padding_id,
        )
        self.positional_encoding = SinusoidalPositionalEncoding(
            model_dimension=model_dimension,
            maximum_sequence_length=maximum_sequence_length,
        )
        self.encoder = TransformerEncoder(
            model_dimension=model_dimension,
            number_of_heads=number_of_heads,
            number_of_layers=number_of_layers,
            feed_forward_dimension=feed_forward_dimension,
            dropout=dropout,
        )
        self.classification_head = nn.Sequential(
            nn.LayerNorm(model_dimension),
            nn.Dropout(dropout),
            nn.Linear(model_dimension, number_of_classes),
        )

    def forward(
        self,
        token_ids: Tensor,
        attention_mask: Tensor,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, tuple[Tensor, ...]]:
        """Return class logits and, when requested, attention weights."""
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")
        if attention_mask.shape != token_ids.shape:
            raise ValueError(
                "attention_mask must have the same shape as token_ids"
            )

        attention_mask = attention_mask.bool()
        token_embeddings = self.embedding(token_ids) * self.embedding_scale
        positioned_embeddings = self.positional_encoding(token_embeddings)
        encoded_tokens, attention_by_layer = self.encoder(
            positioned_embeddings,
            attention_mask,
        )

        expanded_mask = attention_mask.unsqueeze(-1).to(
            encoded_tokens.dtype
        )
        summed_tokens = (encoded_tokens * expanded_mask).sum(dim=1)
        token_counts = expanded_mask.sum(dim=1).clamp(min=1)
        sentence_representation = summed_tokens / token_counts
        logits = self.classification_head(sentence_representation)

        if return_attention:
            return logits, attention_by_layer
        return logits

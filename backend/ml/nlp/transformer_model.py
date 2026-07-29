"""Transformer encoder model for beginner-friendly text classification."""

import math

import torch
from torch import Tensor, nn


class SinusoidalPositionalEncoding(nn.Module):
    """Add a repeatable position signal to every token embedding.

    A Transformer sees all tokens at once. The sine and cosine values give
    position 0, position 1, and so on different numeric patterns, allowing the
    model to learn that word order matters.
    """

    def __init__(
        self,
        model_dimension: int,
        maximum_sequence_length: int,
    ) -> None:
        super().__init__()

        if model_dimension < 1:
            raise ValueError("model_dimension must be positive")
        if maximum_sequence_length < 1:
            raise ValueError("maximum_sequence_length must be positive")

        positions = torch.arange(
            maximum_sequence_length,
            dtype=torch.float32,
        ).unsqueeze(1)
        frequency_divisors = torch.exp(
            torch.arange(0, model_dimension, 2, dtype=torch.float32)
            * (-math.log(10_000.0) / model_dimension)
        )

        encoding = torch.zeros(
            maximum_sequence_length,
            model_dimension,
            dtype=torch.float32,
        )
        encoding[:, 0::2] = torch.sin(positions * frequency_divisors)
        odd_columns = encoding[:, 1::2].shape[1]
        encoding[:, 1::2] = torch.cos(
            positions * frequency_divisors[:odd_columns]
        )

        # Shape: [1, sequence_length, model_dimension]. The leading 1 lets
        # PyTorch broadcast the same positional pattern across every batch.
        self.register_buffer(
            "encoding",
            encoding.unsqueeze(0),
            persistent=False,
        )

    def forward(self, token_embeddings: Tensor) -> Tensor:
        """Return embeddings with position information added."""
        sequence_length = token_embeddings.size(1)
        if sequence_length > self.encoding.size(1):
            raise ValueError(
                f"Sequence length {sequence_length} exceeds the configured "
                f"maximum of {self.encoding.size(1)}"
            )

        position_signal = self.encoding[
            :,
            :sequence_length,
        ].to(dtype=token_embeddings.dtype)
        return token_embeddings + position_signal


class TransformerTextClassifier(nn.Module):
    """Classify text with embeddings, self-attention, and a linear head."""

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
        if model_dimension < 1:
            raise ValueError("model_dimension must be positive")
        if number_of_heads < 1:
            raise ValueError("number_of_heads must be positive")
        if model_dimension % number_of_heads != 0:
            raise ValueError(
                "model_dimension must be divisible by number_of_heads"
            )
        if number_of_layers < 1:
            raise ValueError("number_of_layers must be positive")
        if feed_forward_dimension < 1:
            raise ValueError("feed_forward_dimension must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be at least 0 and less than 1")

        self.embedding_scale = math.sqrt(model_dimension)
        self.embedding = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=model_dimension,
            padding_idx=padding_id,
        )
        self.positional_encoding = SinusoidalPositionalEncoding(
            model_dimension=model_dimension,
            maximum_sequence_length=maximum_sequence_length,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dimension,
            nhead=number_of_heads,
            dim_feedforward=feed_forward_dimension,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=number_of_layers,
            enable_nested_tensor=False,
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
    ) -> Tensor:
        """Return one class score (logit) per class for every sentence."""
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")
        if attention_mask.shape != token_ids.shape:
            raise ValueError(
                "attention_mask must have the same shape as token_ids"
            )

        attention_mask = attention_mask.bool()
        token_embeddings = self.embedding(token_ids) * self.embedding_scale
        positioned_embeddings = self.positional_encoding(token_embeddings)

        # Our mask uses True for real tokens. PyTorch's key-padding mask uses
        # True for positions that must be ignored, so the mask is inverted.
        encoded_tokens = self.encoder(
            positioned_embeddings,
            src_key_padding_mask=~attention_mask,
        )

        # Convert the variable-length token sequence into one sentence vector.
        expanded_mask = attention_mask.unsqueeze(-1).to(
            encoded_tokens.dtype
        )
        summed_tokens = (encoded_tokens * expanded_mask).sum(dim=1)
        real_token_counts = expanded_mask.sum(dim=1).clamp(min=1)
        sentence_representation = summed_tokens / real_token_counts

        return self.classification_head(sentence_representation)
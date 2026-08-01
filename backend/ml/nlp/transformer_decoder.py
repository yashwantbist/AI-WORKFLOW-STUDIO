"""A decoder-only Transformer that cannot attend to future tokens."""

import math

import torch
from torch import Tensor, nn

try:
    # Supports package imports from the repository root.
    from .attention import MultiHeadAttention
    from .masks import combine_causal_and_padding_masks
    from .transformer_encoder import FeedForwardNetwork
    from .transformer_model import SinusoidalPositionalEncoding
except ImportError:
    # Supports direct execution from backend/ml/nlp.
    from attention import MultiHeadAttention
    from masks import combine_causal_and_padding_masks
    from transformer_encoder import FeedForwardNetwork
    from transformer_model import SinusoidalPositionalEncoding


class DecoderBlock(nn.Module):
    """One GPT-style decoder-only block with causal self-attention.

    This educational block keeps Day 10's post-LayerNorm order:

    ``LayerNorm(x + CausalSelfAttention(x))``

    ``LayerNorm(x + FeedForward(x))``

    An encoder-decoder model such as T5 would add a cross-attention sublayer.
    That extra sublayer is intentionally outside this decoder-only lab.
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
        padding_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return decoded tokens and per-head causal attention weights."""
        if token_embeddings.ndim != 3:
            raise ValueError(
                "token_embeddings must have shape [batch, sequence, model]"
            )

        batch_size, sequence_length, _ = token_embeddings.shape
        if padding_mask is None:
            padding_mask = torch.ones(
                batch_size,
                sequence_length,
                dtype=torch.bool,
                device=token_embeddings.device,
            )
        elif padding_mask.shape != token_embeddings.shape[:2]:
            raise ValueError(
                "padding_mask must have shape [batch, sequence]"
            )
        else:
            padding_mask = padding_mask.to(
                device=token_embeddings.device,
                dtype=torch.bool,
            )

        decoder_attention_mask = combine_causal_and_padding_masks(
            padding_mask
        )
        attention_output, attention_weights = self.self_attention(
            token_embeddings,
            decoder_attention_mask,
        )
        token_embeddings = self.attention_layer_norm(
            token_embeddings + self.attention_dropout(attention_output)
        )

        feed_forward_output = self.feed_forward(token_embeddings)
        decoded_tokens = self.feed_forward_layer_norm(
            token_embeddings
            + self.feed_forward_dropout(feed_forward_output)
        )

        # Padded query rows have no useful output. Setting them to zero makes
        # their meaning explicit and prevents accidental downstream use.
        decoded_tokens = decoded_tokens * padding_mask.unsqueeze(-1).to(
            decoded_tokens.dtype
        )
        return decoded_tokens, attention_weights


class TransformerDecoder(nn.Module):
    """Stack decoder-only blocks and retain each layer's attention maps."""

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
                DecoderBlock(
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
        padding_mask: Tensor | None = None,
    ) -> tuple[Tensor, tuple[Tensor, ...]]:
        """Pass tokens through every causal decoder block."""
        attention_by_layer: list[Tensor] = []

        for layer in self.layers:
            token_embeddings, attention_weights = layer(
                token_embeddings,
                padding_mask,
            )
            attention_by_layer.append(attention_weights)

        return token_embeddings, tuple(attention_by_layer)


class CausalLanguageModel(nn.Module):
    """Predict the next-token score at every sequence position."""

    def __init__(
        self,
        vocabulary_size: int,
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
        if not 0 <= padding_id < vocabulary_size:
            raise ValueError("padding_id must be inside the vocabulary")

        self.padding_id = padding_id
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
        self.decoder = TransformerDecoder(
            model_dimension=model_dimension,
            number_of_heads=number_of_heads,
            number_of_layers=number_of_layers,
            feed_forward_dimension=feed_forward_dimension,
            dropout=dropout,
        )
        self.final_layer_norm = nn.LayerNorm(model_dimension)
        self.output_projection = nn.Linear(
            model_dimension,
            vocabulary_size,
            bias=False,
        )

    def forward(
        self,
        token_ids: Tensor,
        attention_mask: Tensor | None = None,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, tuple[Tensor, ...]]:
        """Return next-token logits and optional attention maps."""
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")

        if attention_mask is None:
            attention_mask = token_ids.ne(self.padding_id)
        elif attention_mask.shape != token_ids.shape:
            raise ValueError(
                "attention_mask must have the same shape as token_ids"
            )
        else:
            attention_mask = attention_mask.bool()

        token_embeddings = self.embedding(token_ids) * self.embedding_scale
        positioned_embeddings = self.positional_encoding(token_embeddings)
        decoded_tokens, attention_by_layer = self.decoder(
            positioned_embeddings,
            attention_mask,
        )
        logits = self.output_projection(
            self.final_layer_norm(decoded_tokens)
        )

        if return_attention:
            return logits, attention_by_layer
        return logits

"""A beginner-friendly encoder-decoder Transformer with cross-attention."""

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

try:
    # Supports package imports from the repository root.
    from .attention import MultiHeadAttention
    from .cross_attention import CrossAttention
    from .masks import combine_causal_and_padding_masks
    from .transformer_encoder import (
        FeedForwardNetwork,
        TransformerEncoder,
    )
    from .transformer_model import SinusoidalPositionalEncoding
except ImportError:
    # Supports direct execution from backend/ml/nlp.
    from attention import MultiHeadAttention
    from cross_attention import CrossAttention
    from masks import combine_causal_and_padding_masks
    from transformer_encoder import FeedForwardNetwork, TransformerEncoder
    from transformer_model import SinusoidalPositionalEncoding


@dataclass(frozen=True)
class EncoderOutput:
    """Information calculated once and reused throughout decoding."""

    hidden_states: Tensor
    attention_mask: Tensor
    self_attention: tuple[Tensor, ...]


@dataclass(frozen=True)
class Seq2SeqAttentionMaps:
    """Attention maps from all three attention paths."""

    encoder_self_attention: tuple[Tensor, ...]
    decoder_self_attention: tuple[Tensor, ...]
    cross_attention: tuple[Tensor, ...]


class Seq2SeqDecoderBlock(nn.Module):
    """Decoder block with causal self-attention and cross-attention."""

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
        self.self_attention_dropout = nn.Dropout(dropout)
        self.self_attention_layer_norm = nn.LayerNorm(model_dimension)

        self.cross_attention = CrossAttention(
            model_dimension=model_dimension,
            number_of_heads=number_of_heads,
            dropout=dropout,
        )
        self.cross_attention_dropout = nn.Dropout(dropout)
        self.cross_attention_layer_norm = nn.LayerNorm(model_dimension)

        self.feed_forward = FeedForwardNetwork(
            model_dimension=model_dimension,
            feed_forward_dimension=feed_forward_dimension,
            dropout=dropout,
        )
        self.feed_forward_dropout = nn.Dropout(dropout)
        self.feed_forward_layer_norm = nn.LayerNorm(model_dimension)

    def forward(
        self,
        decoder_states: Tensor,
        encoder_output: EncoderOutput,
        target_attention_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Run self-attention, cross-attention, and the token-wise FFN."""
        if decoder_states.ndim != 3:
            raise ValueError(
                "decoder_states must have shape [batch, sequence, model]"
            )
        if target_attention_mask.shape != decoder_states.shape[:2]:
            raise ValueError(
                "target_attention_mask must match [batch, target_sequence]"
            )

        target_attention_mask = target_attention_mask.to(
            device=decoder_states.device,
            dtype=torch.bool,
        )
        causal_mask = combine_causal_and_padding_masks(
            target_attention_mask
        )
        self_attention_output, self_attention_weights = self.self_attention(
            decoder_states,
            causal_mask,
        )
        decoder_states = self.self_attention_layer_norm(
            decoder_states
            + self.self_attention_dropout(self_attention_output)
        )

        cross_attention_output, cross_attention_weights = (
            self.cross_attention(
                decoder_states=decoder_states,
                encoder_states=encoder_output.hidden_states,
                encoder_attention_mask=encoder_output.attention_mask,
            )
        )
        decoder_states = self.cross_attention_layer_norm(
            decoder_states
            + self.cross_attention_dropout(cross_attention_output)
        )

        feed_forward_output = self.feed_forward(decoder_states)
        decoder_states = self.feed_forward_layer_norm(
            decoder_states
            + self.feed_forward_dropout(feed_forward_output)
        )
        decoder_states = decoder_states * target_attention_mask.unsqueeze(
            -1
        ).to(decoder_states.dtype)
        return (
            decoder_states,
            self_attention_weights,
            cross_attention_weights,
        )


class Seq2SeqDecoder(nn.Module):
    """Stack encoder-decoder blocks and collect their attention maps."""

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
                Seq2SeqDecoderBlock(
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
        decoder_states: Tensor,
        encoder_output: EncoderOutput,
        target_attention_mask: Tensor,
    ) -> tuple[Tensor, tuple[Tensor, ...], tuple[Tensor, ...]]:
        """Pass target states through every encoder-decoder block."""
        self_attention_by_layer: list[Tensor] = []
        cross_attention_by_layer: list[Tensor] = []

        for layer in self.layers:
            (
                decoder_states,
                self_attention_weights,
                cross_attention_weights,
            ) = layer(
                decoder_states,
                encoder_output,
                target_attention_mask,
            )
            self_attention_by_layer.append(self_attention_weights)
            cross_attention_by_layer.append(cross_attention_weights)

        return (
            decoder_states,
            tuple(self_attention_by_layer),
            tuple(cross_attention_by_layer),
        )


class Seq2SeqTransformer(nn.Module):
    """Translate a source sequence into target-token logits."""

    def __init__(
        self,
        source_vocabulary_size: int,
        target_vocabulary_size: int,
        source_padding_id: int,
        target_padding_id: int,
        model_dimension: int = 32,
        number_of_heads: int = 4,
        number_of_encoder_layers: int = 2,
        number_of_decoder_layers: int = 2,
        feed_forward_dimension: int = 64,
        dropout: float = 0.1,
        maximum_sequence_length: int = 128,
    ) -> None:
        super().__init__()

        if source_vocabulary_size < 2 or target_vocabulary_size < 3:
            raise ValueError("Vocabulary sizes are too small")
        if not 0 <= source_padding_id < source_vocabulary_size:
            raise ValueError("source_padding_id is outside its vocabulary")
        if not 0 <= target_padding_id < target_vocabulary_size:
            raise ValueError("target_padding_id is outside its vocabulary")

        self.source_padding_id = source_padding_id
        self.target_padding_id = target_padding_id
        self.embedding_scale = math.sqrt(model_dimension)

        self.source_embedding = nn.Embedding(
            source_vocabulary_size,
            model_dimension,
            padding_idx=source_padding_id,
        )
        self.target_embedding = nn.Embedding(
            target_vocabulary_size,
            model_dimension,
            padding_idx=target_padding_id,
        )
        self.source_positional_encoding = SinusoidalPositionalEncoding(
            model_dimension=model_dimension,
            maximum_sequence_length=maximum_sequence_length,
        )
        self.target_positional_encoding = SinusoidalPositionalEncoding(
            model_dimension=model_dimension,
            maximum_sequence_length=maximum_sequence_length,
        )
        self.encoder = TransformerEncoder(
            model_dimension=model_dimension,
            number_of_heads=number_of_heads,
            number_of_layers=number_of_encoder_layers,
            feed_forward_dimension=feed_forward_dimension,
            dropout=dropout,
        )
        self.decoder = Seq2SeqDecoder(
            model_dimension=model_dimension,
            number_of_heads=number_of_heads,
            number_of_layers=number_of_decoder_layers,
            feed_forward_dimension=feed_forward_dimension,
            dropout=dropout,
        )
        self.final_layer_norm = nn.LayerNorm(model_dimension)
        self.output_projection = nn.Linear(
            model_dimension,
            target_vocabulary_size,
            bias=False,
        )

    def encode(
        self,
        source_token_ids: Tensor,
        source_attention_mask: Tensor | None = None,
    ) -> EncoderOutput:
        """Read the complete source once and return reusable representations."""
        if source_token_ids.ndim != 2:
            raise ValueError(
                "source_token_ids must have shape [batch, source_sequence]"
            )
        if source_attention_mask is None:
            source_attention_mask = source_token_ids.ne(
                self.source_padding_id
            )
        elif source_attention_mask.shape != source_token_ids.shape:
            raise ValueError(
                "source_attention_mask must match source_token_ids"
            )
        else:
            source_attention_mask = source_attention_mask.to(
                device=source_token_ids.device,
                dtype=torch.bool,
            )

        source_embeddings = (
            self.source_embedding(source_token_ids) * self.embedding_scale
        )
        positioned_source = self.source_positional_encoding(
            source_embeddings
        )
        hidden_states, self_attention = self.encoder(
            positioned_source,
            source_attention_mask,
        )
        hidden_states = hidden_states * source_attention_mask.unsqueeze(
            -1
        ).to(hidden_states.dtype)
        return EncoderOutput(
            hidden_states=hidden_states,
            attention_mask=source_attention_mask,
            self_attention=self_attention,
        )

    def decode(
        self,
        target_token_ids: Tensor,
        encoder_output: EncoderOutput,
        target_attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, tuple[Tensor, ...], tuple[Tensor, ...]]:
        """Generate target representations while referencing the source."""
        if target_token_ids.ndim != 2:
            raise ValueError(
                "target_token_ids must have shape [batch, target_sequence]"
            )
        if target_token_ids.size(0) != encoder_output.hidden_states.size(0):
            raise ValueError("Source and target batch sizes must match")
        if target_attention_mask is None:
            target_attention_mask = target_token_ids.ne(
                self.target_padding_id
            )
        elif target_attention_mask.shape != target_token_ids.shape:
            raise ValueError(
                "target_attention_mask must match target_token_ids"
            )
        else:
            target_attention_mask = target_attention_mask.to(
                device=target_token_ids.device,
                dtype=torch.bool,
            )

        target_embeddings = (
            self.target_embedding(target_token_ids) * self.embedding_scale
        )
        positioned_target = self.target_positional_encoding(
            target_embeddings
        )
        (
            decoder_states,
            decoder_self_attention,
            cross_attention,
        ) = self.decoder(
            positioned_target,
            encoder_output,
            target_attention_mask,
        )
        logits = self.output_projection(
            self.final_layer_norm(decoder_states)
        )
        return logits, decoder_self_attention, cross_attention

    def forward(
        self,
        source_token_ids: Tensor,
        target_token_ids: Tensor,
        source_attention_mask: Tensor | None = None,
        target_attention_mask: Tensor | None = None,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, Seq2SeqAttentionMaps]:
        """Encode the source, decode the target, and return target logits."""
        encoder_output = self.encode(
            source_token_ids,
            source_attention_mask,
        )
        logits, decoder_self_attention, cross_attention = self.decode(
            target_token_ids,
            encoder_output,
            target_attention_mask,
        )

        if return_attention:
            return logits, Seq2SeqAttentionMaps(
                encoder_self_attention=encoder_output.self_attention,
                decoder_self_attention=decoder_self_attention,
                cross_attention=cross_attention,
            )
        return logits


def greedy_decode(
    model: Seq2SeqTransformer,
    source_token_ids: Tensor,
    beginning_of_sequence_id: int,
    end_of_sequence_id: int,
    maximum_new_tokens: int,
    source_attention_mask: Tensor | None = None,
) -> Tensor:
    """Generate target IDs one position at a time for demonstration."""
    if maximum_new_tokens < 1:
        raise ValueError("maximum_new_tokens must be positive")

    model.eval()
    batch_size = source_token_ids.size(0)
    generated_ids = torch.full(
        (batch_size, 1),
        beginning_of_sequence_id,
        dtype=torch.long,
        device=source_token_ids.device,
    )
    finished = torch.zeros(
        batch_size,
        dtype=torch.bool,
        device=source_token_ids.device,
    )

    with torch.inference_mode():
        encoder_output = model.encode(
            source_token_ids,
            source_attention_mask,
        )
        for _ in range(maximum_new_tokens):
            logits, _, _ = model.decode(
                generated_ids,
                encoder_output,
            )
            next_token_ids = logits[:, -1].argmax(dim=-1)
            next_token_ids = torch.where(
                finished,
                torch.full_like(next_token_ids, model.target_padding_id),
                next_token_ids,
            )
            generated_ids = torch.cat(
                [generated_ids, next_token_ids.unsqueeze(1)],
                dim=1,
            )
            finished = finished | next_token_ids.eq(end_of_sequence_id)
            if finished.all():
                break

    return generated_ids

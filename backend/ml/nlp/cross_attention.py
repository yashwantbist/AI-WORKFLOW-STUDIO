"""Multi-head attention from decoder queries to encoder representations."""

from torch import Tensor, nn

try:
    # Supports package imports from the repository root.
    from .attention import scaled_dot_product_attention
except ImportError:
    # Supports direct execution from backend/ml/nlp.
    from attention import scaled_dot_product_attention


class CrossAttention(nn.Module):
    """Let decoder positions retrieve information from encoder positions.

    Self-attention creates query, key, and value from the same sequence.
    Cross-attention creates query from the decoder but key and value from the
    encoder. This is how generated tokens can repeatedly reference the source.
    """

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
        """Split the model dimension into parallel attention heads."""
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
        """Join parallel attention heads back into one representation."""
        batch_size, _, sequence_length, _ = values.shape
        return (
            values.transpose(1, 2)
            .contiguous()
            .reshape(batch_size, sequence_length, self.model_dimension)
        )

    def forward(
        self,
        decoder_states: Tensor,
        encoder_states: Tensor,
        encoder_attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return source-aware decoder states and cross-attention weights.

        Shapes:

        - decoder states: ``[batch, target_sequence, model_dimension]``
        - encoder states: ``[batch, source_sequence, model_dimension]``
        - weights: ``[batch, heads, target_sequence, source_sequence]``
        """
        if decoder_states.ndim != 3 or encoder_states.ndim != 3:
            raise ValueError(
                "decoder_states and encoder_states must be 3D tensors"
            )
        if decoder_states.size(0) != encoder_states.size(0):
            raise ValueError(
                "decoder_states and encoder_states need the same batch size"
            )
        if decoder_states.size(-1) != self.model_dimension:
            raise ValueError("decoder_states has the wrong model dimension")
        if encoder_states.size(-1) != self.model_dimension:
            raise ValueError("encoder_states has the wrong model dimension")
        if (
            encoder_attention_mask is not None
            and encoder_attention_mask.shape != encoder_states.shape[:2]
        ):
            raise ValueError(
                "encoder_attention_mask must match [batch, source_sequence]"
            )

        query = self._split_heads(
            self.query_projection(decoder_states)
        )
        key = self._split_heads(
            self.key_projection(encoder_states)
        )
        value = self._split_heads(
            self.value_projection(encoder_states)
        )
        attended_heads, attention_weights = scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attention_mask=encoder_attention_mask,
            dropout=self.attention_dropout,
        )
        combined_heads = self._combine_heads(attended_heads)
        return self.output_projection(combined_heads), attention_weights

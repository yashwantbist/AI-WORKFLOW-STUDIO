"""Embedding-based neural network for text classification."""

from torch import Tensor, nn


class MeanPoolingTextClassifier(nn.Module):
    """Classify text using embeddings, mean pooling, and a linear layer."""

    def __init__(
        self,
        vocabulary_size: int,
        embedding_size: int,
        number_of_classes: int,
        padding_id: int,
    ) -> None:
        super().__init__()

        if vocabulary_size < 2:
            raise ValueError("vocabulary_size must be at least 2")
        if embedding_size < 1:
            raise ValueError("embedding_size must be positive")
        if number_of_classes < 2:
            raise ValueError("number_of_classes must be at least 2")

        self.embedding = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=embedding_size,
            padding_idx=padding_id,
        )
        self.classifier = nn.Linear(
            embedding_size,
            number_of_classes,
        )

    def forward(
        self,
        token_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        """Return class logits for a padded batch of token IDs."""
        embedded_tokens = self.embedding(token_ids)
        expanded_mask = attention_mask.unsqueeze(-1).to(
            embedded_tokens.dtype
        )

        summed_embeddings = (embedded_tokens * expanded_mask).sum(dim=1)
        real_token_counts = expanded_mask.sum(dim=1).clamp(min=1)
        mean_embeddings = summed_embeddings / real_token_counts

        return self.classifier(mean_embeddings)
"""Tests for the Transformer text-classification pipeline."""

import pytest
import torch
from torch import nn

from backend.ml.nlp.dataset import build_data_bundle
from backend.ml.nlp.train_transformer_classifier import (
    run_training_experiment,
)
from backend.ml.nlp.transformer_model import (
    SinusoidalPositionalEncoding,
    TransformerTextClassifier,
)


def create_small_transformer(dropout: float = 0.0) -> TransformerTextClassifier:
    """Create a lightweight model used by unit tests."""
    return TransformerTextClassifier(
        vocabulary_size=20,
        number_of_classes=2,
        padding_id=0,
        model_dimension=16,
        number_of_heads=4,
        number_of_layers=1,
        feed_forward_dimension=32,
        dropout=dropout,
        maximum_sequence_length=16,
    )


def test_positional_encoding_changes_positions_not_shape() -> None:
    positional_encoding = SinusoidalPositionalEncoding(
        model_dimension=8,
        maximum_sequence_length=10,
    )
    embeddings = torch.zeros(2, 4, 8)

    positioned_embeddings = positional_encoding(embeddings)

    assert positioned_embeddings.shape == embeddings.shape
    assert not torch.allclose(
        positioned_embeddings[:, 0, :],
        positioned_embeddings[:, 1, :],
    )


def test_positional_encoding_rejects_overlong_sequence() -> None:
    positional_encoding = SinusoidalPositionalEncoding(
        model_dimension=8,
        maximum_sequence_length=3,
    )

    with pytest.raises(ValueError, match="exceeds"):
        positional_encoding(torch.zeros(1, 4, 8))


def test_transformer_requires_dimension_divisible_by_heads() -> None:
    with pytest.raises(ValueError, match="divisible"):
        TransformerTextClassifier(
            vocabulary_size=20,
            number_of_classes=2,
            padding_id=0,
            model_dimension=10,
            number_of_heads=4,
        )


def test_transformer_returns_one_logit_per_class() -> None:
    model = create_small_transformer()
    token_ids = torch.tensor(
        [
            [2, 3, 4],
            [5, 6, 0],
        ]
    )
    attention_mask = token_ids.ne(0)

    logits = model(token_ids, attention_mask)

    assert logits.shape == (2, 2)


def test_transformer_ignores_padding_tokens() -> None:
    model = create_small_transformer()
    model.eval()
    short_tokens = torch.tensor([[2, 3]])
    padded_tokens = torch.tensor([[2, 3, 0, 0]])

    with torch.inference_mode():
        short_logits = model(short_tokens, short_tokens.ne(0))
        padded_logits = model(padded_tokens, padded_tokens.ne(0))

    assert torch.allclose(short_logits, padded_logits, atol=1e-6)


def test_transformer_encoder_receives_gradients() -> None:
    model = create_small_transformer()
    token_ids = torch.tensor([[2, 3, 4], [5, 6, 7]])
    labels = torch.tensor([1, 0])

    loss = nn.CrossEntropyLoss()(
        model(token_ids, token_ids.ne(0)),
        labels,
    )
    loss.backward()

    assert any(
        parameter.grad is not None
        for parameter in model.encoder.parameters()
    )


def test_training_runner_records_real_metrics() -> None:
    data_bundle = build_data_bundle(
        samples_per_class=10,
        random_seed=42,
    )

    output = run_training_experiment(
        model_name="test Transformer",
        model_factory=lambda: TransformerTextClassifier(
            vocabulary_size=len(data_bundle.vocabulary),
            number_of_classes=2,
            padding_id=data_bundle.vocabulary.padding_id,
            model_dimension=16,
            number_of_heads=4,
            number_of_layers=1,
            feed_forward_dimension=32,
            dropout=0.0,
            maximum_sequence_length=16,
        ),
        data_bundle=data_bundle,
        device=torch.device("cpu"),
        epochs=1,
        batch_size=4,
        learning_rate=0.01,
        print_epochs=False,
    )

    assert len(output.result.history) == 1
    assert output.result.final_training_loss > 0
    assert 0 <= output.result.final_training_accuracy <= 100
    assert 0 <= output.result.final_validation_accuracy <= 100
    assert output.result.training_time_seconds > 0
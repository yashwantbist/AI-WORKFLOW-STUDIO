"""Unit tests for the hand-built Transformer encoder."""

import math

import pytest
import torch
from torch import nn

from backend.ml.nlp.attention import (
    MultiHeadAttention,
    scaled_dot_product_attention,
)
from backend.ml.nlp.transformer_encoder import (
    AttentionTextClassifier,
    EncoderBlock,
)


def test_scaled_dot_product_attention_matches_manual_result() -> None:
    query = torch.tensor([[[[1.0, 0.0]]]])
    key = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    value = torch.tensor([[[[2.0, 0.0], [0.0, 4.0]]]])

    output, weights = scaled_dot_product_attention(query, key, value)

    expected_weights = torch.softmax(
        torch.tensor([1 / math.sqrt(2), 0.0]),
        dim=0,
    )
    expected_output = torch.tensor(
        [
            expected_weights[0] * 2,
            expected_weights[1] * 4,
        ]
    )
    assert torch.allclose(weights[0, 0, 0], expected_weights)
    assert torch.allclose(output[0, 0, 0], expected_output)


def test_attention_mask_removes_padding_probability() -> None:
    query = torch.ones(1, 2, 3, 4)
    key = torch.ones(1, 2, 3, 4)
    value = torch.randn(1, 2, 3, 4)
    attention_mask = torch.tensor([[True, True, False]])

    _, weights = scaled_dot_product_attention(
        query,
        key,
        value,
        attention_mask,
    )

    assert torch.equal(weights[..., 2], torch.zeros_like(weights[..., 2]))
    assert torch.allclose(
        weights.sum(dim=-1),
        torch.ones_like(weights.sum(dim=-1)),
    )


def test_fully_masked_attention_is_finite_and_zero() -> None:
    query = torch.randn(1, 1, 2, 4)
    key = torch.randn(1, 1, 2, 4)
    value = torch.randn(1, 1, 2, 4)

    output, weights = scaled_dot_product_attention(
        query,
        key,
        value,
        torch.tensor([[False, False]]),
    )

    assert torch.isfinite(output).all()
    assert torch.equal(weights, torch.zeros_like(weights))
    assert torch.equal(output, torch.zeros_like(output))


def test_multi_head_attention_returns_expected_shapes() -> None:
    attention = MultiHeadAttention(
        model_dimension=16,
        number_of_heads=4,
        dropout=0.0,
    )
    embeddings = torch.randn(2, 5, 16)

    output, weights = attention(
        embeddings,
        torch.ones(2, 5, dtype=torch.bool),
    )

    assert output.shape == (2, 5, 16)
    assert weights.shape == (2, 4, 5, 5)


def test_multi_head_attention_rejects_incompatible_dimensions() -> None:
    with pytest.raises(ValueError, match="divisible"):
        MultiHeadAttention(
            model_dimension=10,
            number_of_heads=4,
        )


def test_encoder_block_uses_both_residual_paths() -> None:
    block = EncoderBlock(
        model_dimension=8,
        number_of_heads=2,
        feed_forward_dimension=16,
        dropout=0.0,
    )
    embeddings = torch.randn(2, 4, 8)

    for parameter in block.self_attention.parameters():
        nn.init.zeros_(parameter)
    for parameter in block.feed_forward.parameters():
        nn.init.zeros_(parameter)

    output, _ = block(
        embeddings,
        torch.ones(2, 4, dtype=torch.bool),
    )
    expected = block.feed_forward_layer_norm(
        block.attention_layer_norm(embeddings)
    )

    assert torch.allclose(output, expected, atol=1e-6)


def test_encoder_block_receives_gradients() -> None:
    block = EncoderBlock(
        model_dimension=8,
        number_of_heads=2,
        feed_forward_dimension=16,
        dropout=0.0,
    )
    embeddings = torch.randn(2, 4, 8, requires_grad=True)

    output, _ = block(
        embeddings,
        torch.ones(2, 4, dtype=torch.bool),
    )
    output.square().mean().backward()

    assert embeddings.grad is not None
    assert any(
        parameter.grad is not None
        for parameter in block.self_attention.parameters()
    )
    assert any(
        parameter.grad is not None
        for parameter in block.feed_forward.parameters()
    )


def create_tiny_classifier() -> AttentionTextClassifier:
    return AttentionTextClassifier(
        vocabulary_size=24,
        number_of_classes=2,
        padding_id=0,
        model_dimension=8,
        number_of_heads=2,
        number_of_layers=1,
        feed_forward_dimension=16,
        dropout=0.0,
        maximum_sequence_length=8,
    )


def test_classifier_returns_logits_and_attention_by_layer() -> None:
    model = create_tiny_classifier()
    token_ids = torch.tensor([[2, 3, 4], [5, 6, 0]])
    attention_mask = token_ids.ne(0)

    logits, attention_by_layer = model(
        token_ids,
        attention_mask,
        return_attention=True,
    )

    assert logits.shape == (2, 2)
    assert len(attention_by_layer) == 1
    assert attention_by_layer[0].shape == (2, 2, 3, 3)
    assert torch.equal(
        attention_by_layer[0][1, :, :, 2],
        torch.zeros_like(attention_by_layer[0][1, :, :, 2]),
    )


def test_classifier_completes_training_and_reduces_loss() -> None:
    torch.manual_seed(42)
    model = create_tiny_classifier()
    token_ids = torch.tensor(
        [
            [2, 3, 4],
            [2, 4, 3],
            [5, 6, 7],
            [5, 7, 6],
        ]
    )
    attention_mask = token_ids.ne(0)
    labels = torch.tensor([0, 0, 1, 1])
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

    model.eval()
    with torch.no_grad():
        initial_loss = loss_function(
            model(token_ids, attention_mask),
            labels,
        ).item()

    model.train()
    for _ in range(30):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(
            model(token_ids, attention_mask),
            labels,
        )
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        final_logits = model(token_ids, attention_mask)
        final_loss = loss_function(final_logits, labels).item()

    assert final_loss < initial_loss
    assert torch.equal(final_logits.argmax(dim=1), labels)

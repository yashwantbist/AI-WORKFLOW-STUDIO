"""Tests that prove decoder-only attention cannot use future tokens."""

import pytest
import torch
from torch import nn

from backend.ml.nlp.masks import (
    combine_causal_and_padding_masks,
    create_causal_attention_mask,
)
from backend.ml.nlp.transformer_decoder import (
    CausalLanguageModel,
    DecoderBlock,
)
from backend.ml.nlp.transformer_utils import (
    assert_no_future_attention,
    calculate_next_token_loss,
    has_no_future_attention,
    prepare_next_token_batch,
)


def test_five_token_causal_mask_matches_expected_triangle() -> None:
    expected = torch.tensor(
        [
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )

    actual = create_causal_attention_mask(5)

    assert torch.equal(actual, expected)


def test_causal_mask_rejects_nonpositive_length() -> None:
    with pytest.raises(ValueError, match="positive"):
        create_causal_attention_mask(0)


def test_combined_mask_blocks_future_and_padding_positions() -> None:
    padding_mask = torch.tensor(
        [
            [True, True, True, False],
            [True, True, True, True],
        ]
    )

    combined = combine_causal_and_padding_masks(padding_mask)

    assert combined.shape == (2, 4, 4)
    assert not combined[0, :, 3].any()
    assert not combined[0, 3, :].any()
    assert not torch.triu(combined, diagonal=1).any()


def test_decoder_block_assigns_zero_probability_to_future() -> None:
    torch.manual_seed(42)
    block = DecoderBlock(
        model_dimension=16,
        number_of_heads=4,
        feed_forward_dimension=32,
        dropout=0.0,
    )
    embeddings = torch.randn(2, 5, 16)

    output, attention_weights = block(embeddings)

    assert output.shape == embeddings.shape
    assert attention_weights.shape == (2, 4, 5, 5)
    assert has_no_future_attention(attention_weights)
    assert torch.count_nonzero(
        torch.triu(attention_weights, diagonal=1)
    ).item() == 0


def create_tiny_language_model() -> CausalLanguageModel:
    return CausalLanguageModel(
        vocabulary_size=24,
        padding_id=0,
        model_dimension=16,
        number_of_heads=4,
        number_of_layers=2,
        feed_forward_dimension=32,
        dropout=0.0,
        maximum_sequence_length=8,
    )


def test_language_model_returns_logits_and_attention_maps() -> None:
    model = create_tiny_language_model()
    token_ids = torch.tensor([[2, 3, 4, 5], [6, 7, 0, 0]])

    logits, attention_by_layer = model(
        token_ids,
        return_attention=True,
    )

    assert logits.shape == (2, 4, 24)
    assert len(attention_by_layer) == 2
    assert attention_by_layer[0].shape == (2, 4, 4, 4)
    assert all(
        has_no_future_attention(weights)
        for weights in attention_by_layer
    )
    assert not attention_by_layer[0][1, :, :, 2:].any()


def test_changing_future_tokens_cannot_change_earlier_logits() -> None:
    torch.manual_seed(42)
    model = create_tiny_language_model()
    model.eval()
    first_sequence = torch.tensor([[2, 3, 4, 5, 6]])
    changed_future = torch.tensor([[2, 3, 4, 9, 10]])

    with torch.inference_mode():
        first_logits = model(first_sequence)
        changed_logits = model(changed_future)

    # Positions 0, 1, and 2 receive identical history in both sequences.
    assert torch.allclose(
        first_logits[:, :3],
        changed_logits[:, :3],
        atol=1e-6,
    )


def test_next_token_utilities_shift_and_calculate_loss() -> None:
    token_ids = torch.tensor(
        [
            [2, 3, 4, 0],
            [5, 6, 7, 8],
        ]
    )
    input_ids, target_ids, input_mask, target_mask = (
        prepare_next_token_batch(token_ids, padding_id=0)
    )
    logits = torch.randn(2, 3, 12, requires_grad=True)

    loss = calculate_next_token_loss(
        logits,
        target_ids,
        target_mask,
    )
    loss.backward()

    assert torch.equal(input_ids, token_ids[:, :-1])
    assert torch.equal(target_ids, token_ids[:, 1:])
    assert torch.equal(input_mask, input_ids.ne(0))
    assert torch.equal(target_mask, target_ids.ne(0))
    assert loss.item() > 0
    assert logits.grad is not None


def test_verification_helper_detects_future_attention() -> None:
    invalid_weights = torch.eye(4)
    invalid_weights[0, 3] = 0.25

    with pytest.raises(AssertionError, match="Future-token"):
        assert_no_future_attention(invalid_weights)


def test_decoder_block_receives_gradients() -> None:
    block = DecoderBlock(
        model_dimension=8,
        number_of_heads=2,
        feed_forward_dimension=16,
        dropout=0.0,
    )
    embeddings = torch.randn(2, 4, 8, requires_grad=True)

    output, _ = block(embeddings)
    nn.functional.mse_loss(output, torch.zeros_like(output)).backward()

    assert embeddings.grad is not None
    assert any(
        parameter.grad is not None
        for parameter in block.parameters()
    )

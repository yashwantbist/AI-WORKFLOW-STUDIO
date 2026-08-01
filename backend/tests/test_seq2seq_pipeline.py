"""Tests for encoder-decoder communication and cross-attention flow."""

import pytest
import torch

from backend.ml.nlp.cross_attention import CrossAttention
from backend.ml.nlp.seq2seq_transformer import (
    EncoderOutput,
    Seq2SeqTransformer,
    greedy_decode,
)
from backend.ml.nlp.transformer_utils import (
    calculate_next_token_loss,
    has_no_future_attention,
)


def test_cross_attention_returns_target_by_source_weights() -> None:
    torch.manual_seed(42)
    attention = CrossAttention(
        model_dimension=16,
        number_of_heads=4,
        dropout=0.0,
    )
    decoder_states = torch.randn(2, 3, 16)
    encoder_states = torch.randn(2, 5, 16)

    output, weights = attention(decoder_states, encoder_states)

    assert output.shape == (2, 3, 16)
    assert weights.shape == (2, 4, 3, 5)
    assert torch.allclose(
        weights.sum(dim=-1),
        torch.ones(2, 4, 3),
        atol=1e-6,
    )


def test_cross_attention_ignores_padded_encoder_positions() -> None:
    attention = CrossAttention(
        model_dimension=8,
        number_of_heads=2,
        dropout=0.0,
    )
    decoder_states = torch.randn(1, 3, 8)
    encoder_states = torch.randn(1, 4, 8)
    source_mask = torch.tensor([[True, True, False, False]])

    _, weights = attention(
        decoder_states,
        encoder_states,
        source_mask,
    )

    assert torch.equal(
        weights[..., 2:],
        torch.zeros_like(weights[..., 2:]),
    )


def test_cross_attention_rejects_batch_mismatch() -> None:
    attention = CrossAttention(
        model_dimension=8,
        number_of_heads=2,
    )

    with pytest.raises(ValueError, match="batch"):
        attention(torch.randn(2, 3, 8), torch.randn(1, 4, 8))


def create_tiny_seq2seq_model() -> Seq2SeqTransformer:
    return Seq2SeqTransformer(
        source_vocabulary_size=20,
        target_vocabulary_size=24,
        source_padding_id=0,
        target_padding_id=0,
        model_dimension=16,
        number_of_heads=4,
        number_of_encoder_layers=1,
        number_of_decoder_layers=1,
        feed_forward_dimension=32,
        dropout=0.0,
        maximum_sequence_length=12,
    )


def test_encoder_output_interface_contains_reusable_source_state() -> None:
    model = create_tiny_seq2seq_model()
    source_ids = torch.tensor([[2, 3, 4, 0], [5, 6, 7, 8]])

    encoder_output = model.encode(source_ids)

    assert isinstance(encoder_output, EncoderOutput)
    assert encoder_output.hidden_states.shape == (2, 4, 16)
    assert torch.equal(encoder_output.attention_mask, source_ids.ne(0))
    assert len(encoder_output.self_attention) == 1
    assert torch.equal(
        encoder_output.hidden_states[0, 3],
        torch.zeros_like(encoder_output.hidden_states[0, 3]),
    )


def test_seq2seq_returns_all_three_attention_paths() -> None:
    model = create_tiny_seq2seq_model()
    source_ids = torch.tensor([[2, 3, 4, 5], [6, 7, 0, 0]])
    target_ids = torch.tensor([[1, 8, 9], [1, 10, 0]])

    logits, maps = model(
        source_ids,
        target_ids,
        return_attention=True,
    )

    assert logits.shape == (2, 3, 24)
    assert maps.encoder_self_attention[0].shape == (2, 4, 4, 4)
    assert maps.decoder_self_attention[0].shape == (2, 4, 3, 3)
    assert maps.cross_attention[0].shape == (2, 4, 3, 4)
    assert has_no_future_attention(maps.decoder_self_attention[0])
    assert not maps.cross_attention[0][1, :, :, 2:].any()


def test_changing_source_changes_decoder_output() -> None:
    torch.manual_seed(42)
    model = create_tiny_seq2seq_model()
    model.eval()
    first_source = torch.tensor([[2, 3, 4, 5]])
    changed_source = torch.tensor([[6, 7, 8, 9]])
    target_ids = torch.tensor([[1, 10, 11]])

    with torch.inference_mode():
        first_logits = model(first_source, target_ids)
        changed_logits = model(changed_source, target_ids)

    assert not torch.allclose(first_logits, changed_logits)


def test_future_target_tokens_cannot_change_earlier_logits() -> None:
    torch.manual_seed(42)
    model = create_tiny_seq2seq_model()
    model.eval()
    source_ids = torch.tensor([[2, 3, 4, 5]])
    first_target = torch.tensor([[1, 6, 7, 8, 9]])
    changed_future = torch.tensor([[1, 6, 7, 12, 13]])

    with torch.inference_mode():
        first_logits = model(source_ids, first_target)
        changed_logits = model(source_ids, changed_future)

    assert torch.allclose(
        first_logits[:, :3],
        changed_logits[:, :3],
        atol=1e-6,
    )


def test_training_step_sends_gradients_through_cross_attention() -> None:
    torch.manual_seed(42)
    model = create_tiny_seq2seq_model()
    source_ids = torch.tensor([[2, 3, 4], [5, 6, 7]])
    decoder_input_ids = torch.tensor([[1, 8, 9], [1, 10, 11]])
    expected_ids = torch.tensor([[8, 9, 2], [10, 11, 2]])

    logits = model(source_ids, decoder_input_ids)
    loss = calculate_next_token_loss(
        logits,
        expected_ids,
        expected_ids.ne(0),
    )
    loss.backward()

    cross_attention_parameters = (
        model.decoder.layers[0].cross_attention.parameters()
    )
    assert torch.isfinite(loss)
    assert any(
        parameter.grad is not None
        for parameter in cross_attention_parameters
    )


def test_greedy_decode_starts_with_bos_and_respects_length_limit() -> None:
    model = create_tiny_seq2seq_model()
    source_ids = torch.tensor([[2, 3, 4]])

    generated_ids = greedy_decode(
        model=model,
        source_token_ids=source_ids,
        beginning_of_sequence_id=1,
        end_of_sequence_id=2,
        maximum_new_tokens=4,
    )

    assert generated_ids.shape[0] == 1
    assert generated_ids[0, 0].item() == 1
    assert 2 <= generated_ids.shape[1] <= 5

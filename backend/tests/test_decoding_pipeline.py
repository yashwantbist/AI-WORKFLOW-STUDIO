"""Behavior tests for the Day 13 decoding-strategies lab."""

import math

import pytest
import torch

from backend.ml.nlp.beam_search import beam_search_decode
from backend.ml.nlp.decoding import DecodingConfig, greedy_decode
from backend.ml.nlp.generation_inference import generate
from backend.ml.nlp.sampling import (
    apply_temperature,
    filter_top_k,
    filter_top_p,
    top_k_decode,
)


class TransitionProvider:
    """Return a transition-table row for each input token."""

    def __init__(self, transition_logits: torch.Tensor) -> None:
        self.transition_logits = transition_logits

    def __call__(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.transition_logits[token_ids]


def simple_provider() -> TransitionProvider:
    """Create a deterministic path: 0 -> 1 -> 2 -> EOS(3)."""
    logits = torch.tensor(
        [
            [-4.0, 5.0, 0.0, -4.0],
            [-4.0, 0.0, 5.0, -4.0],
            [-4.0, 0.0, -4.0, 5.0],
            [-4.0, -4.0, -4.0, 5.0],
        ]
    )
    return TransitionProvider(logits)


def test_greedy_selects_highest_score_and_stops_at_eos() -> None:
    result = greedy_decode(
        simple_provider(),
        torch.tensor([[0]], dtype=torch.long),
        max_new_tokens=10,
        eos_token_id=3,
    )

    assert result.token_ids.tolist() == [[0, 1, 2, 3]]
    assert result.generated_token_count == 3


def test_lower_temperature_makes_distribution_more_confident() -> None:
    logits = torch.tensor([[2.0, 1.0, 0.0]])
    normal_probability = torch.softmax(logits, dim=-1).max()
    colder_probability = torch.softmax(
        apply_temperature(logits, temperature=0.5),
        dim=-1,
    ).max()

    assert colder_probability > normal_probability


def test_top_k_keeps_exactly_k_tokens() -> None:
    logits = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0]])
    filtered = filter_top_k(logits, top_k=3)

    assert torch.isfinite(filtered).sum().item() == 3
    assert torch.isneginf(filtered[0, 3:]).all()


def test_top_p_keeps_smallest_set_that_reaches_probability_mass() -> None:
    probabilities = torch.tensor([[0.52, 0.28, 0.12, 0.08]])
    filtered = filter_top_p(probabilities.log(), top_p=0.90)

    assert torch.isfinite(filtered).tolist() == [[True, True, True, False]]


def test_seed_makes_sampling_reproducible() -> None:
    prompt = torch.tensor([[0]], dtype=torch.long)
    first = top_k_decode(
        simple_provider(),
        prompt,
        max_new_tokens=4,
        top_k=3,
        seed=123,
    )
    second = top_k_decode(
        simple_provider(),
        prompt,
        max_new_tokens=4,
        top_k=3,
        seed=123,
    )

    assert torch.equal(first.token_ids, second.token_ids)


def test_beam_width_one_matches_greedy() -> None:
    prompt = torch.tensor([[0]], dtype=torch.long)
    greedy_result = greedy_decode(
        simple_provider(),
        prompt,
        max_new_tokens=3,
        eos_token_id=3,
    )
    beam_result = beam_search_decode(
        simple_provider(),
        prompt,
        max_new_tokens=3,
        beam_width=1,
        eos_token_id=3,
    )

    assert torch.equal(greedy_result.token_ids, beam_result.token_ids)


def test_beam_search_can_find_better_complete_sequence_than_greedy() -> None:
    # From token 0, A is locally better than B: P(A)=0.60, P(B)=0.40.
    # But B -> EOS is so strong that B/EOS wins over the whole sequence.
    logits = torch.full((5, 5), -20.0)
    logits[0, 1] = math.log(0.60)  # A
    logits[0, 2] = math.log(0.40)  # B
    logits[1, 4] = math.log(0.51)  # A -> EOS
    logits[1, 3] = math.log(0.49)  # A -> X
    logits[2, 4] = math.log(0.99)  # B -> EOS
    logits[2, 3] = math.log(0.01)  # B -> X
    logits[3, 4] = 0.0
    logits[4, 4] = 0.0
    provider = TransitionProvider(logits)
    prompt = torch.tensor([[0]], dtype=torch.long)

    greedy_result = greedy_decode(
        provider,
        prompt,
        max_new_tokens=2,
        eos_token_id=4,
    )
    beam_result = beam_search_decode(
        provider,
        prompt,
        max_new_tokens=2,
        beam_width=2,
        eos_token_id=4,
    )

    assert greedy_result.token_ids.tolist() == [[0, 1, 4]]
    assert beam_result.token_ids.tolist() == [[0, 2, 4]]
    assert (
        beam_result.cumulative_log_probability
        > greedy_result.cumulative_log_probability
    )


@pytest.mark.parametrize(
    "strategy",
    ["greedy", "beam", "temperature", "top_k", "top_p"],
)
def test_shared_interface_dispatches_every_strategy(strategy: str) -> None:
    config = DecodingConfig(
        strategy=strategy,  # type: ignore[arg-type]
        max_new_tokens=2,
        eos_token_id=3,
        beam_width=2,
        top_k=2,
        top_p=0.8,
        seed=7,
    )
    result = generate(
        simple_provider(),
        torch.tensor([[0]], dtype=torch.long),
        config,
    )

    assert result.strategy == strategy
    assert result.generated_token_count <= 2
    assert result.elapsed_seconds >= 0


def test_invalid_decoding_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="temperature"):
        DecodingConfig(strategy="top_p", temperature=0)

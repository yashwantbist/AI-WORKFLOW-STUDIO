"""Beam search for keeping several promising token sequences at once."""

from dataclasses import dataclass

import torch
from torch import Tensor

try:
    from .decoding import (
        GenerationResult,
        LogitsProvider,
        get_next_token_logits,
        validate_single_prompt,
    )
except ImportError:
    from decoding import (
        GenerationResult,
        LogitsProvider,
        get_next_token_logits,
        validate_single_prompt,
    )


@dataclass(frozen=True)
class Beam:
    """One candidate sequence being considered by beam search."""

    token_ids: Tensor
    cumulative_log_probability: float
    finished: bool = False

    def ranking_score(
        self,
        prompt_length: int,
        length_penalty: float,
    ) -> float:
        """Normalize a beam score so longer sequences are not over-penalized."""
        generated_length = max(self.token_ids.size(1) - prompt_length, 1)
        if length_penalty == 0:
            return self.cumulative_log_probability
        return self.cumulative_log_probability / (
            generated_length**length_penalty
        )


def beam_search_decode(
    logits_provider: LogitsProvider,
    prompt_token_ids: Tensor,
    max_new_tokens: int,
    beam_width: int = 3,
    eos_token_id: int | None = None,
    length_penalty: float = 0.0,
) -> GenerationResult:
    """Explore the best ``beam_width`` sequence candidates at every step."""
    validate_single_prompt(prompt_token_ids)
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if beam_width < 1:
        raise ValueError("beam_width must be positive")
    if length_penalty < 0:
        raise ValueError("length_penalty cannot be negative")

    prompt_length = prompt_token_ids.size(1)
    beams = [
        Beam(
            token_ids=prompt_token_ids.clone(),
            cumulative_log_probability=0.0,
        )
    ]

    for _ in range(max_new_tokens):
        candidates: list[Beam] = []

        for beam in beams:
            if beam.finished:
                candidates.append(beam)
                continue

            next_token_logits = get_next_token_logits(
                logits_provider,
                beam.token_ids,
            )
            log_probabilities = torch.log_softmax(
                next_token_logits,
                dim=-1,
            )
            candidate_count = min(beam_width, log_probabilities.size(-1))
            top_log_probabilities, top_token_ids = torch.topk(
                log_probabilities,
                k=candidate_count,
                dim=-1,
            )

            for candidate_index in range(candidate_count):
                next_token_id = top_token_ids[:, candidate_index : candidate_index + 1]
                token_id_value = int(next_token_id.item())
                candidates.append(
                    Beam(
                        token_ids=torch.cat(
                            [beam.token_ids, next_token_id],
                            dim=1,
                        ),
                        cumulative_log_probability=(
                            beam.cumulative_log_probability
                            + float(
                                top_log_probabilities[
                                    0,
                                    candidate_index,
                                ].item()
                            )
                        ),
                        finished=(
                            eos_token_id is not None
                            and token_id_value == eos_token_id
                        ),
                    )
                )

        beams = sorted(
            candidates,
            key=lambda candidate: candidate.ranking_score(
                prompt_length,
                length_penalty,
            ),
            reverse=True,
        )[:beam_width]

        if all(beam.finished for beam in beams):
            break

    best_beam = max(
        beams,
        key=lambda candidate: candidate.ranking_score(
            prompt_length,
            length_penalty,
        ),
    )
    return GenerationResult(
        strategy="beam",
        token_ids=best_beam.token_ids,
        prompt_length=prompt_length,
        cumulative_log_probability=(
            best_beam.cumulative_log_probability
        ),
    )

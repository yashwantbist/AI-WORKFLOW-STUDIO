"""Shared data structures and greedy decoding for text generation.

The functions in this module expect a ``logits_provider``: any callable that
accepts token IDs with shape ``[batch, sequence]`` and returns either logits
or a tuple whose first item is logits. Day 11's ``CausalLanguageModel`` already
has exactly that interface.
"""

from dataclasses import dataclass
from typing import Callable, Literal, TypeAlias

import torch
from torch import Tensor


DecodingStrategy = Literal[
    "greedy",
    "beam",
    "temperature",
    "top_k",
    "top_p",
]
ModelOutput: TypeAlias = Tensor | tuple[object, ...]
LogitsProvider: TypeAlias = Callable[[Tensor], ModelOutput]


@dataclass(frozen=True)
class DecodingConfig:
    """All decoding settings in one validated object.

    Only the fields needed by the selected strategy are used. Keeping them in
    one object makes it easy to switch strategies without changing the model.
    """

    strategy: DecodingStrategy = "greedy"
    max_new_tokens: int = 20
    eos_token_id: int | None = None
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    beam_width: int = 3
    length_penalty: float = 0.0
    seed: int | None = None

    def __post_init__(self) -> None:
        valid_strategies = {
            "greedy",
            "beam",
            "temperature",
            "top_k",
            "top_p",
        }
        if self.strategy not in valid_strategies:
            raise ValueError(
                f"strategy must be one of {sorted(valid_strategies)}"
            )
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be greater than zero")
        if self.top_k < 1:
            raise ValueError("top_k must be positive")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in the interval (0, 1]")
        if self.beam_width < 1:
            raise ValueError("beam_width must be positive")
        if self.length_penalty < 0:
            raise ValueError("length_penalty cannot be negative")
        if self.eos_token_id is not None and self.eos_token_id < 0:
            raise ValueError("eos_token_id cannot be negative")


@dataclass(frozen=True)
class GenerationResult:
    """Tokens plus useful metadata returned by every strategy."""

    strategy: DecodingStrategy
    token_ids: Tensor
    prompt_length: int
    cumulative_log_probability: float
    elapsed_seconds: float = 0.0

    @property
    def generated_token_ids(self) -> Tensor:
        """Return only the continuation, excluding the original prompt."""
        return self.token_ids[:, self.prompt_length :]

    @property
    def generated_token_count(self) -> int:
        """Return the number of generated positions."""
        return self.token_ids.size(1) - self.prompt_length


def validate_single_prompt(prompt_token_ids: Tensor) -> None:
    """Validate the simple one-prompt interface used by this learning lab."""
    if prompt_token_ids.ndim != 2:
        raise ValueError(
            "prompt_token_ids must have shape [batch, sequence]"
        )
    if prompt_token_ids.size(0) != 1:
        raise ValueError("this educational API currently supports batch size 1")
    if prompt_token_ids.size(1) < 1:
        raise ValueError("the prompt must contain at least one token")
    if prompt_token_ids.dtype != torch.long:
        raise ValueError("prompt_token_ids must use torch.long token IDs")


def extract_next_token_logits(model_output: ModelOutput) -> Tensor:
    """Extract the final-position scores from a common model output format."""
    if isinstance(model_output, tuple):
        if not model_output:
            raise ValueError("the model returned an empty tuple")
        model_output = model_output[0]

    if not isinstance(model_output, Tensor):
        raise TypeError("the model must return a Tensor or a tuple starting with one")

    if model_output.ndim == 3:
        # Language models usually return [batch, sequence, vocabulary].
        next_token_logits = model_output[:, -1, :]
    elif model_output.ndim == 2:
        # A lightweight provider may directly return [batch, vocabulary].
        next_token_logits = model_output
    else:
        raise ValueError(
            "model logits must have shape [batch, vocabulary] or "
            "[batch, sequence, vocabulary]"
        )

    if next_token_logits.size(-1) < 2:
        raise ValueError("the vocabulary must contain at least two tokens")
    return next_token_logits


def get_next_token_logits(
    logits_provider: LogitsProvider,
    token_ids: Tensor,
) -> Tensor:
    """Ask the model for scores and select the final sequence position."""
    return extract_next_token_logits(logits_provider(token_ids))


def greedy_decode(
    logits_provider: LogitsProvider,
    prompt_token_ids: Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
) -> GenerationResult:
    """Repeatedly choose the single highest-scoring next token."""
    validate_single_prompt(prompt_token_ids)
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")

    generated_ids = prompt_token_ids.clone()
    prompt_length = generated_ids.size(1)
    cumulative_log_probability = 0.0

    for _ in range(max_new_tokens):
        next_token_logits = get_next_token_logits(
            logits_provider,
            generated_ids,
        )
        log_probabilities = torch.log_softmax(next_token_logits, dim=-1)
        next_token_id = log_probabilities.argmax(dim=-1, keepdim=True)
        selected_log_probability = log_probabilities.gather(
            dim=-1,
            index=next_token_id,
        )
        cumulative_log_probability += float(
            selected_log_probability.item()
        )
        generated_ids = torch.cat([generated_ids, next_token_id], dim=1)

        if (
            eos_token_id is not None
            and int(next_token_id.item()) == eos_token_id
        ):
            break

    return GenerationResult(
        strategy="greedy",
        token_ids=generated_ids,
        prompt_length=prompt_length,
        cumulative_log_probability=cumulative_log_probability,
    )

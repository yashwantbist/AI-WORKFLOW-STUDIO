"""Temperature, top-k, and top-p sampling utilities."""

import torch
from torch import Tensor

try:
    from .decoding import (
        DecodingStrategy,
        GenerationResult,
        LogitsProvider,
        get_next_token_logits,
        validate_single_prompt,
    )
except ImportError:
    from decoding import (
        DecodingStrategy,
        GenerationResult,
        LogitsProvider,
        get_next_token_logits,
        validate_single_prompt,
    )


def apply_temperature(logits: Tensor, temperature: float) -> Tensor:
    """Scale logits before softmax; smaller values sharpen probabilities."""
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero")
    return logits / temperature


def filter_top_k(logits: Tensor, top_k: int) -> Tensor:
    """Keep exactly the highest ``k`` logits and block the rest."""
    if top_k < 1:
        raise ValueError("top_k must be positive")

    vocabulary_size = logits.size(-1)
    kept_count = min(top_k, vocabulary_size)
    top_values, top_indices = torch.topk(logits, kept_count, dim=-1)
    filtered_logits = torch.full_like(logits, -torch.inf)
    return filtered_logits.scatter(-1, top_indices, top_values)


def filter_top_p(logits: Tensor, top_p: float) -> Tensor:
    """Keep the smallest high-probability set whose total reaches ``p``."""
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in the interval (0, 1]")

    sorted_logits, sorted_indices = torch.sort(
        logits,
        descending=True,
        dim=-1,
    )
    sorted_probabilities = torch.softmax(sorted_logits, dim=-1)
    cumulative_probabilities = sorted_probabilities.cumsum(dim=-1)

    # Shift the removal decision right. This keeps the first token that makes
    # the cumulative total reach or cross p, so at least one token survives.
    remove_sorted = cumulative_probabilities >= top_p
    remove_sorted[..., 1:] = remove_sorted[..., :-1].clone()
    remove_sorted[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(remove_sorted, -torch.inf)

    filtered_logits = torch.full_like(logits, -torch.inf)
    return filtered_logits.scatter(-1, sorted_indices, sorted_logits)


def sample_next_token(
    logits: Tensor,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Randomly select one token and return its ID and log probability."""
    probabilities = torch.softmax(logits, dim=-1)
    if not torch.isfinite(probabilities).all():
        raise ValueError("sampling probabilities contain NaN or infinity")
    if (probabilities.sum(dim=-1) <= 0).any():
        raise ValueError("at least one token must remain available")

    next_token_id = torch.multinomial(
        probabilities,
        num_samples=1,
        generator=generator,
    )
    selected_probability = probabilities.gather(-1, next_token_id)
    return next_token_id, selected_probability.clamp_min(1e-12).log()


def _sampling_decode(
    logits_provider: LogitsProvider,
    prompt_token_ids: Tensor,
    strategy: DecodingStrategy,
    max_new_tokens: int,
    eos_token_id: int | None,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    seed: int | None,
) -> GenerationResult:
    """Shared token loop used by all three random sampling strategies."""
    validate_single_prompt(prompt_token_ids)
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero")

    generator = None
    if seed is not None:
        generator = torch.Generator(device=prompt_token_ids.device)
        generator.manual_seed(seed)

    generated_ids = prompt_token_ids.clone()
    prompt_length = generated_ids.size(1)
    cumulative_log_probability = 0.0

    for _ in range(max_new_tokens):
        logits = get_next_token_logits(logits_provider, generated_ids)
        filtered_logits = apply_temperature(logits, temperature)
        if top_k is not None:
            filtered_logits = filter_top_k(filtered_logits, top_k)
        if top_p is not None:
            filtered_logits = filter_top_p(filtered_logits, top_p)

        next_token_id, selected_log_probability = sample_next_token(
            filtered_logits,
            generator,
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
        strategy=strategy,
        token_ids=generated_ids,
        prompt_length=prompt_length,
        cumulative_log_probability=cumulative_log_probability,
    )


def temperature_decode(
    logits_provider: LogitsProvider,
    prompt_token_ids: Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    eos_token_id: int | None = None,
    seed: int | None = None,
) -> GenerationResult:
    """Sample from the complete vocabulary after temperature scaling."""
    return _sampling_decode(
        logits_provider=logits_provider,
        prompt_token_ids=prompt_token_ids,
        strategy="temperature",
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
        temperature=temperature,
        top_k=None,
        top_p=None,
        seed=seed,
    )


def top_k_decode(
    logits_provider: LogitsProvider,
    prompt_token_ids: Tensor,
    max_new_tokens: int,
    top_k: int,
    temperature: float = 1.0,
    eos_token_id: int | None = None,
    seed: int | None = None,
) -> GenerationResult:
    """Sample after keeping only the fixed number of best tokens."""
    return _sampling_decode(
        logits_provider=logits_provider,
        prompt_token_ids=prompt_token_ids,
        strategy="top_k",
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
        temperature=temperature,
        top_k=top_k,
        top_p=None,
        seed=seed,
    )


def top_p_decode(
    logits_provider: LogitsProvider,
    prompt_token_ids: Tensor,
    max_new_tokens: int,
    top_p: float,
    temperature: float = 1.0,
    eos_token_id: int | None = None,
    seed: int | None = None,
) -> GenerationResult:
    """Sample from a probability-mass-based, dynamically sized token set."""
    return _sampling_decode(
        logits_provider=logits_provider,
        prompt_token_ids=prompt_token_ids,
        strategy="top_p",
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
        temperature=temperature,
        top_k=None,
        top_p=top_p,
        seed=seed,
    )

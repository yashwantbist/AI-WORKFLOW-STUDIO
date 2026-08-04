"""One configurable inference function for every decoding strategy.

This filename intentionally does not replace ``inference.py`` because the
earlier NLP classifier lab already uses that module.
"""

from dataclasses import replace
from time import perf_counter

import torch
from torch import nn

try:
    from .beam_search import beam_search_decode
    from .decoding import (
        DecodingConfig,
        GenerationResult,
        LogitsProvider,
        greedy_decode,
    )
    from .sampling import (
        temperature_decode,
        top_k_decode,
        top_p_decode,
    )
except ImportError:
    from beam_search import beam_search_decode
    from decoding import (
        DecodingConfig,
        GenerationResult,
        LogitsProvider,
        greedy_decode,
    )
    from sampling import temperature_decode, top_k_decode, top_p_decode


def _dispatch_generation(
    logits_provider: LogitsProvider,
    prompt_token_ids: torch.Tensor,
    config: DecodingConfig,
) -> GenerationResult:
    """Send a validated configuration to its decoding implementation."""
    shared_arguments = {
        "logits_provider": logits_provider,
        "prompt_token_ids": prompt_token_ids,
        "max_new_tokens": config.max_new_tokens,
        "eos_token_id": config.eos_token_id,
    }

    if config.strategy == "greedy":
        return greedy_decode(**shared_arguments)
    if config.strategy == "beam":
        return beam_search_decode(
            **shared_arguments,
            beam_width=config.beam_width,
            length_penalty=config.length_penalty,
        )
    if config.strategy == "temperature":
        return temperature_decode(
            **shared_arguments,
            temperature=config.temperature,
            seed=config.seed,
        )
    if config.strategy == "top_k":
        return top_k_decode(
            **shared_arguments,
            top_k=config.top_k,
            temperature=config.temperature,
            seed=config.seed,
        )
    return top_p_decode(
        **shared_arguments,
        top_p=config.top_p,
        temperature=config.temperature,
        seed=config.seed,
    )


def generate(
    logits_provider: LogitsProvider,
    prompt_token_ids: torch.Tensor,
    config: DecodingConfig | None = None,
) -> GenerationResult:
    """Generate a continuation using the strategy selected in ``config``.

    If the provider is a PyTorch module, inference mode is enabled temporarily
    and the module's original train/eval state is restored afterward.
    """
    if config is None:
        config = DecodingConfig()

    model = logits_provider if isinstance(logits_provider, nn.Module) else None
    was_training = model.training if model is not None else None
    if model is not None:
        model.eval()

    started_at = perf_counter()
    try:
        with torch.inference_mode():
            result = _dispatch_generation(
                logits_provider,
                prompt_token_ids,
                config,
            )
    finally:
        if model is not None and was_training is not None:
            model.train(was_training)

    return replace(
        result,
        elapsed_seconds=perf_counter() - started_at,
    )

"""Safe structured log payloads for LLM inference."""

from __future__ import annotations

from .models import GenerationResult


def generation_completed_event(
    result: GenerationResult,
) -> dict[str, object]:
    """Create a structured event without including prompt or answer text."""

    return {
        "event": "llm_generation_completed",
        "model": result.model,
        "latency_ms": result.metrics.latency_ms,
        "input_tokens": result.metrics.input_tokens,
        "output_tokens": result.metrics.output_tokens,
        "total_tokens": result.metrics.total_tokens,
        "estimated_cost": result.metrics.estimated_cost,
        "temperature": result.config.temperature,
        "max_tokens": result.config.max_tokens,
        "top_p": result.config.top_p,
        "top_k": result.config.top_k,
    }


def generation_failed_event(
    *,
    error: Exception,
    config,
) -> dict[str, object]:
    """Create a failure event without leaking prompt contents."""

    return {
        "event": "llm_generation_failed",
        "error_type": type(error).__name__,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "top_p": config.top_p,
        "top_k": config.top_k,
    }

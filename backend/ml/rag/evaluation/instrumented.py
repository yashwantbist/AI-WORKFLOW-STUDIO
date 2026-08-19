"""Latency instrumentation and normalized observable inference."""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter

from .models import (
    GenerationConfig,
    GenerationResult,
    InferenceMetrics,
)
from .pricing import ModelPricing, estimate_cost
from .provider import LLMProvider


Clock = Callable[[], float]


class ObservableLLM:
    """Wrap an LLM provider with normalized metrics.

    The timer covers only the provider generation call. Retrieval, database
    work, prompt construction, and unrelated pipeline work should be measured
    separately.
    """

    def __init__(
        self,
        provider: LLMProvider,
        *,
        pricing: ModelPricing | None = None,
        clock: Clock = perf_counter,
    ) -> None:
        self._provider = provider
        self._pricing = pricing
        self._clock = clock

    def generate(
        self,
        prompt: str,
        config: GenerationConfig,
    ) -> GenerationResult:
        if not prompt.strip():
            raise ValueError("prompt cannot be empty")

        started = self._clock()

        # Let provider exceptions propagate. Callers can decide retry/fallback
        # policy, and a higher-level operation logger can record the failure.
        raw = self._provider.generate(
            prompt,
            config,
        )

        finished = self._clock()
        latency_ms = (finished - started) * 1000.0

        estimated_cost = estimate_cost(
            raw.usage,
            self._pricing,
        )

        return GenerationResult(
            text=raw.text,
            model=raw.model,
            config=config,
            metrics=InferenceMetrics(
                latency_ms=latency_ms,
                usage=raw.usage,
                estimated_cost=estimated_cost,
            ),
            provider_metadata=raw.provider_metadata,
        )

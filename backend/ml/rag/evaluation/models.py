"""Provider-independent data models for observable LLM inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class GenerationConfig:
    temperature: float = 0.2
    max_tokens: int = 512
    top_p: float = 1.0
    top_k: int | None = None

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in the interval (0, 1]")
        if self.top_k is not None and self.top_k < 1:
            raise ValueError("top_k must be at least 1 when provided")

    def to_dict(self) -> dict[str, object]:
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }


@dataclass(frozen=True)
class InferenceUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("input_tokens", self.input_tokens),
            ("output_tokens", self.output_tokens),
        ):
            if value is not None and value < 0:
                raise ValueError(f"{name} must be >= 0 when provided")

    @property
    def total_tokens(self) -> int | None:
        if self.input_tokens is None or self.output_tokens is None:
            return None
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, int | None]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True)
class InferenceMetrics:
    latency_ms: float
    usage: InferenceUsage
    estimated_cost: float | None = None

    def __post_init__(self) -> None:
        if self.latency_ms < 0:
            raise ValueError("latency_ms must be >= 0")
        if self.estimated_cost is not None and self.estimated_cost < 0:
            raise ValueError("estimated_cost must be >= 0 when provided")

    @property
    def input_tokens(self) -> int | None:
        return self.usage.input_tokens

    @property
    def output_tokens(self) -> int | None:
        return self.usage.output_tokens

    @property
    def total_tokens(self) -> int | None:
        return self.usage.total_tokens

    @property
    def output_tokens_per_second(self) -> float | None:
        if self.output_tokens is None:
            return None
        if self.latency_ms <= 0:
            return None
        return self.output_tokens / (self.latency_ms / 1000.0)

    def to_dict(self) -> dict[str, object]:
        return {
            "latency_ms": self.latency_ms,
            **self.usage.to_dict(),
            "estimated_cost": self.estimated_cost,
            "output_tokens_per_second": self.output_tokens_per_second,
        }


@dataclass(frozen=True)
class ProviderGeneration:
    """Raw normalized provider payload before instrumentation is attached."""

    text: str
    model: str
    usage: InferenceUsage = InferenceUsage()
    provider_metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class GenerationResult:
    text: str
    model: str
    config: GenerationConfig
    metrics: InferenceMetrics
    provider_metadata: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "model": self.model,
            "config": self.config.to_dict(),
            "metrics": self.metrics.to_dict(),
            "provider_metadata": (
                dict(self.provider_metadata)
                if self.provider_metadata is not None
                else None
            ),
        }

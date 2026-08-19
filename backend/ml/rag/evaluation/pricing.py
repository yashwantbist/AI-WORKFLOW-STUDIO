"""Optional pricing models for request-cost estimation."""

from __future__ import annotations

from dataclasses import dataclass

from .models import InferenceUsage


@dataclass(frozen=True)
class ModelPricing:
    input_per_million: float
    output_per_million: float
    currency: str = "USD"

    def __post_init__(self) -> None:
        if self.input_per_million < 0:
            raise ValueError("input_per_million must be >= 0")
        if self.output_per_million < 0:
            raise ValueError("output_per_million must be >= 0")
        if not self.currency.strip():
            raise ValueError("currency cannot be empty")


def estimate_cost(
    usage: InferenceUsage,
    pricing: ModelPricing | None,
) -> float | None:
    """Estimate request cost only when pricing and complete usage are known."""

    if pricing is None:
        return None

    if usage.input_tokens is None or usage.output_tokens is None:
        return None

    input_cost = (
        usage.input_tokens
        / 1_000_000
        * pricing.input_per_million
    )
    output_cost = (
        usage.output_tokens
        / 1_000_000
        * pricing.output_per_million
    )

    return input_cost + output_cost

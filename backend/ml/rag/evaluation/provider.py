"""Provider boundary for LLM generation."""

from __future__ import annotations

from typing import Protocol

from .models import GenerationConfig, ProviderGeneration


class LLMProvider(Protocol):
    def generate(
        self,
        prompt: str,
        config: GenerationConfig,
    ) -> ProviderGeneration:
        """Generate text without exposing provider-specific response objects."""
        ...

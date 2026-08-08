"""LLM provider interfaces and optional OpenAI Responses API integration."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Protocol
from dotenv import load_dotenv


load_dotenv()

class LLMProvider(Protocol):
    """Generation interface used by RAGPipeline."""

    def generate(self, *, instructions: str, prompt: str) -> str:
        """Generate one answer from grounded instructions and prompt."""


@dataclass
class RecordingDemoProvider:
    """Offline provider for learning, demos, and tests.

    It proves that retrieved context reaches the generation layer but is not
    intended to imitate a real language model.
    """

    answer_text: str = (
        "Demo generation completed from the supplied retrieved context."
    )
    last_instructions: str | None = None
    last_prompt: str | None = None

    def generate(self, *, instructions: str, prompt: str) -> str:
        self.last_instructions = instructions
        self.last_prompt = prompt
        return self.answer_text


class OpenAIResponsesProvider:
    """Optional OpenAI provider using the Responses API."""

    def __init__(
        self,
        model: str,
        *,
        client: Any | None = None,
    ) -> None:
        clean_model = model.strip()
        if not clean_model:
            raise ValueError("model cannot be empty")

        if client is None:
            try:
                from openai import OpenAI
            except ImportError as error:
                raise RuntimeError(
                    "OpenAI provider requires the 'openai' package. "
                    "Install it with: python -m pip install -U openai"
                ) from error
            client = OpenAI()

        self._model = clean_model
        self._client = client

    @classmethod
    def from_environment(cls) -> "OpenAIResponsesProvider":
        model = os.getenv("OPENAI_MODEL", "").strip()
        if not model:
            raise RuntimeError(
                "Set OPENAI_MODEL to the model you want to use. "
                "Keep OPENAI_API_KEY in the environment, never in source code."
            )
        return cls(model=model)

    def generate(self, *, instructions: str, prompt: str) -> str:
        response = self._client.responses.create(
            model=self._model,
            instructions=instructions,
            input=prompt,
        )
        output_text = str(getattr(response, "output_text", "")).strip()
        if not output_text:
            raise RuntimeError("LLM provider returned an empty response")
        return output_text

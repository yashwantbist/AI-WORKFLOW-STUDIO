from dataclasses import dataclass

import pytest

from backend.ml.rag.evaluation import (
    GenerationConfig,
    InferenceUsage,
    ModelPricing,
    ObservableLLM,
    ProviderGeneration,
    estimate_cost,
    generation_completed_event,
    generation_failed_event,
)


class FakeClock:
    def __init__(self, *values: float) -> None:
        self._values = iter(values)

    def __call__(self) -> float:
        return next(self._values)


class FakeLLMProvider:
    def __init__(self, response: ProviderGeneration) -> None:
        self.response = response
        self.calls = []

    def generate(self, prompt, config):
        self.calls.append((prompt, config))
        return self.response


class FailingProvider:
    def generate(self, prompt, config):
        raise RuntimeError("provider unavailable")


def test_successful_generation_preserves_config_and_usage():
    config = GenerationConfig(
        temperature=0.2,
        max_tokens=200,
        top_p=0.9,
        top_k=50,
    )
    provider = FakeLLMProvider(
        ProviderGeneration(
            text="Grounded answer.",
            model="fake-model",
            usage=InferenceUsage(
                input_tokens=125,
                output_tokens=25,
            ),
            provider_metadata={"request_id": "abc"},
        )
    )

    llm = ObservableLLM(
        provider,
        clock=FakeClock(10.0, 10.25),
    )

    result = llm.generate(
        "Explain retrieval.",
        config,
    )

    assert result.text == "Grounded answer."
    assert result.model == "fake-model"
    assert result.config == config
    assert result.metrics.latency_ms == pytest.approx(250.0)
    assert result.metrics.input_tokens == 125
    assert result.metrics.output_tokens == 25
    assert result.metrics.total_tokens == 150
    assert provider.calls == [
        ("Explain retrieval.", config)
    ]


def test_missing_usage_stays_none():
    provider = FakeLLMProvider(
        ProviderGeneration(
            text="Answer.",
            model="fake-model",
            usage=InferenceUsage(),
        )
    )

    result = ObservableLLM(
        provider,
        clock=FakeClock(0.0, 0.1),
    ).generate(
        "Prompt",
        GenerationConfig(),
    )

    assert result.metrics.input_tokens is None
    assert result.metrics.output_tokens is None
    assert result.metrics.total_tokens is None
    assert result.metrics.estimated_cost is None
    assert result.metrics.output_tokens_per_second is None


def test_zero_output_tokens_are_valid_and_throughput_is_zero():
    provider = FakeLLMProvider(
        ProviderGeneration(
            text="",
            model="fake-model",
            usage=InferenceUsage(
                input_tokens=100,
                output_tokens=0,
            ),
        )
    )

    result = ObservableLLM(
        provider,
        clock=FakeClock(5.0, 5.5),
    ).generate(
        "Prompt",
        GenerationConfig(),
    )

    assert result.metrics.output_tokens == 0
    assert result.metrics.output_tokens_per_second == 0.0


def test_cost_estimation_uses_external_pricing():
    usage = InferenceUsage(
        input_tokens=1_000_000,
        output_tokens=500_000,
    )
    pricing = ModelPricing(
        input_per_million=2.0,
        output_per_million=8.0,
    )

    assert estimate_cost(
        usage,
        pricing,
    ) == pytest.approx(6.0)


def test_cost_is_none_without_verified_pricing():
    usage = InferenceUsage(
        input_tokens=100,
        output_tokens=50,
    )

    assert estimate_cost(
        usage,
        None,
    ) is None


def test_cost_is_none_when_usage_is_incomplete():
    pricing = ModelPricing(
        input_per_million=2.0,
        output_per_million=8.0,
    )

    assert estimate_cost(
        InferenceUsage(
            input_tokens=100,
            output_tokens=None,
        ),
        pricing,
    ) is None


def test_observable_llm_applies_pricing_when_usage_exists():
    provider = FakeLLMProvider(
        ProviderGeneration(
            text="Answer.",
            model="fake-model",
            usage=InferenceUsage(
                input_tokens=1_000,
                output_tokens=200,
            ),
        )
    )
    pricing = ModelPricing(
        input_per_million=1.0,
        output_per_million=5.0,
    )

    result = ObservableLLM(
        provider,
        pricing=pricing,
        clock=FakeClock(1.0, 1.2),
    ).generate(
        "Prompt",
        GenerationConfig(),
    )

    expected = (
        1_000 / 1_000_000 * 1.0
        + 200 / 1_000_000 * 5.0
    )
    assert result.metrics.estimated_cost == pytest.approx(expected)


def test_provider_exception_is_not_swallowed():
    llm = ObservableLLM(
        FailingProvider(),
        clock=FakeClock(1.0),
    )

    with pytest.raises(
        RuntimeError,
        match="provider unavailable",
    ):
        llm.generate(
            "Prompt",
            GenerationConfig(),
        )


def test_completed_log_does_not_include_prompt_or_generated_text():
    provider = FakeLLMProvider(
        ProviderGeneration(
            text="Sensitive answer.",
            model="fake-model",
            usage=InferenceUsage(
                input_tokens=10,
                output_tokens=5,
            ),
        )
    )

    result = ObservableLLM(
        provider,
        clock=FakeClock(2.0, 2.1),
    ).generate(
        "Sensitive private prompt",
        GenerationConfig(
            temperature=0.1,
            max_tokens=80,
            top_p=0.95,
        ),
    )

    event = generation_completed_event(result)

    assert event["event"] == "llm_generation_completed"
    assert event["model"] == "fake-model"
    assert event["input_tokens"] == 10
    assert event["output_tokens"] == 5
    assert event["temperature"] == 0.1
    assert "prompt" not in event
    assert "text" not in event
    assert "Sensitive private prompt" not in repr(event)
    assert "Sensitive answer." not in repr(event)


def test_failed_log_contains_error_type_but_not_prompt():
    config = GenerationConfig()

    event = generation_failed_event(
        error=TimeoutError("request timed out"),
        config=config,
    )

    assert event["event"] == "llm_generation_failed"
    assert event["error_type"] == "TimeoutError"
    assert "prompt" not in event


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"temperature": -0.1}, "temperature"),
        ({"max_tokens": 0}, "max_tokens"),
        ({"top_p": 0.0}, "top_p"),
        ({"top_p": 1.1}, "top_p"),
        ({"top_k": 0}, "top_k"),
    ],
)
def test_invalid_generation_config_is_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        GenerationConfig(**kwargs)


def test_output_tokens_per_second_uses_measured_latency():
    provider = FakeLLMProvider(
        ProviderGeneration(
            text="Answer.",
            model="fake-model",
            usage=InferenceUsage(
                input_tokens=20,
                output_tokens=100,
            ),
        )
    )

    result = ObservableLLM(
        provider,
        clock=FakeClock(3.0, 5.0),
    ).generate(
        "Prompt",
        GenerationConfig(),
    )

    assert result.metrics.latency_ms == pytest.approx(2000.0)
    assert result.metrics.output_tokens_per_second == pytest.approx(50.0)


def test_empty_prompt_is_rejected_before_provider_call():
    provider = FakeLLMProvider(
        ProviderGeneration(
            text="should not run",
            model="fake-model",
        )
    )

    with pytest.raises(ValueError, match="prompt"):
        ObservableLLM(provider).generate(
            "   ",
            GenerationConfig(),
        )

    assert provider.calls == []

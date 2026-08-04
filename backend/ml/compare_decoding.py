"""Run every decoding strategy against the same tiny language model.

This demonstration needs no download and no training. The toy model stores a
table of next-token scores, which lets the lab focus only on decoding.
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

try:
    from .decoding import DecodingConfig, GenerationResult
    from .generation_inference import generate
except ImportError:
    from decoding import DecodingConfig, GenerationResult
    from generation_inference import generate


TOKENS = [
    "<pad>",
    "<bos>",
    "<eos>",
    "the",
    "cat",
    "dog",
    "bird",
    "sat",
    "ran",
    "sang",
    "on",
    "mat",
    "quickly",
    ".",
]
TOKEN_TO_ID = {token: token_id for token_id, token in enumerate(TOKENS)}


class ToyLanguageModel(nn.Module):
    """Return learned-looking logits from a hand-authored transition table."""

    def __init__(self) -> None:
        super().__init__()
        vocabulary_size = len(TOKENS)
        transitions = torch.full(
            (vocabulary_size, vocabulary_size),
            -6.0,
        )

        def set_choices(source: str, choices: dict[str, float]) -> None:
            for target, score in choices.items():
                transitions[TOKEN_TO_ID[source], TOKEN_TO_ID[target]] = score

        set_choices("<pad>", {"<eos>": 5.0})
        set_choices("<bos>", {"the": 5.0})
        set_choices("<eos>", {"<eos>": 5.0})
        set_choices("the", {"cat": 4.0, "dog": 3.4, "bird": 3.0})
        set_choices("cat", {"sat": 4.0, "ran": 2.2, "sang": 1.8})
        set_choices("dog", {"ran": 4.0, "sat": 2.4, "sang": 2.0})
        set_choices("bird", {"sang": 4.0, "sat": 2.2, "ran": 1.5})
        set_choices("sat", {"on": 4.0, "quickly": 2.7, ".": 2.0})
        set_choices("ran", {"quickly": 4.0, ".": 3.1, "on": 1.0})
        set_choices("sang", {".": 4.0, "quickly": 2.5})
        set_choices("on", {"mat": 4.0, ".": 2.5})
        set_choices("mat", {".": 4.0})
        set_choices("quickly", {".": 4.0})
        set_choices(".", {"<eos>": 5.0})

        self.register_buffer("transition_logits", transitions)

    def forward(self, token_ids: Tensor) -> Tensor:
        """Return logits with shape [batch, sequence, vocabulary]."""
        return self.transition_logits[token_ids]


@dataclass(frozen=True)
class ComparisonMetrics:
    """Simple measurements that do not pretend to judge writing subjectively."""

    unique_token_ratio: float
    repeated_bigram_ratio: float
    average_log_probability: float
    ended_with_eos: bool


def decode_text(token_ids: Tensor) -> str:
    """Convert IDs to readable words while hiding control tokens."""
    hidden_tokens = {"<pad>", "<bos>", "<eos>"}
    words = [
        TOKENS[token_id]
        for token_id in token_ids.squeeze(0).tolist()
        if TOKENS[token_id] not in hidden_tokens
    ]
    return " ".join(words).replace(" .", ".")


def calculate_metrics(result: GenerationResult) -> ComparisonMetrics:
    """Measure diversity, repetition, confidence, and completion."""
    generated = result.generated_token_ids.squeeze(0).tolist()
    content_ids = [
        token_id
        for token_id in generated
        if token_id not in {TOKEN_TO_ID["<pad>"], TOKEN_TO_ID["<eos>"]}
    ]
    unique_token_ratio = (
        len(set(content_ids)) / len(content_ids) if content_ids else 0.0
    )

    bigrams = list(zip(content_ids, content_ids[1:]))
    repeated_bigram_ratio = (
        1.0 - len(set(bigrams)) / len(bigrams) if bigrams else 0.0
    )
    average_log_probability = (
        result.cumulative_log_probability / result.generated_token_count
        if result.generated_token_count
        else 0.0
    )
    ended_with_eos = bool(
        generated and generated[-1] == TOKEN_TO_ID["<eos>"]
    )
    return ComparisonMetrics(
        unique_token_ratio=unique_token_ratio,
        repeated_bigram_ratio=repeated_bigram_ratio,
        average_log_probability=average_log_probability,
        ended_with_eos=ended_with_eos,
    )


def strategy_configurations() -> list[DecodingConfig]:
    """Return comparable settings for all five required strategies."""
    shared = {
        "max_new_tokens": 8,
        "eos_token_id": TOKEN_TO_ID["<eos>"],
    }
    return [
        DecodingConfig(strategy="greedy", **shared),
        DecodingConfig(
            strategy="beam",
            beam_width=3,
            length_penalty=0.6,
            **shared,
        ),
        DecodingConfig(
            strategy="temperature",
            temperature=1.3,
            seed=42,
            **shared,
        ),
        DecodingConfig(
            strategy="top_k",
            top_k=3,
            temperature=1.0,
            seed=42,
            **shared,
        ),
        DecodingConfig(
            strategy="top_p",
            top_p=0.85,
            temperature=1.0,
            seed=42,
            **shared,
        ),
    ]


def print_comparison(rows: list[tuple[GenerationResult, ComparisonMetrics]]) -> None:
    """Print a Markdown-style table that is easy to copy into the README."""
    print("\nDecoding comparison")
    print(
        "| Strategy | Output | EOS | Unique tokens | "
        "Repeated bigrams | Avg log P | Time (ms) |"
    )
    print("|---|---|---:|---:|---:|---:|---:|")
    for result, metrics in rows:
        print(
            f"| {result.strategy} | {decode_text(result.token_ids)} "
            f"| {'yes' if metrics.ended_with_eos else 'no'} "
            f"| {metrics.unique_token_ratio:.0%} "
            f"| {metrics.repeated_bigram_ratio:.0%} "
            f"| {metrics.average_log_probability:.3f} "
            f"| {result.elapsed_seconds * 1000:.3f} |"
        )


def main() -> None:
    """Generate the same prompt with every decoding strategy."""
    torch.manual_seed(0)
    model = ToyLanguageModel()
    prompt_token_ids = torch.tensor(
        [[TOKEN_TO_ID["<bos>"], TOKEN_TO_ID["dog"]]],
        dtype=torch.long,
    )

    rows = []
    for config in strategy_configurations():
        result = generate(model, prompt_token_ids, config)
        rows.append((result, calculate_metrics(result)))

    print(f"Prompt: {decode_text(prompt_token_ids)!r}")
    print_comparison(rows)
    print(
        "\nQuality is subjective: read each output yourself. The remaining "
        "columns are measured automatically."
    )


if __name__ == "__main__":
    main()

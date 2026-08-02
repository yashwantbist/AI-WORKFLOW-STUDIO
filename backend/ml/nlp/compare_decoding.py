"""Compare decoding strategies using the same tiny language model.

This demonstration requires no download or training. The toy model stores a
table of next-token scores, allowing the lab to focus on decoding strategies.
"""

import argparse
from dataclasses import dataclass

import torch
from torch import Tensor, nn

try:
    from .decoding import DecodingConfig, GenerationResult
    from .generation_inference import generate
except ImportError:
    from decoding import DecodingConfig, GenerationResult
    from generation_inference import generate


# This is the complete vocabulary understood by the toy model.
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

# Convert words to integer token IDs.
#
# Example:
# "cat" -> 4
# "dog" -> 5
TOKEN_TO_ID = {
    token: token_id
    for token_id, token in enumerate(TOKENS)
}


class ToyLanguageModel(nn.Module):
    """Predict the next token using a hand-written transition table."""

    def __init__(self) -> None:
        super().__init__()

        vocabulary_size = len(TOKENS)

        # Every transition begins with a very low score.
        transitions = torch.full(
            (vocabulary_size, vocabulary_size),
            -6.0,
        )

        def set_choices(
            source: str,
            choices: dict[str, float],
        ) -> None:
            """Set possible next tokens and their scores."""

            for target, score in choices.items():
                source_id = TOKEN_TO_ID[source]
                target_id = TOKEN_TO_ID[target]
                transitions[source_id, target_id] = score

        # Beginning and ending behavior.
        set_choices("<pad>", {"<eos>": 5.0})
        set_choices("<bos>", {"the": 5.0})
        set_choices("<eos>", {"<eos>": 5.0})

        # Possible words after "the".
        set_choices(
            "the",
            {
                "cat": 4.0,
                "dog": 3.4,
                "bird": 3.0,
            },
        )

        # Possible words after each animal.
        set_choices(
            "cat",
            {
                "sat": 4.0,
                "ran": 2.2,
                "sang": 1.8,
            },
        )

        set_choices(
            "dog",
            {
                "ran": 4.0,
                "sat": 2.4,
                "sang": 2.0,
            },
        )

        set_choices(
            "bird",
            {
                "sang": 4.0,
                "sat": 2.2,
                "ran": 1.5,
            },
        )

        # Possible words after actions.
        set_choices(
            "sat",
            {
                "on": 4.0,
                "quickly": 2.7,
                ".": 2.0,
            },
        )

        set_choices(
            "ran",
            {
                "quickly": 4.0,
                ".": 3.1,
                "on": 1.0,
            },
        )

        set_choices(
            "sang",
            {
                ".": 4.0,
                "quickly": 2.5,
            },
        )

        # Complete the remaining sentence paths.
        set_choices(
            "on",
            {
                "mat": 4.0,
                ".": 2.5,
            },
        )

        set_choices("mat", {".": 4.0})
        set_choices("quickly", {".": 4.0})
        set_choices(".", {"<eos>": 5.0})

        # A buffer belongs to the model but is not a trainable parameter.
        self.register_buffer(
            "transition_logits",
            transitions,
        )

    def forward(self, token_ids: Tensor) -> Tensor:
        """Return logits with [batch, sequence, vocabulary] shape."""

        return self.transition_logits[token_ids]


@dataclass(frozen=True)
class ComparisonMetrics:
    """Measurements collected from one generated sequence."""

    unique_token_ratio: float
    repeated_bigram_ratio: float
    average_log_probability: float
    ended_with_eos: bool


def encode_prompt(prompt_text: str) -> Tensor:
    """Convert a typed prompt into token IDs."""

    # This allows the user to enter:
    #
    # "the dog."
    #
    # and converts it into:
    #
    # ["the", "dog", "."]
    normalized_text = prompt_text.lower().replace(".", " . ")
    prompt_words = normalized_text.split()

    if not prompt_words:
        raise ValueError("The prompt cannot be empty.")

    unsupported_words = [
        word
        for word in prompt_words
        if word not in TOKEN_TO_ID
        or word in {"<pad>", "<bos>", "<eos>"}
    ]

    if unsupported_words:
        supported_words = [
            token
            for token in TOKENS
            if token not in {"<pad>", "<bos>", "<eos>"}
        ]

        raise ValueError(
            f"Unsupported prompt words: {unsupported_words}. "
            f"Supported words are: {supported_words}"
        )

    # Add <bos> automatically.
    words_with_bos = ["<bos>", *prompt_words]

    token_ids = [
        TOKEN_TO_ID[word]
        for word in words_with_bos
    ]

    return torch.tensor(
        [token_ids],
        dtype=torch.long,
    )


def decode_text(token_ids: Tensor) -> str:
    """Convert generated token IDs back into readable text."""

    hidden_tokens = {
        "<pad>",
        "<bos>",
        "<eos>",
    }

    words = [
        TOKENS[token_id]
        for token_id in token_ids.squeeze(0).tolist()
        if TOKENS[token_id] not in hidden_tokens
    ]

    text = " ".join(words)

    # Remove the space before periods.
    return text.replace(" .", ".")


def calculate_metrics(
    result: GenerationResult,
) -> ComparisonMetrics:
    """Calculate diversity, repetition, probability, and completion."""

    generated_ids = (
        result.generated_token_ids
        .squeeze(0)
        .tolist()
    )

    ignored_ids = {
        TOKEN_TO_ID["<pad>"],
        TOKEN_TO_ID["<eos>"],
    }

    content_ids = [
        token_id
        for token_id in generated_ids
        if token_id not in ignored_ids
    ]

    if content_ids:
        unique_token_ratio = (
            len(set(content_ids))
            / len(content_ids)
        )
    else:
        unique_token_ratio = 0.0

    bigrams = list(
        zip(
            content_ids,
            content_ids[1:],
        )
    )

    if bigrams:
        repeated_bigram_ratio = (
            1.0
            - len(set(bigrams))
            / len(bigrams)
        )
    else:
        repeated_bigram_ratio = 0.0

    if result.generated_token_count:
        average_log_probability = (
            result.cumulative_log_probability
            / result.generated_token_count
        )
    else:
        average_log_probability = 0.0

    ended_with_eos = bool(
        generated_ids
        and generated_ids[-1]
        == TOKEN_TO_ID["<eos>"]
    )

    return ComparisonMetrics(
        unique_token_ratio=unique_token_ratio,
        repeated_bigram_ratio=repeated_bigram_ratio,
        average_log_probability=average_log_probability,
        ended_with_eos=ended_with_eos,
    )


def strategy_configurations() -> list[DecodingConfig]:
    """Create configurations for all five decoding strategies."""

    shared_settings = {
        "max_new_tokens": 8,
        "eos_token_id": TOKEN_TO_ID["<eos>"],
    }

    return [
        DecodingConfig(
            strategy="greedy",
            **shared_settings,
        ),
        DecodingConfig(
            strategy="beam",
            beam_width=3,
            length_penalty=0.6,
            **shared_settings,
        ),
        DecodingConfig(
            strategy="temperature",
            temperature=1.3,
            seed=42,
            **shared_settings,
        ),
        DecodingConfig(
            strategy="top_k",
            top_k=3,
            temperature=1.0,
            seed=42,
            **shared_settings,
        ),
        DecodingConfig(
            strategy="top_p",
            top_p=0.85,
            temperature=1.0,
            seed=42,
            **shared_settings,
        ),
    ]


def print_comparison(
    rows: list[
        tuple[
            GenerationResult,
            ComparisonMetrics,
        ]
    ],
) -> None:
    """Print the decoding results as a Markdown table."""

    print("\nDecoding comparison")

    print(
        "| Strategy | Output | EOS | Unique tokens | "
        "Repeated bigrams | Avg log P | Time (ms) |"
    )

    print(
        "|---|---|---:|---:|---:|---:|---:|"
    )

    for result, metrics in rows:
        output_text = decode_text(result.token_ids)

        ended_with_eos = (
            "yes"
            if metrics.ended_with_eos
            else "no"
        )

        elapsed_milliseconds = (
            result.elapsed_seconds * 1000
        )

        print(
            f"| {result.strategy} "
            f"| {output_text} "
            f"| {ended_with_eos} "
            f"| {metrics.unique_token_ratio:.0%} "
            f"| {metrics.repeated_bigram_ratio:.0%} "
            f"| {metrics.average_log_probability:.3f} "
            f"| {elapsed_milliseconds:.3f} |"
        )


def parse_arguments() -> argparse.Namespace:
    """Read the prompt supplied through PowerShell."""

    parser = argparse.ArgumentParser(
        description=(
            "Compare greedy, beam, temperature, "
            "top-k, and top-p decoding."
        )
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="the",
        help=(
            "Prompt constructed from the toy vocabulary. "
            'Example: --prompt "the dog"'
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Generate the same prompt with every decoding strategy."""

    arguments = parse_arguments()

    torch.manual_seed(0)

    model = ToyLanguageModel()

    prompt_token_ids = encode_prompt(
        arguments.prompt
    )

    rows = []

    for config in strategy_configurations():
        result = generate(
            model,
            prompt_token_ids,
            config,
        )

        metrics = calculate_metrics(result)

        rows.append(
            (
                result,
                metrics,
            )
        )

    print(
        f"Prompt: "
        f"{decode_text(prompt_token_ids)!r}"
    )

    print_comparison(rows)

    print(
        "\nQuality is subjective: read each output yourself. "
        "The remaining columns are measured automatically."
    )


if __name__ == "__main__":
    main()
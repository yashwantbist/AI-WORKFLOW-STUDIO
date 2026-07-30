"""Train the custom encoder and visualize its learned attention weights."""

import argparse
from pathlib import Path
from time import perf_counter

import torch
from torch import Tensor

try:
    # Supports: python -m backend.ml.nlp.visualize_attention
    from .dataset import CLASS_NAMES, DataBundle, build_data_bundle
    from .train_transformer_classifier import (
        BATCH_SIZE,
        LEARNING_RATE,
        RANDOM_SEED,
        SAMPLES_PER_CLASS,
        run_training_experiment,
        select_device,
        set_random_seed,
        synchronize_device,
    )
    from .transformer_encoder import AttentionTextClassifier
except ImportError:
    # Supports: python backend/ml/nlp/visualize_attention.py
    from dataset import CLASS_NAMES, DataBundle, build_data_bundle
    from train_transformer_classifier import (
        BATCH_SIZE,
        LEARNING_RATE,
        RANDOM_SEED,
        SAMPLES_PER_CLASS,
        run_training_experiment,
        select_device,
        set_random_seed,
        synchronize_device,
    )
    from transformer_encoder import AttentionTextClassifier


MODEL_DIMENSION = 32
NUMBER_OF_HEADS = 4
NUMBER_OF_LAYERS = 2
FEED_FORWARD_DIMENSION = 64
MAXIMUM_SEQUENCE_LENGTH = 128
DEFAULT_TEXT = "this course is helpful and i recommend it"
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[3]
    / "images"
    / "attention_weights.png"
)


def create_model(
    data_bundle: DataBundle,
    dropout: float,
) -> AttentionTextClassifier:
    """Create the same small architecture used in the Day 9 comparison."""
    return AttentionTextClassifier(
        vocabulary_size=len(data_bundle.vocabulary),
        number_of_classes=len(CLASS_NAMES),
        padding_id=data_bundle.vocabulary.padding_id,
        model_dimension=MODEL_DIMENSION,
        number_of_heads=NUMBER_OF_HEADS,
        number_of_layers=NUMBER_OF_LAYERS,
        feed_forward_dimension=FEED_FORWARD_DIMENSION,
        dropout=dropout,
        maximum_sequence_length=MAXIMUM_SEQUENCE_LENGTH,
    )


def train_model(
    data_bundle: DataBundle,
    device: torch.device,
    epochs: int,
) -> tuple[AttentionTextClassifier, float, float]:
    """Train a model and return it with measured train/validation accuracy."""
    output = run_training_experiment(
        model_name="Hand-built Transformer encoder",
        model_factory=lambda: create_model(data_bundle, dropout=0.1),
        data_bundle=data_bundle,
        device=device,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        random_seed=RANDOM_SEED,
        print_epochs=True,
    )
    model = create_model(data_bundle, dropout=0.1).to(device)
    model.load_state_dict(output.model_state)
    model.eval()
    return (
        model,
        output.result.final_training_accuracy,
        output.result.final_validation_accuracy,
    )


def prepare_sentence(
    text: str,
    data_bundle: DataBundle,
    device: torch.device,
) -> tuple[list[str], Tensor, Tensor]:
    """Tokenize a sentence through the existing project vocabulary."""
    token_ids_list = data_bundle.vocabulary.encode(text)
    token_ids = torch.tensor(
        [token_ids_list],
        dtype=torch.long,
        device=device,
    )
    attention_mask = token_ids.ne(data_bundle.vocabulary.padding_id)

    id_to_token = {
        token_id: token
        for token, token_id in data_bundle.vocabulary.to_dict().items()
    }
    tokens = [
        id_to_token.get(token_id, "<unk>")
        for token_id in token_ids_list
    ]
    return tokens, token_ids, attention_mask


def measure_inference_latency(
    model: AttentionTextClassifier,
    token_ids: Tensor,
    attention_mask: Tensor,
    device: torch.device,
    repetitions: int = 200,
) -> float:
    """Return average end-to-end latency in milliseconds per sentence."""
    if repetitions < 1:
        raise ValueError("repetitions must be positive")

    model.eval()
    with torch.inference_mode():
        for _ in range(10):
            model(token_ids, attention_mask)

        synchronize_device(device)
        start = perf_counter()
        for _ in range(repetitions):
            model(token_ids, attention_mask)
        synchronize_device(device)

    return 1_000 * (perf_counter() - start) / repetitions


def save_attention_plot(
    tokens: list[str],
    attention_weights: Tensor,
    output_path: Path,
) -> None:
    """Save one heatmap per attention head.

    ``attention_weights`` must have shape
    ``[heads, query_sequence, key_sequence]``.
    """
    import matplotlib.pyplot as plt

    if attention_weights.ndim != 3:
        raise ValueError(
            "attention_weights must have shape [heads, query, key]"
        )

    number_of_heads = attention_weights.size(0)
    figure, axes = plt.subplots(
        1,
        number_of_heads,
        figsize=(5 * number_of_heads, 4.5),
        squeeze=False,
    )

    for head_index in range(number_of_heads):
        axis = axes[0, head_index]
        image = axis.imshow(
            attention_weights[head_index].detach().cpu().numpy(),
            cmap="Blues",
            vmin=0,
            vmax=1,
        )
        axis.set_title(f"Head {head_index + 1}")
        axis.set_xlabel("Key token attended to")
        axis.set_ylabel("Query token")
        axis.set_xticks(range(len(tokens)), labels=tokens, rotation=45)
        axis.set_yticks(range(len(tokens)), labels=tokens)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    figure.suptitle("Last Encoder Layer: Multi-Head Self-Attention")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the custom encoder and visualize its attention.",
    )
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--latency-runs", type=int, default=200)
    arguments = parser.parse_args()

    if arguments.epochs < 1:
        parser.error("--epochs must be at least 1")
    if arguments.latency_runs < 1:
        parser.error("--latency-runs must be at least 1")
    return arguments


def main() -> None:
    """Train, predict, benchmark, and save per-head attention heatmaps."""
    arguments = parse_arguments()
    set_random_seed(RANDOM_SEED)
    device = select_device()
    data_bundle = build_data_bundle(
        samples_per_class=SAMPLES_PER_CLASS,
        random_seed=RANDOM_SEED,
    )
    model, training_accuracy, validation_accuracy = train_model(
        data_bundle=data_bundle,
        device=device,
        epochs=arguments.epochs,
    )
    tokens, token_ids, attention_mask = prepare_sentence(
        text=arguments.text,
        data_bundle=data_bundle,
        device=device,
    )

    with torch.inference_mode():
        logits, attention_by_layer = model(
            token_ids,
            attention_mask,
            return_attention=True,
        )

    try:
        selected_attention = attention_by_layer[arguments.layer][0]
    except IndexError as error:
        raise ValueError(
            f"Layer {arguments.layer} does not exist; "
            f"choose from {-len(attention_by_layer)} to "
            f"{len(attention_by_layer) - 1}"
        ) from error

    latency_ms = measure_inference_latency(
        model=model,
        token_ids=token_ids,
        attention_mask=attention_mask,
        device=device,
        repetitions=arguments.latency_runs,
    )
    predicted_class = CLASS_NAMES[logits.argmax(dim=1).item()]
    probabilities = torch.softmax(logits, dim=1)[0]

    save_attention_plot(
        tokens=tokens,
        attention_weights=selected_attention,
        output_path=arguments.output,
    )

    print(f"\nText: {arguments.text}")
    print(
        f"Prediction: {predicted_class} "
        f"({probabilities.max().item() * 100:.2f}% confidence)"
    )
    print(f"Final training accuracy: {training_accuracy:.2f}%")
    print(f"Final validation accuracy: {validation_accuracy:.2f}%")
    print(f"Average inference latency: {latency_ms:.4f} ms/sentence")
    print(f"Saved attention visualization to: {arguments.output}")


if __name__ == "__main__":
    main()

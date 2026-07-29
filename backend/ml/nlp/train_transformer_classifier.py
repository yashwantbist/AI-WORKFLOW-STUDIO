"""Train a Transformer encoder on the Day 8 sentiment dataset."""

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Final

import torch
from torch import Tensor, nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader

try:
    # Supports: python -m backend.ml.nlp.train_transformer_classifier
    from .dataset import (
        CLASS_NAMES,
        DataBundle,
        build_data_bundle,
        create_data_loaders,
    )
    from .transformer_model import TransformerTextClassifier
except ImportError:
    # Supports: python backend/ml/nlp/train_transformer_classifier.py
    from dataset import (
        CLASS_NAMES,
        DataBundle,
        build_data_bundle,
        create_data_loaders,
    )
    from transformer_model import TransformerTextClassifier


BATCH_SIZE: Final = 16
LEARNING_RATE: Final = 0.01
EPOCHS: Final = 10
RANDOM_SEED: Final = 42
SAMPLES_PER_CLASS: Final = 120
MODEL_DIMENSION: Final = 32
NUMBER_OF_HEADS: Final = 4
NUMBER_OF_LAYERS: Final = 2
FEED_FORWARD_DIMENSION: Final = 64
DROPOUT: Final = 0.1
MAXIMUM_SEQUENCE_LENGTH: Final = 128
DEFAULT_CHECKPOINT: Final = (
    Path(__file__).resolve().parent
    / "artifacts"
    / "transformer_classifier.pt"
)


@dataclass(frozen=True)
class EpochMetrics:
    """Metrics measured after one training epoch."""

    epoch: int
    training_loss: float
    training_accuracy: float
    validation_accuracy: float


@dataclass(frozen=True)
class TrainingResult:
    """Final metrics and history from one model experiment."""

    model_name: str
    final_training_loss: float
    final_training_accuracy: float
    final_validation_accuracy: float
    training_time_seconds: float
    parameter_count: int
    history: tuple[EpochMetrics, ...]


@dataclass(frozen=True)
class TrainingOutput:
    """Measured results plus weights for optional checkpoint saving."""

    result: TrainingResult
    model_state: dict[str, Tensor]


def select_device() -> torch.device:
    """Use the best available PyTorch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize_device(device: torch.device) -> None:
    """Wait for accelerator work before reading the timer."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def set_random_seed(random_seed: int) -> None:
    """Reset PyTorch randomness so experiments are repeatable."""
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)


def train_one_epoch(
    model: nn.Module,
    training_loader: DataLoader,
    loss_function: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Update the model once over the training data."""
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_examples = 0

    for token_ids, attention_mask, labels in training_loader:
        token_ids = token_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(token_ids, attention_mask)
        loss = loss_function(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct_predictions += (
            logits.argmax(dim=1) == labels
        ).sum().item()
        total_examples += batch_size

    return (
        total_loss / total_examples,
        100 * correct_predictions / total_examples,
    )


def calculate_accuracy(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> float:
    """Measure accuracy without changing the model."""
    model.eval()
    correct_predictions = 0
    total_examples = 0

    with torch.inference_mode():
        for token_ids, attention_mask, labels in data_loader:
            token_ids = token_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            logits = model(token_ids, attention_mask)
            correct_predictions += (
                logits.argmax(dim=1) == labels
            ).sum().item()
            total_examples += labels.size(0)

    return 100 * correct_predictions / total_examples


def copy_state_to_cpu(model: nn.Module) -> dict[str, Tensor]:
    """Copy weights to CPU so a checkpoint works on another machine."""
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.state_dict().items()
    }


def run_training_experiment(
    model_name: str,
    model_factory: Callable[[], nn.Module],
    data_bundle: DataBundle,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    random_seed: int = RANDOM_SEED,
    print_epochs: bool = True,
) -> TrainingOutput:
    """Train any text classifier under the shared comparison settings."""
    set_random_seed(random_seed)
    training_loader, validation_loader = create_data_loaders(
        data_bundle=data_bundle,
        batch_size=batch_size,
        random_seed=random_seed,
    )
    model = model_factory().to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(f"\nModel: {model_name} | Trainable parameters: {parameter_count:,}")

    history: list[EpochMetrics] = []
    training_time_seconds = 0.0

    for epoch in range(1, epochs + 1):
        synchronize_device(device)
        training_start = perf_counter()

        training_loss, training_accuracy = train_one_epoch(
            model=model,
            training_loader=training_loader,
            loss_function=loss_function,
            optimizer=optimizer,
            device=device,
        )

        synchronize_device(device)
        training_time_seconds += perf_counter() - training_start
        validation_accuracy = calculate_accuracy(
            model,
            validation_loader,
            device,
        )
        epoch_metrics = EpochMetrics(
            epoch=epoch,
            training_loss=training_loss,
            training_accuracy=training_accuracy,
            validation_accuracy=validation_accuracy,
        )
        history.append(epoch_metrics)

        if print_epochs:
            print(
                f"Epoch {epoch}/{epochs} | "
                f"Training loss: {training_loss:.4f} | "
                f"Training accuracy: {training_accuracy:.2f}% | "
                f"Validation accuracy: {validation_accuracy:.2f}%"
            )

    print(f"Measured training time: {training_time_seconds:.4f} seconds")

    final_metrics = history[-1]
    return TrainingOutput(
        result=TrainingResult(
            model_name=model_name,
            final_training_loss=final_metrics.training_loss,
            final_training_accuracy=final_metrics.training_accuracy,
            final_validation_accuracy=final_metrics.validation_accuracy,
            training_time_seconds=training_time_seconds,
            parameter_count=parameter_count,
            history=tuple(history),
        ),
        model_state=copy_state_to_cpu(model),
    )


def build_transformer_factory(
    vocabulary_size: int,
    number_of_classes: int,
    padding_id: int,
    model_dimension: int,
    number_of_heads: int,
    number_of_layers: int,
    feed_forward_dimension: int,
    dropout: float,
) -> Callable[[], TransformerTextClassifier]:
    """Return a function that creates a fresh Transformer."""

    def create_model() -> TransformerTextClassifier:
        return TransformerTextClassifier(
            vocabulary_size=vocabulary_size,
            number_of_classes=number_of_classes,
            padding_id=padding_id,
            model_dimension=model_dimension,
            number_of_heads=number_of_heads,
            number_of_layers=number_of_layers,
            feed_forward_dimension=feed_forward_dimension,
            dropout=dropout,
            maximum_sequence_length=MAXIMUM_SEQUENCE_LENGTH,
        )

    return create_model


def save_checkpoint(
    output: TrainingOutput,
    data_bundle: DataBundle,
    checkpoint_path: Path,
    model_configuration: dict[str, int | float],
) -> None:
    """Save model weights, vocabulary, and architecture settings."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "model_state_dict": output.model_state,
            "model_configuration": model_configuration,
            "vocabulary": data_bundle.vocabulary.to_dict(),
            "class_names": list(CLASS_NAMES),
        },
        checkpoint_path,
    )
    print(f"Saved Transformer checkpoint to: {checkpoint_path}")


def parse_arguments() -> argparse.Namespace:
    """Read command-line training options."""
    parser = argparse.ArgumentParser(
        description="Train a Transformer sentiment classifier.",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--model-dimension", type=int, default=MODEL_DIMENSION)
    parser.add_argument("--heads", type=int, default=NUMBER_OF_HEADS)
    parser.add_argument("--layers", type=int, default=NUMBER_OF_LAYERS)
    parser.add_argument(
        "--feed-forward-dimension",
        type=int,
        default=FEED_FORWARD_DIMENSION,
    )
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument("--no-save", action="store_true")
    arguments = parser.parse_args()

    if arguments.epochs < 1:
        parser.error("--epochs must be at least 1")
    if arguments.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if arguments.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if arguments.model_dimension < 1:
        parser.error("--model-dimension must be positive")
    if arguments.heads < 1:
        parser.error("--heads must be positive")
    if arguments.model_dimension % arguments.heads != 0:
        parser.error("--model-dimension must be divisible by --heads")
    if arguments.layers < 1:
        parser.error("--layers must be positive")
    if arguments.feed_forward_dimension < 1:
        parser.error("--feed-forward-dimension must be positive")
    if not 0 <= arguments.dropout < 1:
        parser.error("--dropout must be at least 0 and less than 1")

    return arguments


def main() -> None:
    """Train the configured Transformer and print measured results."""
    arguments = parse_arguments()
    device = select_device()
    data_bundle = build_data_bundle(
        samples_per_class=SAMPLES_PER_CLASS,
        random_seed=RANDOM_SEED,
    )
    model_configuration: dict[str, int | float] = {
        "model_dimension": arguments.model_dimension,
        "number_of_heads": arguments.heads,
        "number_of_layers": arguments.layers,
        "feed_forward_dimension": arguments.feed_forward_dimension,
        "dropout": arguments.dropout,
        "maximum_sequence_length": MAXIMUM_SEQUENCE_LENGTH,
    }
    model_factory = build_transformer_factory(
        vocabulary_size=len(data_bundle.vocabulary),
        number_of_classes=len(CLASS_NAMES),
        padding_id=data_bundle.vocabulary.padding_id,
        model_dimension=arguments.model_dimension,
        number_of_heads=arguments.heads,
        number_of_layers=arguments.layers,
        feed_forward_dimension=arguments.feed_forward_dimension,
        dropout=arguments.dropout,
    )

    print(f"Training on: {device}")
    print(
        f"Dataset: same generated sentiment split as Day 8 | "
        f"Training examples: {len(data_bundle.training_dataset)} | "
        f"Validation examples: {len(data_bundle.test_dataset)}"
    )
    print(f"Vocabulary size: {len(data_bundle.vocabulary)} tokens")
    print(
        f"Transformer: dimension={arguments.model_dimension}, "
        f"heads={arguments.heads}, layers={arguments.layers}"
    )

    output = run_training_experiment(
        model_name="Transformer encoder",
        model_factory=model_factory,
        data_bundle=data_bundle,
        device=device,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
    )

    if not arguments.no_save:
        save_checkpoint(
            output=output,
            data_bundle=data_bundle,
            checkpoint_path=arguments.checkpoint,
            model_configuration=model_configuration,
        )


if __name__ == "__main__":
    main()
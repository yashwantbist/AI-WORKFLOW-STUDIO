"""Train and compare embedding-based sentiment classifiers."""

import argparse
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Final

import torch
from torch import Tensor, nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader

try:
    # Supports: python -m backend.ml.nlp.train_text_classifier
    from .dataset import (
        CLASS_NAMES,
        DataBundle,
        build_data_bundle,
        create_data_loaders,
    )
    from .model import MeanPoolingTextClassifier
except ImportError:
    # Supports: python backend/ml/nlp/train_text_classifier.py
    from dataset import (
        CLASS_NAMES,
        DataBundle,
        build_data_bundle,
        create_data_loaders,
    )
    from model import MeanPoolingTextClassifier


BATCH_SIZE: Final = 16
LEARNING_RATE: Final = 0.01
EPOCHS: Final = 10
RANDOM_SEED: Final = 42
SAMPLES_PER_CLASS: Final = 120
EMBEDDING_SIZES: Final = (32, 128)
DEFAULT_CHECKPOINT: Final = (
    Path(__file__).resolve().parent
    / "artifacts"
    / "text_classifier.pt"
)


@dataclass(frozen=True)
class ExperimentResult:
    """Final measured values from one embedding-size experiment."""

    embedding_size: int
    final_training_loss: float
    final_training_accuracy: float
    final_test_accuracy: float
    training_time_seconds: float


@dataclass(frozen=True)
class ExperimentOutput:
    """Measured result and model weights from one experiment."""

    result: ExperimentResult
    model_state: dict[str, Tensor]


def select_device() -> torch.device:
    """Use the best available PyTorch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def synchronize_device(device: torch.device) -> None:
    """Wait for queued accelerator work before recording a time."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def set_random_seed() -> None:
    """Reset random state for a fair embedding-size comparison."""
    torch.manual_seed(RANDOM_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)


def train_one_epoch(
    model: nn.Module,
    training_loader: DataLoader,
    loss_function: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train once and return example-weighted loss and accuracy."""
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_examples = 0

    for token_ids, attention_mask, labels in training_loader:
        token_ids = token_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        predictions = model(token_ids, attention_mask)
        loss = loss_function(predictions, labels)
        loss.backward()
        optimizer.step()

        current_batch_size = labels.size(0)
        total_loss += loss.item() * current_batch_size
        correct_predictions += (
            predictions.argmax(dim=1) == labels
        ).sum().item()
        total_examples += current_batch_size

    return (
        total_loss / total_examples,
        100 * correct_predictions / total_examples,
    )


def calculate_accuracy(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> float:
    """Return classification accuracy for a data loader."""
    model.eval()
    correct_predictions = 0
    total_examples = 0

    with torch.inference_mode():
        for token_ids, attention_mask, labels in data_loader:
            token_ids = token_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            predictions = model(token_ids, attention_mask)
            correct_predictions += (
                predictions.argmax(dim=1) == labels
            ).sum().item()
            total_examples += labels.size(0)

    return 100 * correct_predictions / total_examples


def copy_state_to_cpu(model: nn.Module) -> dict[str, Tensor]:
    """Copy model parameters to CPU for a portable checkpoint."""
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.state_dict().items()
    }


def run_experiment(
    embedding_size: int,
    data_bundle: DataBundle,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> ExperimentOutput:
    """Train one embedding configuration and return measured results."""
    set_random_seed()
    training_loader, test_loader = create_data_loaders(
        data_bundle=data_bundle,
        batch_size=batch_size,
        random_seed=RANDOM_SEED,
    )
    model = MeanPoolingTextClassifier(
        vocabulary_size=len(data_bundle.vocabulary),
        embedding_size=embedding_size,
        number_of_classes=len(CLASS_NAMES),
        padding_id=data_bundle.vocabulary.padding_id,
    ).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    print(f"\nExperiment: embedding_size={embedding_size}")

    final_training_loss = 0.0
    final_training_accuracy = 0.0
    final_test_accuracy = 0.0
    training_time_seconds = 0.0

    for epoch in range(1, epochs + 1):
        synchronize_device(device)
        training_start = perf_counter()

        final_training_loss, final_training_accuracy = train_one_epoch(
            model=model,
            training_loader=training_loader,
            loss_function=loss_function,
            optimizer=optimizer,
            device=device,
        )

        synchronize_device(device)
        training_time_seconds += perf_counter() - training_start
        final_test_accuracy = calculate_accuracy(
            model,
            test_loader,
            device,
        )

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Average training loss: {final_training_loss:.4f} | "
            f"Training accuracy: {final_training_accuracy:.2f}% | "
            f"Test accuracy: {final_test_accuracy:.2f}%"
        )

    print(f"Measured training time: {training_time_seconds:.4f} seconds")

    return ExperimentOutput(
        result=ExperimentResult(
            embedding_size=embedding_size,
            final_training_loss=final_training_loss,
            final_training_accuracy=final_training_accuracy,
            final_test_accuracy=final_test_accuracy,
            training_time_seconds=training_time_seconds,
        ),
        model_state=copy_state_to_cpu(model),
    )


def print_comparison(results: list[ExperimentResult]) -> None:
    """Print only metrics measured by the completed experiments."""
    print("\nMeasured NLP results")
    print(
        "Embedding | Final loss | Train accuracy | "
        "Test accuracy | Training time"
    )
    print("-" * 76)

    for result in results:
        print(
            f"{result.embedding_size:^9} | "
            f"{result.final_training_loss:>10.4f} | "
            f"{result.final_training_accuracy:>13.2f}% | "
            f"{result.final_test_accuracy:>12.2f}% | "
            f"{result.training_time_seconds:>12.4f}s"
        )


def save_checkpoint(
    output: ExperimentOutput,
    data_bundle: DataBundle,
    checkpoint_path: Path,
) -> None:
    """Save model weights and vocabulary for the inference script."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "model_state_dict": output.model_state,
            "embedding_size": output.result.embedding_size,
            "vocabulary": data_bundle.vocabulary.to_dict(),
            "class_names": list(CLASS_NAMES),
        },
        checkpoint_path,
    )
    print(
        f"Saved best measured model "
        f"(embedding_size={output.result.embedding_size}) to: "
        f"{checkpoint_path}"
    )


def parse_arguments() -> argparse.Namespace:
    """Parse options for one run or the two-part embedding lab."""
    parser = argparse.ArgumentParser(
        description="Train an embedding-based sentiment classifier.",
    )
    parser.add_argument(
        "--embedding-size",
        type=int,
        choices=EMBEDDING_SIZES,
        default=32,
        help="Embedding size for one experiment (default: 32).",
    )
    parser.add_argument(
        "--compare-embeddings",
        action="store_true",
        help="Run both embedding_size=32 and embedding_size=128.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"Number of training epochs (default: {EPOCHS}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Examples per batch (default: {BATCH_SIZE}).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=LEARNING_RATE,
        help=f"Adam learning rate (default: {LEARNING_RATE}).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path for the best measured model checkpoint.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save a model checkpoint after training.",
    )
    arguments = parser.parse_args()

    if arguments.epochs < 1:
        parser.error("--epochs must be at least 1")
    if arguments.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if arguments.learning_rate <= 0:
        parser.error("--learning-rate must be positive")

    return arguments


def main() -> None:
    """Run the requested NLP experiments and report measured values."""
    arguments = parse_arguments()
    device = select_device()
    data_bundle = build_data_bundle(
        samples_per_class=SAMPLES_PER_CLASS,
        random_seed=RANDOM_SEED,
    )
    embedding_sizes = (
        EMBEDDING_SIZES
        if arguments.compare_embeddings
        else (arguments.embedding_size,)
    )

    print(f"Training on: {device}")
    print(
        f"Dataset: generated sentiment | "
        f"Training examples: {len(data_bundle.training_dataset)} | "
        f"Test examples: {len(data_bundle.test_dataset)}"
    )
    print(f"Vocabulary size: {len(data_bundle.vocabulary)} tokens")

    outputs = [
        run_experiment(
            embedding_size=embedding_size,
            data_bundle=data_bundle,
            device=device,
            epochs=arguments.epochs,
            batch_size=arguments.batch_size,
            learning_rate=arguments.learning_rate,
        )
        for embedding_size in embedding_sizes
    ]
    print_comparison([output.result for output in outputs])

    if not arguments.no_save:
        best_output = max(
            outputs,
            key=lambda output: output.result.final_test_accuracy,
        )
        save_checkpoint(
            best_output,
            data_bundle,
            arguments.checkpoint,
        )


if __name__ == "__main__":
    main()
"""Train and evaluate a convolutional neural network on MNIST."""

import argparse
from dataclasses import dataclass
from time import perf_counter

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

try:
    # Supports: python -m backend.ml.train_cnn_mnist
    from .cnn_model import MNISTCNN
    from .train_mnist import (
        BATCH_SIZE,
        DATA_DIRECTORY,
        EPOCHS,
        LEARNING_RATE,
        select_device,
    )
except ImportError:
    # Supports: python backend/ml/train_cnn_mnist.py
    from cnn_model import MNISTCNN
    from train_mnist import (
        BATCH_SIZE,
        DATA_DIRECTORY,
        EPOCHS,
        LEARNING_RATE,
        select_device,
    )


RANDOM_SEED = 42


@dataclass(frozen=True)
class ExperimentResult:
    """Final measured values from one CNN training experiment."""

    kernel_size: int
    final_training_loss: float
    final_training_accuracy: float
    final_test_accuracy: float
    training_time_seconds: float


def create_data_loaders(random_seed: int) -> tuple[DataLoader, DataLoader]:
    """Reuse the earlier MNIST ToTensor data-loading pipeline."""
    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root=DATA_DIRECTORY,
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        root=DATA_DIRECTORY,
        train=False,
        download=True,
        transform=transform,
    )

    shuffle_generator = torch.Generator().manual_seed(random_seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=shuffle_generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    return train_loader, test_loader


def synchronize_device(device: torch.device) -> None:
    """Wait for queued accelerator work before recording a time."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    loss_function: nn.Module,
    optimizer: Adam,
    device: torch.device,
) -> tuple[float, float]:
    """Train for one epoch and return average loss and training accuracy."""
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_examples = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        predictions = model(images)
        loss = loss_function(predictions, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct_predictions += (
            predictions.argmax(dim=1) == labels
        ).sum().item()
        total_examples += batch_size

    average_loss = total_loss / total_examples
    training_accuracy = 100 * correct_predictions / total_examples

    return average_loss, training_accuracy


def calculate_accuracy(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> float:
    """Return classification accuracy for a data loader."""
    model.eval()
    correct_predictions = 0
    total_examples = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            predictions = model(images)
            predicted_classes = predictions.argmax(dim=1)

            correct_predictions += (predicted_classes == labels).sum().item()
            total_examples += labels.size(0)

    return 100 * correct_predictions / total_examples


def run_experiment(
    kernel_size: int,
    device: torch.device,
) -> ExperimentResult:
    """Train one CNN configuration for five epochs and return its results."""
    torch.manual_seed(RANDOM_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    train_loader, test_loader = create_data_loaders(RANDOM_SEED)
    model = MNISTCNN(kernel_size=kernel_size).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\nCNN experiment: kernel_size={kernel_size}")

    final_training_loss = 0.0
    final_training_accuracy = 0.0
    final_test_accuracy = 0.0
    training_time_seconds = 0.0

    for epoch in range(1, EPOCHS + 1):
        synchronize_device(device)
        training_start = perf_counter()

        final_training_loss, final_training_accuracy = train_one_epoch(
            model,
            train_loader,
            loss_function,
            optimizer,
            device,
        )

        synchronize_device(device)
        training_time_seconds += perf_counter() - training_start
        final_test_accuracy = calculate_accuracy(model, test_loader, device)

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Average training loss: {final_training_loss:.4f} | "
            f"Training accuracy: {final_training_accuracy:.2f}% | "
            f"Test accuracy: {final_test_accuracy:.2f}%"
        )

    print(f"Measured training time: {training_time_seconds:.2f} seconds")

    return ExperimentResult(
        kernel_size=kernel_size,
        final_training_loss=final_training_loss,
        final_training_accuracy=final_training_accuracy,
        final_test_accuracy=final_test_accuracy,
        training_time_seconds=training_time_seconds,
    )


def print_comparison(results: list[ExperimentResult]) -> None:
    """Print a measured comparison without assuming which model is better."""
    print("\nMeasured CNN results")
    print(
        "Kernel | Final loss | Train accuracy | "
        "Test accuracy | Training time"
    )
    print("-" * 72)

    for result in results:
        print(
            f"{result.kernel_size:^6} | "
            f"{result.final_training_loss:>10.4f} | "
            f"{result.final_training_accuracy:>13.2f}% | "
            f"{result.final_test_accuracy:>12.2f}% | "
            f"{result.training_time_seconds:>12.2f}s"
        )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line options for one run or the kernel comparison lab."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate a CNN on the MNIST dataset.",
    )
    parser.add_argument(
        "--kernel-size",
        type=int,
        choices=(3, 5),
        default=3,
        help="Convolution kernel size for a single experiment (default: 3).",
    )
    parser.add_argument(
        "--compare-kernels",
        action="store_true",
        help="Run both kernel_size=3 and kernel_size=5 experiments.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the requested CNN experiment and print only measured values."""
    arguments = parse_arguments()
    device = select_device()
    kernel_sizes = (3, 5) if arguments.compare_kernels else (
        arguments.kernel_size,
    )

    print(f"Training on: {device}")
    results = [
        run_experiment(kernel_size=kernel_size, device=device)
        for kernel_size in kernel_sizes
    ]
    print_comparison(results)


if __name__ == "__main__":
    main()
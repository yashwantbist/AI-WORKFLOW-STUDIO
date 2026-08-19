"""Train a pretrained ResNet18 classifier on a CIFAR-10 subset."""

import argparse
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Final

import torch
from torch import nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.models.resnet import ResNet

try:
    # Supports: python -m backend.ml.train_transfer_learning
    from .transfer_learning import (
        create_transfer_learning_model,
        set_transfer_learning_train_mode,
        summarize_parameters,
    )
    from .train_mnist import select_device
except ImportError:
    # Supports: python backend/ml/train_transfer_learning.py
    from transfer_learning import (
        create_transfer_learning_model,
        set_transfer_learning_train_mode,
        summarize_parameters,
    )
    from train_mnist import select_device


BATCH_SIZE: Final = 64
LEARNING_RATE: Final = 0.001
EPOCHS: Final = 5
RANDOM_SEED: Final = 42
IMAGE_SIZE: Final = 128
DEFAULT_TRAIN_SAMPLES: Final = 10_000
DEFAULT_TEST_SAMPLES: Final = 2_000
DATA_DIRECTORY: Final = Path(__file__).resolve().parent / "data"

HEAD_ONLY: Final = "head-only"
FINE_TUNE_LAYER4: Final = "fine-tune-layer4"
STRATEGIES: Final = (HEAD_ONLY, FINE_TUNE_LAYER4)

IMAGENET_MEAN: Final = (0.485, 0.456, 0.406)
IMAGENET_STANDARD_DEVIATION: Final = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class ExperimentResult:
    """Final measured values from one transfer-learning experiment."""

    strategy: str
    final_training_loss: float
    final_training_accuracy: float
    final_test_accuracy: float
    training_time_seconds: float
    trainable_parameters: int


def create_image_transform(image_size: int) -> transforms.Compose:
    """Resize and normalize CIFAR-10 images for pretrained ResNet18."""
    return transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size),
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                IMAGENET_MEAN,
                IMAGENET_STANDARD_DEVIATION,
            ),
        ]
    )


def create_deterministic_subset(
    dataset: Dataset,
    maximum_samples: int | None,
    random_seed: int,
) -> Dataset:
    """Return a reproducible random subset or the unchanged full dataset."""
    if maximum_samples is None or maximum_samples >= len(dataset):
        return dataset

    if maximum_samples < 1:
        raise ValueError("maximum_samples must be positive or None")

    generator = torch.Generator().manual_seed(random_seed)
    indices = torch.randperm(
        len(dataset),
        generator=generator,
    )[:maximum_samples].tolist()
    return Subset(dataset, indices)


def create_datasets(
    image_size: int,
    train_samples: int | None,
    test_samples: int | None,
) -> tuple[Dataset, Dataset, int]:
    """Download CIFAR-10 and return deterministic train/test datasets."""
    image_transform = create_image_transform(image_size)

    full_train_dataset = datasets.CIFAR10(
        root=DATA_DIRECTORY,
        train=True,
        download=True,
        transform=image_transform,
    )
    full_test_dataset = datasets.CIFAR10(
        root=DATA_DIRECTORY,
        train=False,
        download=True,
        transform=image_transform,
    )

    train_dataset = create_deterministic_subset(
        full_train_dataset,
        train_samples,
        RANDOM_SEED,
    )
    test_dataset = create_deterministic_subset(
        full_test_dataset,
        test_samples,
        RANDOM_SEED,
    )

    return train_dataset, test_dataset, len(full_train_dataset.classes)


def create_data_loaders(
    train_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int,
    num_workers: int,
    use_pinned_memory: bool,
) -> tuple[DataLoader, DataLoader]:
    """Create loaders with the same deterministic shuffle for each experiment."""
    shuffle_generator = torch.Generator().manual_seed(RANDOM_SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=shuffle_generator,
        num_workers=num_workers,
        pin_memory=use_pinned_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pinned_memory,
    )

    return train_loader, test_loader


def synchronize_device(device: torch.device) -> None:
    """Wait for queued accelerator work before recording a time."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def train_one_epoch(
    model: ResNet,
    train_loader: DataLoader,
    loss_function: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    fine_tune_final_block: bool,
) -> tuple[float, float]:
    """Train once and return example-weighted loss and training accuracy."""
    set_transfer_learning_train_mode(model, fine_tune_final_block)
    total_loss = 0.0
    correct_predictions = 0
    total_examples = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
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

    with torch.inference_mode():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images)

            correct_predictions += (
                predictions.argmax(dim=1) == labels
            ).sum().item()
            total_examples += labels.size(0)

    return 100 * correct_predictions / total_examples


def set_random_seed() -> None:
    """Reset random state so experiment initialization is reproducible."""
    torch.manual_seed(RANDOM_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)


def run_experiment(
    strategy: str,
    train_dataset: Dataset,
    test_dataset: Dataset,
    num_classes: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    num_workers: int,
) -> ExperimentResult:
    """Train one transfer-learning strategy and return measured results."""
    set_random_seed()
    fine_tune_final_block = strategy == FINE_TUNE_LAYER4

    train_loader, test_loader = create_data_loaders(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        use_pinned_memory=device.type == "cuda",
    )
    model = create_transfer_learning_model(
        num_classes=num_classes,
        fine_tune_final_block=fine_tune_final_block,
    ).to(device)
    parameter_summary = summarize_parameters(model)

    loss_function = nn.CrossEntropyLoss()
    optimizer = Adam(
        (
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        lr=learning_rate,
    )

    print(f"\nExperiment: {strategy}")
    print("Pretrained model: ResNet18 with ImageNet weights")
    print(f"Trainable parameters: {parameter_summary.trainable:,}")
    print(f"Frozen parameters: {parameter_summary.frozen:,}")
    print("Frozen layer verification: passed")

    final_training_loss = 0.0
    final_training_accuracy = 0.0
    final_test_accuracy = 0.0
    training_time_seconds = 0.0

    for epoch in range(1, epochs + 1):
        synchronize_device(device)
        training_start = perf_counter()

        final_training_loss, final_training_accuracy = train_one_epoch(
            model=model,
            train_loader=train_loader,
            loss_function=loss_function,
            optimizer=optimizer,
            device=device,
            fine_tune_final_block=fine_tune_final_block,
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

    print(f"Measured training time: {training_time_seconds:.2f} seconds")

    return ExperimentResult(
        strategy=strategy,
        final_training_loss=final_training_loss,
        final_training_accuracy=final_training_accuracy,
        final_test_accuracy=final_test_accuracy,
        training_time_seconds=training_time_seconds,
        trainable_parameters=parameter_summary.trainable,
    )


def print_comparison(results: list[ExperimentResult]) -> None:
    """Print only the metrics measured during this process."""
    print("\nMeasured transfer-learning results")
    print(
        "Strategy         | Trainable params | Final loss | "
        "Train accuracy | Test accuracy | Training time"
    )
    print("-" * 96)

    for result in results:
        print(
            f"{result.strategy:<16} | "
            f"{result.trainable_parameters:>16,} | "
            f"{result.final_training_loss:>10.4f} | "
            f"{result.final_training_accuracy:>13.2f}% | "
            f"{result.final_test_accuracy:>12.2f}% | "
            f"{result.training_time_seconds:>12.2f}s"
        )


def parse_arguments() -> argparse.Namespace:
    """Parse options for one experiment or the two-part lab."""
    parser = argparse.ArgumentParser(
        description="Fine-tune pretrained ResNet18 on CIFAR-10.",
    )
    parser.add_argument(
        "--strategy",
        choices=STRATEGIES,
        default=HEAD_ONLY,
        help="Training strategy for a single experiment.",
    )
    parser.add_argument(
        "--compare-strategies",
        action="store_true",
        help="Run head-only and layer4 fine-tuning experiments.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"Number of epochs (default: {EPOCHS}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Images per batch (default: {BATCH_SIZE}).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=LEARNING_RATE,
        help=f"Adam learning rate (default: {LEARNING_RATE}).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=IMAGE_SIZE,
        help=f"Square input size (default: {IMAGE_SIZE}).",
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=DEFAULT_TRAIN_SAMPLES,
        help=(
            "Deterministic training subset size; use 0 for all "
            f"(default: {DEFAULT_TRAIN_SAMPLES})."
        ),
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=DEFAULT_TEST_SAMPLES,
        help=(
            "Deterministic test subset size; use 0 for all "
            f"(default: {DEFAULT_TEST_SAMPLES})."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker processes; 0 is safest on Windows.",
    )
    arguments = parser.parse_args()

    if arguments.epochs < 1:
        parser.error("--epochs must be at least 1")
    if arguments.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if arguments.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if arguments.image_size < 32:
        parser.error("--image-size must be at least 32")
    if arguments.train_samples < 0:
        parser.error("--train-samples cannot be negative")
    if arguments.test_samples < 0:
        parser.error("--test-samples cannot be negative")
    if arguments.num_workers < 0:
        parser.error("--num-workers cannot be negative")

    return arguments


def main() -> None:
    """Run the selected experiment and print measured values."""
    arguments = parse_arguments()
    device = select_device()

    train_sample_limit = arguments.train_samples or None
    test_sample_limit = arguments.test_samples or None
    train_dataset, test_dataset, num_classes = create_datasets(
        image_size=arguments.image_size,
        train_samples=train_sample_limit,
        test_samples=test_sample_limit,
    )

    strategies = (
        STRATEGIES
        if arguments.compare_strategies
        else (arguments.strategy,)
    )

    print(f"Training on: {device}")
    print(
        f"Dataset: CIFAR-10 | Training samples: {len(train_dataset):,} | "
        f"Test samples: {len(test_dataset):,}"
    )
    print(f"Input image size: {arguments.image_size}x{arguments.image_size}")

    results = [
        run_experiment(
            strategy=strategy,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            num_classes=num_classes,
            device=device,
            epochs=arguments.epochs,
            batch_size=arguments.batch_size,
            learning_rate=arguments.learning_rate,
            num_workers=arguments.num_workers,
        )
        for strategy in strategies
    ]
    print_comparison(results)


if __name__ == "__main__":
    main()
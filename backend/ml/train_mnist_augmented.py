"""Train the MNIST classifier with controlled data augmentation."""

from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

try:
    # Supports: python -m backend.ml.train_mnist_augmented
    from .models import MNISTClassifier
except ImportError:
    # Supports: python backend/ml/train_mnist_augmented.py
    from models import MNISTClassifier


BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 5
RANDOM_SEED = 42
DATA_DIRECTORY = Path(__file__).resolve().parent / "data"

TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.RandomRotation(10),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
        ),
        transforms.ToTensor(),
    ]
)

# Evaluation data must remain unchanged so every epoch uses the same fair test.
EVALUATION_TRANSFORM = transforms.ToTensor()


def select_device() -> torch.device:
    """Use the best available PyTorch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def create_data_loaders() -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create augmented training and unaugmented evaluation data loaders."""
    augmented_train_dataset = datasets.MNIST(
        root=DATA_DIRECTORY,
        train=True,
        download=True,
        transform=TRAIN_TRANSFORM,
    )

    clean_train_dataset = datasets.MNIST(
        root=DATA_DIRECTORY,
        train=True,
        download=True,
        transform=EVALUATION_TRANSFORM,
    )

    test_dataset = datasets.MNIST(
        root=DATA_DIRECTORY,
        train=False,
        download=True,
        transform=EVALUATION_TRANSFORM,
    )

    shuffle_generator = torch.Generator().manual_seed(RANDOM_SEED)

    train_loader = DataLoader(
        augmented_train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=shuffle_generator,
    )
    train_evaluation_loader = DataLoader(
        clean_train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    return train_loader, train_evaluation_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    loss_function: nn.Module,
    optimizer: Adam,
    device: torch.device,
) -> float:
    """Train for one epoch and return the loss averaged across all examples."""
    model.train()
    total_loss = 0.0
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
        total_examples += batch_size

    return total_loss / total_examples


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


def main() -> None:
    """Train for five epochs and report training and test metrics."""
    torch.manual_seed(RANDOM_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    device = select_device()
    train_loader, train_evaluation_loader, test_loader = create_data_loaders()

    model = MNISTClassifier().to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Training on: {device}")
    print("Training transform: rotation ±10°, translation up to 10%")

    for epoch in range(1, EPOCHS + 1):
        average_training_loss = train_one_epoch(
            model,
            train_loader,
            loss_function,
            optimizer,
            device,
        )
        training_accuracy = calculate_accuracy(
            model,
            train_evaluation_loader,
            device,
        )
        test_accuracy = calculate_accuracy(model, test_loader, device)

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Average training loss: {average_training_loss:.4f} | "
            f"Training accuracy: {training_accuracy:.2f}% | "
            f"Test accuracy: {test_accuracy:.2f}%"
        )


if __name__ == "__main__":
    main()
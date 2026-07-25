"""Train a simple neural network on the MNIST dataset."""

from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

try:
    # Supports: python -m backend.ml.train_mnist
    from .models import MNISTClassifier
except ImportError:
    # Supports: python backend/ml/train_mnist.py
    from models import MNISTClassifier


BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 5
DATA_DIRECTORY = Path(__file__).resolve().parent / "data"


def select_device() -> torch.device:
    """Use the best available PyTorch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def create_train_loader() -> DataLoader:
    """Download MNIST when needed and return its training data loader."""
    train_dataset = datasets.MNIST(
        root=DATA_DIRECTORY,
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )

    return DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    loss_function: nn.Module,
    optimizer: Adam,
    device: torch.device,
) -> float:
    """Train the model for one epoch and return its average loss."""
    model.train()
    total_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        predictions = model(images)
        loss = loss_function(predictions, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def calculate_accuracy(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
) -> float:
    """Calculate classification accuracy across the training dataset."""
    model.eval()
    correct_predictions = 0
    total_examples = 0

    with torch.no_grad():
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            predictions = model(images)
            predicted_classes = predictions.argmax(dim=1)

            correct_predictions += (predicted_classes == labels).sum().item()
            total_examples += labels.size(0)

    return 100 * correct_predictions / total_examples


def main() -> None:
    """Load MNIST, train for five epochs, and print training accuracy."""
    torch.manual_seed(42)

    device = select_device()
    train_loader = create_train_loader()
    model = MNISTClassifier().to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Training on: {device}")

    for epoch in range(1, EPOCHS + 1):
        average_loss = train_one_epoch(
            model,
            train_loader,
            loss_function,
            optimizer,
            device,
        )
        print(f"Epoch {epoch}/{EPOCHS} - Loss: {average_loss:.4f}")

    training_accuracy = calculate_accuracy(model, train_loader, device)
    print(f"Final training accuracy: {training_accuracy:.2f}%")


if __name__ == "__main__":
    main()
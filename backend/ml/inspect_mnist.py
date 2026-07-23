"""Load and inspect one batch from the MNIST training dataset."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


BATCH_SIZE = 32
DATA_DIRECTORY = Path(__file__).parent / "data"


def create_data_loader() -> DataLoader:
    """Download MNIST and return a shuffled training DataLoader."""

    # Convert each MNIST image into a PyTorch tensor.
    transform = transforms.ToTensor()

    training_data = datasets.MNIST(
        root=DATA_DIRECTORY,
        train=True,
        download=True,
        transform=transform,
    )

    # A fixed generator seed makes the shuffled order reproducible.
    generator = torch.Generator().manual_seed(42)

    return DataLoader(
        training_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Recommended for a simple Windows learning environment.
        generator=generator,
    )


def main() -> None:
    """Retrieve one batch and display its tensor information."""

    loader = create_data_loader()

    images, labels = next(iter(loader))

    print(f"Number of training examples: {len(loader.dataset)}")
    print(f"Number of batches: {len(loader)}")
    print(f"Image batch shape: {images.shape}")
    print(f"Label batch shape: {labels.shape}")
    print(f"First 10 labels: {labels[:10]}")

    # Basic validation so errors are visible immediately.
    assert images.shape == (BATCH_SIZE, 1, 28, 28)
    assert labels.shape == (BATCH_SIZE,)

    print("MNIST batch validation passed.")


if __name__ == "__main__":
    main()
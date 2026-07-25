"""Neural network models used by the ML training scripts."""

from torch import nn


class MNISTClassifier(nn.Module):
    """A simple fully connected neural network for MNIST classification."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, images):
        """Return class logits for a batch of MNIST images."""
        return self.network(images)
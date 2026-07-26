"""Convolutional neural-network models for MNIST classification."""

from torch import Tensor, nn


class MNISTCNN(nn.Module):
    """A two-layer convolutional neural network for 28x28 MNIST images."""

    def __init__(self, kernel_size: int = 3) -> None:
        super().__init__()

        if kernel_size not in (3, 5):
            raise ValueError("kernel_size must be either 3 or 5")

        self.kernel_size = kernel_size
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        feature_map_size = self._calculate_feature_map_size(
            image_size=28,
            kernel_size=kernel_size,
        )
        flattened_features = 64 * feature_map_size * feature_map_size

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_features, 10),
        )

    @staticmethod
    def _calculate_feature_map_size(
        image_size: int,
        kernel_size: int,
    ) -> int:
        """Return the width or height after two convolution/pooling blocks."""
        feature_map_size = image_size

        for _ in range(2):
            feature_map_size = feature_map_size - kernel_size + 1
            feature_map_size = feature_map_size // 2

        return feature_map_size

    def forward(self, images: Tensor) -> Tensor:
        """Return class logits for a batch of MNIST images."""
        features = self.features(images)
        return self.classifier(features)
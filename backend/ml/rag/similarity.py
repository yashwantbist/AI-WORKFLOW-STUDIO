"""Vector similarity utilities used by semantic search."""

from collections.abc import Sequence
import math


def dot_product(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the dot product of two vectors with equal dimensions."""
    if len(left) != len(right):
        raise ValueError("Vectors must have the same number of dimensions")

    return sum(left_value * right_value for left_value, right_value in zip(left, right))


def vector_magnitude(vector: Sequence[float]) -> float:
    """Return the Euclidean length of a vector."""
    return math.sqrt(sum(value * value for value in vector))


def cosine_similarity(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    """Measure how closely two vectors point in the same direction.

    A zero vector has no direction, so this educational implementation returns
    0.0 whenever either input has magnitude zero.
    """
    if len(left) != len(right):
        raise ValueError("Vectors must have the same number of dimensions")

    left_magnitude = vector_magnitude(left)
    right_magnitude = vector_magnitude(right)

    if left_magnitude == 0.0 or right_magnitude == 0.0:
        return 0.0

    similarity = dot_product(left, right) / (left_magnitude * right_magnitude)

    # Floating-point rounding can produce values such as 1.0000000000000002.
    return max(-1.0, min(1.0, similarity))

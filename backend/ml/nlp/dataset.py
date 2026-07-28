"""Tokenization, vocabulary, and dataset utilities for sentiment analysis."""

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import product
import random
import re

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


PAD_TOKEN = "<pad>"
UNKNOWN_TOKEN = "<unk>"
CLASS_NAMES = ("negative", "positive")
TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")

SUBJECTS = (
    "application",
    "book",
    "course",
    "experience",
    "meal",
    "movie",
    "product",
    "service",
)
SENTENCE_TEMPLATES = (
    "i {verb} this {adjective} {subject}",
    "this {subject} is {adjective} and i {verb} it",
    "what a {adjective} {subject} i {verb} it",
    "the {adjective} {subject} makes me {verb} it",
)
POSITIVE_ADJECTIVES = (
    "amazing",
    "excellent",
    "fantastic",
    "helpful",
    "pleasant",
    "reliable",
    "wonderful",
    "valuable",
)
NEGATIVE_ADJECTIVES = (
    "awful",
    "confusing",
    "disappointing",
    "horrible",
    "poor",
    "unpleasant",
    "unreliable",
    "worthless",
)
POSITIVE_VERBS = ("appreciate", "enjoy", "love", "recommend")
NEGATIVE_VERBS = ("avoid", "dislike", "hate", "regret")


@dataclass(frozen=True)
class TextExample:
    """One text sample and its integer class label."""

    text: str
    label: int


@dataclass(frozen=True)
class DataBundle:
    """Datasets and vocabulary shared by all embedding experiments."""

    training_dataset: "TextClassificationDataset"
    test_dataset: "TextClassificationDataset"
    vocabulary: "Vocabulary"


def tokenize(text: str) -> list[str]:
    """Lowercase text and split it into simple word tokens."""
    return TOKEN_PATTERN.findall(text.lower())


class Vocabulary:
    """Map tokens to deterministic integer IDs."""

    def __init__(self, token_to_id: dict[str, int]) -> None:
        if token_to_id.get(PAD_TOKEN) != 0:
            raise ValueError(f"{PAD_TOKEN} must have token ID 0")
        if token_to_id.get(UNKNOWN_TOKEN) != 1:
            raise ValueError(f"{UNKNOWN_TOKEN} must have token ID 1")

        expected_ids = set(range(len(token_to_id)))
        if set(token_to_id.values()) != expected_ids:
            raise ValueError("Vocabulary token IDs must be contiguous")

        self._token_to_id = dict(token_to_id)

    @classmethod
    def build(
        cls,
        texts: Iterable[str],
        minimum_frequency: int = 1,
    ) -> "Vocabulary":
        """Create a vocabulary from training text only."""
        if minimum_frequency < 1:
            raise ValueError("minimum_frequency must be at least 1")

        token_counts = Counter(
            token
            for text in texts
            for token in tokenize(text)
        )
        ordered_tokens = sorted(
            (
                token
                for token, count in token_counts.items()
                if count >= minimum_frequency
            ),
            key=lambda token: (-token_counts[token], token),
        )

        token_to_id = {
            PAD_TOKEN: 0,
            UNKNOWN_TOKEN: 1,
        }
        token_to_id.update(
            {
                token: index
                for index, token in enumerate(ordered_tokens, start=2)
            }
        )
        return cls(token_to_id)

    @property
    def padding_id(self) -> int:
        """Return the ID used to pad shorter sequences."""
        return self._token_to_id[PAD_TOKEN]

    @property
    def unknown_id(self) -> int:
        """Return the ID used for tokens not seen during training."""
        return self._token_to_id[UNKNOWN_TOKEN]

    def encode(self, text: str) -> list[int]:
        """Tokenize text and convert every token to an integer ID."""
        tokens = tokenize(text)
        if not tokens:
            return [self.unknown_id]

        return [
            self._token_to_id.get(token, self.unknown_id)
            for token in tokens
        ]

    def to_dict(self) -> dict[str, int]:
        """Return a copy suitable for saving in a model checkpoint."""
        return dict(self._token_to_id)

    def __len__(self) -> int:
        return len(self._token_to_id)


class TextClassificationDataset(Dataset):
    """Hold numericalized text examples for PyTorch."""

    def __init__(
        self,
        examples: Sequence[TextExample],
        vocabulary: Vocabulary,
    ) -> None:
        self._items = [
            (
                torch.tensor(
                    vocabulary.encode(example.text),
                    dtype=torch.long,
                ),
                example.label,
            )
            for example in examples
        ]

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        return self._items[index]

    def __len__(self) -> int:
        return len(self._items)


def generate_sentiment_examples(
    samples_per_class: int = 120,
    random_seed: int = 42,
) -> list[TextExample]:
    """Generate a balanced sentiment dataset in memory.

    The dataset is created at runtime from phrase components, so no downloaded
    dataset or generated data file needs to be committed to Git.
    """
    if samples_per_class < 1:
        raise ValueError("samples_per_class must be at least 1")

    class_components = (
        (0, NEGATIVE_ADJECTIVES, NEGATIVE_VERBS),
        (1, POSITIVE_ADJECTIVES, POSITIVE_VERBS),
    )
    random_generator = random.Random(random_seed)
    examples: list[TextExample] = []

    for label, adjectives, verbs in class_components:
        candidates = [
            TextExample(
                text=template.format(
                    adjective=adjective,
                    subject=subject,
                    verb=verb,
                ),
                label=label,
            )
            for template, subject, adjective, verb in product(
                SENTENCE_TEMPLATES,
                SUBJECTS,
                adjectives,
                verbs,
            )
        ]

        if samples_per_class > len(candidates):
            raise ValueError(
                "samples_per_class exceeds the unique generated examples"
            )

        random_generator.shuffle(candidates)
        examples.extend(candidates[:samples_per_class])

    random_generator.shuffle(examples)
    return examples


def split_examples(
    examples: Sequence[TextExample],
    test_fraction: float = 0.2,
    random_seed: int = 42,
) -> tuple[list[TextExample], list[TextExample]]:
    """Create a deterministic, class-balanced training/test split."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")

    random_generator = random.Random(random_seed)
    training_examples: list[TextExample] = []
    test_examples: list[TextExample] = []

    for label in range(len(CLASS_NAMES)):
        label_examples = [
            example for example in examples
            if example.label == label
        ]
        random_generator.shuffle(label_examples)

        test_count = max(1, round(len(label_examples) * test_fraction))
        test_examples.extend(label_examples[:test_count])
        training_examples.extend(label_examples[test_count:])

    random_generator.shuffle(training_examples)
    random_generator.shuffle(test_examples)
    return training_examples, test_examples


def build_data_bundle(
    samples_per_class: int = 120,
    test_fraction: float = 0.2,
    random_seed: int = 42,
) -> DataBundle:
    """Generate examples, split them, and build vocabulary from training text."""
    examples = generate_sentiment_examples(
        samples_per_class=samples_per_class,
        random_seed=random_seed,
    )
    training_examples, test_examples = split_examples(
        examples,
        test_fraction=test_fraction,
        random_seed=random_seed,
    )
    vocabulary = Vocabulary.build(
        example.text for example in training_examples
    )

    return DataBundle(
        training_dataset=TextClassificationDataset(
            training_examples,
            vocabulary,
        ),
        test_dataset=TextClassificationDataset(
            test_examples,
            vocabulary,
        ),
        vocabulary=vocabulary,
    )


def collate_text_batch(
    batch: Sequence[tuple[Tensor, int]],
) -> tuple[Tensor, Tensor, Tensor]:
    """Pad variable-length sequences and create their attention mask."""
    token_sequences, labels = zip(*batch)
    padded_token_ids = pad_sequence(
        token_sequences,
        batch_first=True,
        padding_value=0,
    )
    attention_mask = padded_token_ids.ne(0)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    return padded_token_ids, attention_mask, label_tensor


def create_data_loaders(
    data_bundle: DataBundle,
    batch_size: int,
    random_seed: int,
) -> tuple[DataLoader, DataLoader]:
    """Create repeatable loaders shared by both embedding experiments."""
    shuffle_generator = torch.Generator().manual_seed(random_seed)

    training_loader = DataLoader(
        data_bundle.training_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=shuffle_generator,
        collate_fn=collate_text_batch,
    )
    test_loader = DataLoader(
        data_bundle.test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_text_batch,
    )
    return training_loader, test_loader
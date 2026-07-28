"""Tests for the Day 8 NLP text-classification pipeline."""

import torch

from backend.ml.nlp.dataset import (
    PAD_TOKEN,
    UNKNOWN_TOKEN,
    Vocabulary,
    build_data_bundle,
    collate_text_batch,
    generate_sentiment_examples,
    split_examples,
    tokenize,
)
from backend.ml.nlp.model import MeanPoolingTextClassifier


def test_tokenize_lowercases_and_removes_punctuation() -> None:
    assert tokenize("I LOVE this course!") == [
        "i",
        "love",
        "this",
        "course",
    ]


def test_vocabulary_reserves_padding_and_unknown_ids() -> None:
    vocabulary = Vocabulary.build(
        ["I love AI", "AI is useful"],
    )

    assert vocabulary.to_dict()[PAD_TOKEN] == 0
    assert vocabulary.to_dict()[UNKNOWN_TOKEN] == 1
    assert vocabulary.encode("missing") == [vocabulary.unknown_id]


def test_generated_dataset_is_balanced_and_split_is_repeatable() -> None:
    examples = generate_sentiment_examples(
        samples_per_class=20,
        random_seed=42,
    )
    first_split = split_examples(examples, random_seed=42)
    second_split = split_examples(examples, random_seed=42)

    assert sum(example.label == 0 for example in examples) == 20
    assert sum(example.label == 1 for example in examples) == 20
    assert first_split == second_split


def test_data_bundle_builds_vocabulary_from_training_examples() -> None:
    data_bundle = build_data_bundle(
        samples_per_class=20,
        test_fraction=0.2,
        random_seed=42,
    )

    assert len(data_bundle.training_dataset) == 32
    assert len(data_bundle.test_dataset) == 8
    assert len(data_bundle.vocabulary) > 2


def test_collate_batch_pads_sequences_and_creates_mask() -> None:
    token_ids, attention_mask, labels = collate_text_batch(
        [
            (torch.tensor([2, 3, 4]), 1),
            (torch.tensor([5]), 0),
        ]
    )

    assert token_ids.tolist() == [[2, 3, 4], [5, 0, 0]]
    assert attention_mask.tolist() == [
        [True, True, True],
        [True, False, False],
    ]
    assert labels.tolist() == [1, 0]


def test_classifier_returns_one_logit_per_class() -> None:
    model = MeanPoolingTextClassifier(
        vocabulary_size=10,
        embedding_size=32,
        number_of_classes=2,
        padding_id=0,
    )
    token_ids = torch.tensor(
        [
            [2, 3, 4],
            [5, 6, 0],
        ]
    )
    attention_mask = token_ids.ne(0)

    logits = model(token_ids, attention_mask)

    assert logits.shape == (2, 2)


def test_mean_pooling_ignores_padding() -> None:
    model = MeanPoolingTextClassifier(
        vocabulary_size=10,
        embedding_size=32,
        number_of_classes=2,
        padding_id=0,
    )
    short_tokens = torch.tensor([[2, 3]])
    padded_tokens = torch.tensor([[2, 3, 0, 0]])

    short_logits = model(short_tokens, short_tokens.ne(0))
    padded_logits = model(padded_tokens, padded_tokens.ne(0))

    assert torch.allclose(short_logits, padded_logits)
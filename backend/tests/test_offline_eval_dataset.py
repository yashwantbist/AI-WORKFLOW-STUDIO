import json

import pytest

from backend.ml.rag.evaluation.offline_dataset import (
    OfflineEvaluationDataset,
)


def write_dataset(tmp_path, payload):
    path = tmp_path / "eval.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_loads_valid_dataset(tmp_path):
    path = write_dataset(
        tmp_path,
        {
            "name": "golden",
            "description": "test",
            "cases": [
                {
                    "id": "q1",
                    "query": "Question?",
                    "relevant_ids": ["c1", "c2"],
                    "tags": ["direct"],
                }
            ],
        },
    )

    dataset = OfflineEvaluationDataset.load(path)

    assert dataset.name == "golden"
    assert len(dataset.cases) == 1
    assert dataset.cases[0].relevant_ids == ("c1", "c2")


def test_allows_explicit_empty_relevant_set(tmp_path):
    path = write_dataset(
        tmp_path,
        {
            "name": "golden",
            "cases": [
                {
                    "id": "q1",
                    "query": "No answer exists?",
                    "relevant_ids": [],
                }
            ],
        },
    )

    dataset = OfflineEvaluationDataset.load(path)

    assert dataset.cases[0].relevant_ids == ()


def test_rejects_missing_relevant_ids(tmp_path):
    path = write_dataset(
        tmp_path,
        {
            "name": "bad",
            "cases": [
                {
                    "id": "q1",
                    "query": "Question?",
                }
            ],
        },
    )

    with pytest.raises(ValueError, match="relevant_ids"):
        OfflineEvaluationDataset.load(path)


def test_rejects_non_list_relevant_ids(tmp_path):
    path = write_dataset(
        tmp_path,
        {
            "name": "bad",
            "cases": [
                {
                    "id": "q1",
                    "query": "Question?",
                    "relevant_ids": "c1",
                }
            ],
        },
    )

    with pytest.raises(ValueError, match="must be a list"):
        OfflineEvaluationDataset.load(path)


def test_rejects_duplicate_case_ids(tmp_path):
    path = write_dataset(
        tmp_path,
        {
            "name": "bad",
            "cases": [
                {
                    "id": "q1",
                    "query": "First?",
                    "relevant_ids": ["c1"],
                },
                {
                    "id": "q1",
                    "query": "Second?",
                    "relevant_ids": ["c2"],
                },
            ],
        },
    )

    with pytest.raises(ValueError, match="IDs must be unique"):
        OfflineEvaluationDataset.load(path)


def test_rejects_duplicate_relevance_labels(tmp_path):
    path = write_dataset(
        tmp_path,
        {
            "name": "bad",
            "cases": [
                {
                    "id": "q1",
                    "query": "Question?",
                    "relevant_ids": ["c1", "c1"],
                }
            ],
        },
    )

    with pytest.raises(ValueError, match="relevant_ids must be unique"):
        OfflineEvaluationDataset.load(path)

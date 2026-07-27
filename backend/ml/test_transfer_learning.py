"""Tests for ResNet18 transfer-learning layer configuration."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from torch import nn

from backend.ml.transfer_learning import (
    create_transfer_learning_model,
    set_transfer_learning_train_mode,
    summarize_parameters,
)


def test_head_only_experiment_freezes_the_backbone() -> None:
    """Only the replacement classifier should be trainable."""
    model = create_transfer_learning_model(
        num_classes=10,
        fine_tune_final_block=False,
        pretrained=False,
    )

    trainable_names = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    assert model.fc.out_features == 10
    assert trainable_names == {"fc.weight", "fc.bias"}

    summary = summarize_parameters(model)
    assert summary.trainable == model.fc.weight.numel() + model.fc.bias.numel()
    assert summary.frozen > summary.trainable


def test_second_experiment_unfreezes_layer4_and_classifier() -> None:
    """Layer4 and the classifier should be trainable in experiment two."""
    model = create_transfer_learning_model(
        num_classes=10,
        fine_tune_final_block=True,
        pretrained=False,
    )

    trainable_names = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]

    assert any(name.startswith("layer4.") for name in trainable_names)
    assert "fc.weight" in trainable_names
    assert "fc.bias" in trainable_names
    assert all(
        name.startswith(("layer4.", "fc."))
        for name in trainable_names
    )


def test_frozen_batch_norm_statistics_stay_in_evaluation_mode() -> None:
    """Frozen BatchNorm modules should not update running statistics."""
    model = create_transfer_learning_model(
        num_classes=10,
        fine_tune_final_block=True,
        pretrained=False,
    )

    set_transfer_learning_train_mode(
        model,
        fine_tune_final_block=True,
    )

    assert model.bn1.training is False

    final_block_batch_norm_layers = [
        module
        for module in model.layer4.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    ]
    assert final_block_batch_norm_layers
    assert all(module.training for module in final_block_batch_norm_layers)
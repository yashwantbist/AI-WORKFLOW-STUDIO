"""ResNet18 model setup helpers for transfer-learning experiments."""

from dataclasses import dataclass

from torch import nn
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models.resnet import ResNet


@dataclass(frozen=True)
class ParameterSummary:
    """Counts used to verify which model parameters will be optimized."""

    trainable: int
    frozen: int

    @property
    def total(self) -> int:
        """Return the total number of model parameters."""
        return self.trainable + self.frozen


def create_transfer_learning_model(
    num_classes: int,
    fine_tune_final_block: bool = False,
    pretrained: bool = True,
) -> ResNet:
    """Create ResNet18 with a new classifier for the target dataset.

    Every pretrained parameter is frozen first. The newly created ``fc`` layer
    remains trainable. When ``fine_tune_final_block`` is true, ``layer4`` is
    also unfrozen for the second lab experiment.

    Set ``pretrained=False`` only for offline tests; real training should keep
    the default so ImageNet weights are loaded.
    """
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")

    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)

    for parameter in model.parameters():
        parameter.requires_grad = False

    input_features = model.fc.in_features
    model.fc = nn.Linear(input_features, num_classes)

    if fine_tune_final_block:
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True

    verify_trainable_layers(model, fine_tune_final_block)
    return model


def verify_trainable_layers(
    model: ResNet,
    fine_tune_final_block: bool,
) -> None:
    """Raise an error if parameters outside the intended layers are trainable."""
    allowed_prefixes = ("fc.",)

    if fine_tune_final_block:
        allowed_prefixes = ("layer4.", "fc.")

    trainable_names = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]

    if not trainable_names:
        raise RuntimeError("The model has no trainable parameters")

    unexpected_names = [
        name for name in trainable_names
        if not name.startswith(allowed_prefixes)
    ]

    if unexpected_names:
        unexpected = ", ".join(unexpected_names)
        raise RuntimeError(f"Unexpected trainable parameters: {unexpected}")

    required_prefixes = ("fc.",)
    if fine_tune_final_block:
        required_prefixes = ("layer4.", "fc.")

    for prefix in required_prefixes:
        if not any(name.startswith(prefix) for name in trainable_names):
            raise RuntimeError(f"No trainable parameters found under {prefix}")


def set_transfer_learning_train_mode(
    model: ResNet,
    fine_tune_final_block: bool,
) -> None:
    """Set training mode without updating frozen BatchNorm statistics.

    ``requires_grad=False`` freezes weights and biases, but calling
    ``model.train()`` would still update BatchNorm running statistics. Frozen
    BatchNorm modules are therefore returned to evaluation mode.
    """
    model.train()

    for module_name, module in model.named_modules():
        if not isinstance(module, nn.modules.batchnorm._BatchNorm):
            continue

        belongs_to_trainable_block = (
            fine_tune_final_block
            and (
                module_name == "layer4"
                or module_name.startswith("layer4.")
            )
        )

        if not belongs_to_trainable_block:
            module.eval()


def summarize_parameters(model: nn.Module) -> ParameterSummary:
    """Return trainable and frozen parameter counts."""
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    frozen = sum(
        parameter.numel()
        for parameter in model.parameters()
        if not parameter.requires_grad
    )
    return ParameterSummary(trainable=trainable, frozen=frozen)
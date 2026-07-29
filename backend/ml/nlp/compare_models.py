"""Fairly compare the Day 8 pooling model with the Transformer encoder."""

import argparse

try:
    # Supports: python -m backend.ml.nlp.compare_models
    from .dataset import CLASS_NAMES, build_data_bundle
    from .model import MeanPoolingTextClassifier
    from .train_transformer_classifier import (
        BATCH_SIZE,
        DROPOUT,
        EPOCHS,
        FEED_FORWARD_DIMENSION,
        LEARNING_RATE,
        MODEL_DIMENSION,
        NUMBER_OF_HEADS,
        NUMBER_OF_LAYERS,
        RANDOM_SEED,
        SAMPLES_PER_CLASS,
        TrainingResult,
        build_transformer_factory,
        run_training_experiment,
        select_device,
    )
except ImportError:
    # Supports: python backend/ml/nlp/compare_models.py
    from dataset import CLASS_NAMES, build_data_bundle
    from model import MeanPoolingTextClassifier
    from train_transformer_classifier import (
        BATCH_SIZE,
        DROPOUT,
        EPOCHS,
        FEED_FORWARD_DIMENSION,
        LEARNING_RATE,
        MODEL_DIMENSION,
        NUMBER_OF_HEADS,
        NUMBER_OF_LAYERS,
        RANDOM_SEED,
        SAMPLES_PER_CLASS,
        TrainingResult,
        build_transformer_factory,
        run_training_experiment,
        select_device,
    )


def print_comparison(results: tuple[TrainingResult, ...]) -> None:
    """Print measured model metrics in a readable table."""
    print("\nMeasured architecture comparison")
    print(
        "Model             | Parameters | Final loss | Train acc. | "
        "Validation acc. | Train time"
    )
    print("-" * 99)

    for result in results:
        print(
            f"{result.model_name:<17} | "
            f"{result.parameter_count:>10,} | "
            f"{result.final_training_loss:>10.4f} | "
            f"{result.final_training_accuracy:>9.2f}% | "
            f"{result.final_validation_accuracy:>14.2f}% | "
            f"{result.training_time_seconds:>9.4f}s"
        )


def parse_arguments() -> argparse.Namespace:
    """Read shared experiment settings."""
    parser = argparse.ArgumentParser(
        description=(
            "Train the pooling baseline and Transformer under identical "
            "conditions."
        ),
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    arguments = parser.parse_args()

    if arguments.epochs < 1:
        parser.error("--epochs must be at least 1")
    if arguments.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if arguments.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    return arguments


def main() -> None:
    """Run the controlled baseline-versus-Transformer experiment."""
    arguments = parse_arguments()
    device = select_device()
    data_bundle = build_data_bundle(
        samples_per_class=SAMPLES_PER_CLASS,
        random_seed=RANDOM_SEED,
    )

    print(f"Training on: {device}")
    print(
        f"Shared data: {len(data_bundle.training_dataset)} training, "
        f"{len(data_bundle.test_dataset)} validation"
    )
    print(
        f"Controlled settings: seed={RANDOM_SEED}, "
        f"epochs={arguments.epochs}, batch_size={arguments.batch_size}, "
        f"Adam learning_rate={arguments.learning_rate}"
    )
    print(
        f"Shared representation size: {MODEL_DIMENSION} | "
        f"Transformer heads: {NUMBER_OF_HEADS} | "
        f"Transformer layers: {NUMBER_OF_LAYERS}"
    )

    baseline_factory = lambda: MeanPoolingTextClassifier(
        vocabulary_size=len(data_bundle.vocabulary),
        embedding_size=MODEL_DIMENSION,
        number_of_classes=len(CLASS_NAMES),
        padding_id=data_bundle.vocabulary.padding_id,
    )
    transformer_factory = build_transformer_factory(
        vocabulary_size=len(data_bundle.vocabulary),
        number_of_classes=len(CLASS_NAMES),
        padding_id=data_bundle.vocabulary.padding_id,
        model_dimension=MODEL_DIMENSION,
        number_of_heads=NUMBER_OF_HEADS,
        number_of_layers=NUMBER_OF_LAYERS,
        feed_forward_dimension=FEED_FORWARD_DIMENSION,
        dropout=DROPOUT,
    )

    baseline_output = run_training_experiment(
        model_name="Embedding + mean",
        model_factory=baseline_factory,
        data_bundle=data_bundle,
        device=device,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
    )
    transformer_output = run_training_experiment(
        model_name="Transformer",
        model_factory=transformer_factory,
        data_bundle=data_bundle,
        device=device,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
    )

    print_comparison(
        (
            baseline_output.result,
            transformer_output.result,
        )
    )


if __name__ == "__main__":
    main()
"""Run sentiment predictions with a trained text-classifier checkpoint."""

import argparse
from pathlib import Path

import torch

try:
    # Supports: python -m backend.ml.nlp.inference
    from .dataset import Vocabulary, tokenize
    from .model import MeanPoolingTextClassifier
    from .train_text_classifier import DEFAULT_CHECKPOINT
except ImportError:
    # Supports: python backend/ml/nlp/inference.py
    from dataset import Vocabulary, tokenize
    from model import MeanPoolingTextClassifier
    from train_text_classifier import DEFAULT_CHECKPOINT


def load_classifier(
    checkpoint_path: Path,
) -> tuple[MeanPoolingTextClassifier, Vocabulary, list[str]]:
    """Rebuild the model and vocabulary stored in a checkpoint."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Train the classifier first."
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    vocabulary = Vocabulary(checkpoint["vocabulary"])
    class_names = list(checkpoint["class_names"])
    model = MeanPoolingTextClassifier(
        vocabulary_size=len(vocabulary),
        embedding_size=int(checkpoint["embedding_size"]),
        number_of_classes=len(class_names),
        padding_id=vocabulary.padding_id,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, vocabulary, class_names


def predict_sentiment(
    text: str,
    model: MeanPoolingTextClassifier,
    vocabulary: Vocabulary,
    class_names: list[str],
) -> tuple[str, float, list[float]]:
    """Return predicted label, confidence, and all class probabilities."""
    token_ids = torch.tensor(
        [vocabulary.encode(text)],
        dtype=torch.long,
    )
    attention_mask = token_ids.ne(vocabulary.padding_id)

    with torch.inference_mode():
        logits = model(token_ids, attention_mask)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)

    predicted_index = int(probabilities.argmax().item())
    return (
        class_names[predicted_index],
        float(probabilities[predicted_index].item()),
        [float(probability) for probability in probabilities],
    )


def parse_arguments() -> argparse.Namespace:
    """Parse inference text and optional checkpoint path."""
    parser = argparse.ArgumentParser(
        description="Predict positive or negative sentiment.",
    )
    parser.add_argument(
        "text",
        nargs="+",
        help="Text to classify. Quotation marks are optional.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Trained checkpoint path.",
    )
    return parser.parse_args()


def main() -> None:
    """Load a trained model and classify user-provided text."""
    arguments = parse_arguments()
    text = " ".join(arguments.text)
    model, vocabulary, class_names = load_classifier(arguments.checkpoint)
    predicted_label, confidence, probabilities = predict_sentiment(
        text,
        model,
        vocabulary,
        class_names,
    )

    print(f"Text: {text}")
    print(f"Tokens: {tokenize(text)}")
    print(f"Token IDs: {vocabulary.encode(text)}")
    print(f"Prediction: {predicted_label}")
    print(f"Confidence: {confidence:.2%}")
    print(
        "Probabilities: "
        + ", ".join(
            f"{class_name}={probability:.2%}"
            for class_name, probability in zip(
                class_names,
                probabilities,
            )
        )
    )


if __name__ == "__main__":
    main()
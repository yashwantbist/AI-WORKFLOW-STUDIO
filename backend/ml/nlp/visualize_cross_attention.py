"""Train a toy translator and visualize all encoder-decoder attention paths."""

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence

try:
    # Supports: python -m backend.ml.nlp.visualize_cross_attention
    from .seq2seq_transformer import (
        Seq2SeqAttentionMaps,
        Seq2SeqTransformer,
        greedy_decode,
    )
    from .transformer_utils import calculate_next_token_loss
except ImportError:
    # Supports: python backend/ml/nlp/visualize_cross_attention.py
    from seq2seq_transformer import (
        Seq2SeqAttentionMaps,
        Seq2SeqTransformer,
        greedy_decode,
    )
    from transformer_utils import calculate_next_token_loss


SOURCE_VOCABULARY = {
    "<pad>": 0,
    "<unk>": 1,
    "bonjour": 2,
    "tout": 3,
    "le": 4,
    "monde": 5,
    "merci": 6,
    "ami": 7,
    "au": 8,
    "revoir": 9,
}
TARGET_VOCABULARY = {
    "<pad>": 0,
    "<bos>": 1,
    "<eos>": 2,
    "hello": 3,
    "everyone": 4,
    "thank": 5,
    "you": 6,
    "friend": 7,
    "goodbye": 8,
}
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[3]
    / "images"
    / "cross_attention.png"
)


@dataclass(frozen=True)
class ToyTranslation:
    """One source-token sequence and its target-token sequence."""

    source_tokens: tuple[str, ...]
    target_tokens: tuple[str, ...]


TOY_TRANSLATIONS = (
    ToyTranslation(("bonjour",), ("<bos>", "hello", "<eos>")),
    ToyTranslation(
        ("bonjour", "tout", "le", "monde"),
        ("<bos>", "hello", "everyone", "<eos>"),
    ),
    ToyTranslation(("merci",), ("<bos>", "thank", "you", "<eos>")),
    ToyTranslation(
        ("merci", "ami"),
        ("<bos>", "thank", "you", "friend", "<eos>"),
    ),
    ToyTranslation(("au", "revoir"), ("<bos>", "goodbye", "<eos>")),
    ToyTranslation(
        ("bonjour", "ami"),
        ("<bos>", "hello", "friend", "<eos>"),
    ),
)


def encode_tokens(
    tokens: tuple[str, ...],
    vocabulary: dict[str, int],
) -> Tensor:
    """Convert readable tokens into model-readable integer IDs."""
    unknown_id = vocabulary.get("<unk>", 0)
    return torch.tensor(
        [vocabulary.get(token, unknown_id) for token in tokens],
        dtype=torch.long,
    )


def create_toy_batch() -> tuple[Tensor, Tensor]:
    """Pad all toy translations into two batch-first tensors."""
    source_sequences = [
        encode_tokens(example.source_tokens, SOURCE_VOCABULARY)
        for example in TOY_TRANSLATIONS
    ]
    target_sequences = [
        encode_tokens(example.target_tokens, TARGET_VOCABULARY)
        for example in TOY_TRANSLATIONS
    ]
    source_token_ids = pad_sequence(
        source_sequences,
        batch_first=True,
        padding_value=SOURCE_VOCABULARY["<pad>"],
    )
    target_token_ids = pad_sequence(
        target_sequences,
        batch_first=True,
        padding_value=TARGET_VOCABULARY["<pad>"],
    )
    return source_token_ids, target_token_ids


def create_model() -> Seq2SeqTransformer:
    """Create a small CPU-friendly encoder-decoder Transformer."""
    return Seq2SeqTransformer(
        source_vocabulary_size=len(SOURCE_VOCABULARY),
        target_vocabulary_size=len(TARGET_VOCABULARY),
        source_padding_id=SOURCE_VOCABULARY["<pad>"],
        target_padding_id=TARGET_VOCABULARY["<pad>"],
        model_dimension=32,
        number_of_heads=4,
        number_of_encoder_layers=1,
        number_of_decoder_layers=1,
        feed_forward_dimension=64,
        dropout=0.0,
        maximum_sequence_length=16,
    )


def train_toy_translator(
    model: Seq2SeqTransformer,
    training_steps: int,
) -> tuple[float, float]:
    """Teach the model the tiny examples with teacher forcing."""
    if training_steps < 1:
        raise ValueError("training_steps must be positive")

    source_token_ids, target_token_ids = create_toy_batch()
    decoder_input_ids = target_token_ids[:, :-1]
    expected_next_ids = target_token_ids[:, 1:]
    expected_next_mask = expected_next_ids.ne(
        TARGET_VOCABULARY["<pad>"]
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    final_loss = 0.0
    final_accuracy = 0.0

    model.train()
    for step in range(1, training_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = model(source_token_ids, decoder_input_ids)
        loss = calculate_next_token_loss(
            logits,
            expected_next_ids,
            expected_next_mask,
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        predictions = logits.argmax(dim=-1)
        correct = (
            predictions[expected_next_mask]
            == expected_next_ids[expected_next_mask]
        ).sum().item()
        total = expected_next_mask.sum().item()
        final_loss = loss.item()
        final_accuracy = 100 * correct / total

        if step == 1 or step % 50 == 0 or step == training_steps:
            print(
                f"Step {step}/{training_steps} | "
                f"Loss: {final_loss:.4f} | "
                f"Token accuracy: {final_accuracy:.2f}%"
            )

        if step >= 50 and final_loss < 0.005 and final_accuracy == 100:
            print(f"Early stop: toy dataset learned at step {step}.")
            break

    return final_loss, final_accuracy


def decode_target_ids(token_ids: Tensor) -> list[str]:
    """Convert generated target IDs back into readable words."""
    id_to_token = {
        token_id: token
        for token, token_id in TARGET_VOCABULARY.items()
    }
    decoded_tokens: list[str] = []
    for token_id in token_ids.tolist():
        token = id_to_token.get(token_id, "<unk>")
        if token == "<bos>":
            continue
        if token in {"<eos>", "<pad>"}:
            break
        decoded_tokens.append(token)
    return decoded_tokens


def annotate_matrix(axis, values: Tensor) -> None:
    """Write numeric attention values into their heatmap cells."""
    for row_index in range(values.size(0)):
        for column_index in range(values.size(1)):
            value = values[row_index, column_index].item()
            axis.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value > 0.55 else "black",
                fontsize=8,
            )


def draw_attention_matrix(
    axis,
    values: Tensor,
    key_tokens: tuple[str, ...],
    query_tokens: tuple[str, ...],
    title: str,
    color_map: str,
) -> None:
    """Draw one labeled query-by-key attention matrix."""
    values = values.detach().float().cpu()
    image = axis.imshow(values, cmap=color_map, vmin=0, vmax=1)
    axis.set_title(title)
    axis.set_xlabel("Key / information source")
    axis.set_ylabel("Query / token reading")
    axis.set_xticks(
        range(len(key_tokens)),
        labels=key_tokens,
        rotation=45,
    )
    axis.set_yticks(range(len(query_tokens)), labels=query_tokens)
    annotate_matrix(axis, values)
    axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)


def save_attention_visualization(
    attention_maps: Seq2SeqAttentionMaps,
    source_tokens: tuple[str, ...],
    decoder_input_tokens: tuple[str, ...],
    output_path: Path,
) -> None:
    """Save encoder self-, decoder self-, and cross-attention heatmaps."""
    import matplotlib.pyplot as plt

    encoder_weights = attention_maps.encoder_self_attention[-1][0, 0]
    decoder_weights = attention_maps.decoder_self_attention[-1][0, 0]
    cross_weights = attention_maps.cross_attention[-1][0, 0]
    figure, axes = plt.subplots(1, 3, figsize=(17, 5))

    draw_attention_matrix(
        axes[0],
        encoder_weights,
        source_tokens,
        source_tokens,
        "Encoder self-attention",
        "Greens",
    )
    draw_attention_matrix(
        axes[1],
        decoder_weights,
        decoder_input_tokens,
        decoder_input_tokens,
        "Decoder causal self-attention",
        "Blues",
    )
    draw_attention_matrix(
        axes[2],
        cross_weights,
        source_tokens,
        decoder_input_tokens,
        "Cross-attention: target queries source",
        "Purples",
    )

    figure.suptitle("Encoder-Decoder Transformer Attention Flow")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and visualize a toy encoder-decoder Transformer.",
    )
    parser.add_argument("--training-steps", type=int, default=300)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    if arguments.training_steps < 1:
        parser.error("--training-steps must be positive")
    return arguments


def main() -> None:
    """Train the toy translator, run inference, and save attention maps."""
    arguments = parse_arguments()
    torch.manual_seed(42)
    model = create_model()
    final_loss, final_accuracy = train_toy_translator(
        model,
        arguments.training_steps,
    )
    model.eval()

    source_tokens = ("bonjour", "tout", "le", "monde")
    source_token_ids = encode_tokens(
        source_tokens,
        SOURCE_VOCABULARY,
    ).unsqueeze(0)
    generated_ids = greedy_decode(
        model=model,
        source_token_ids=source_token_ids,
        beginning_of_sequence_id=TARGET_VOCABULARY["<bos>"],
        end_of_sequence_id=TARGET_VOCABULARY["<eos>"],
        maximum_new_tokens=5,
    )
    generated_tokens = decode_target_ids(generated_ids[0])

    decoder_input_tokens = ("<bos>", "hello", "everyone")
    decoder_input_ids = encode_tokens(
        decoder_input_tokens,
        TARGET_VOCABULARY,
    ).unsqueeze(0)
    with torch.inference_mode():
        _, attention_maps = model(
            source_token_ids,
            decoder_input_ids,
            return_attention=True,
        )

    save_attention_visualization(
        attention_maps=attention_maps,
        source_tokens=source_tokens,
        decoder_input_tokens=decoder_input_tokens,
        output_path=arguments.output,
    )

    print(f"\nSource: {' '.join(source_tokens)}")
    print(f"Generated translation: {' '.join(generated_tokens)}")
    print(f"Final toy loss: {final_loss:.4f}")
    print(f"Final toy token accuracy: {final_accuracy:.2f}%")
    print(f"Saved attention visualization to: {arguments.output}")


if __name__ == "__main__":
    main()

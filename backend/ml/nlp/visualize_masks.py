"""Visualize causal permissions and the resulting attention probabilities."""

import argparse
from pathlib import Path

import torch
from torch import Tensor

try:
    # Supports: python -m backend.ml.nlp.visualize_masks
    from .masks import create_causal_attention_mask
    from .transformer_decoder import CausalLanguageModel
    from .transformer_utils import (
        assert_no_future_attention,
        format_boolean_matrix,
    )
except ImportError:
    # Supports: python backend/ml/nlp/visualize_masks.py
    from masks import create_causal_attention_mask
    from transformer_decoder import CausalLanguageModel
    from transformer_utils import (
        assert_no_future_attention,
        format_boolean_matrix,
    )


TOKENS = ("The", "cat", "sat", "on", "mat")
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[3]
    / "images"
    / "causal_attention_mask.png"
)


def annotate_matrix(axis, values: Tensor) -> None:
    """Write each matrix value into its heatmap cell."""
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
                fontsize=9,
            )


def configure_axis(axis, title: str, tokens: tuple[str, ...]) -> None:
    """Add matching query/key labels to a heatmap."""
    axis.set_title(title)
    axis.set_xlabel("Key position (information source)")
    axis.set_ylabel("Query position (token reading)")
    axis.set_xticks(range(len(tokens)), labels=tokens, rotation=45)
    axis.set_yticks(range(len(tokens)), labels=tokens)


def save_mask_visualization(
    causal_mask: Tensor,
    attention_weights: Tensor,
    output_path: Path,
    tokens: tuple[str, ...] = TOKENS,
) -> None:
    """Save the rule matrix beside one learned attention-head matrix."""
    import matplotlib.pyplot as plt

    if causal_mask.shape != attention_weights.shape:
        raise ValueError(
            "causal_mask and attention_weights must have the same shape"
        )
    if causal_mask.shape != (len(tokens), len(tokens)):
        raise ValueError("Matrix dimensions must match the token count")

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    mask_values = causal_mask.float().cpu()
    probability_values = attention_weights.detach().float().cpu()

    first_image = axes[0].imshow(
        mask_values,
        cmap="Greens",
        vmin=0,
        vmax=1,
    )
    configure_axis(axes[0], "Causal permission matrix", tokens)
    annotate_matrix(axes[0], mask_values)
    figure.colorbar(first_image, ax=axes[0], fraction=0.046, pad=0.04)

    second_image = axes[1].imshow(
        probability_values,
        cmap="Blues",
        vmin=0,
        vmax=1,
    )
    configure_axis(axes[1], "Actual probabilities: layer 1, head 1", tokens)
    annotate_matrix(axes[1], probability_values)
    figure.colorbar(second_image, ax=axes[1], fraction=0.046, pad=0.04)

    figure.suptitle("Decoder-Only Transformer: Future Positions Are Blocked")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize causal self-attention.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    """Build a tiny decoder and prove it assigns zero weight to the future."""
    arguments = parse_arguments()
    torch.manual_seed(42)

    # IDs 0 and 1 are reserved for padding and unknown tokens. The five
    # visible words therefore use IDs 2 through 6.
    token_ids = torch.tensor([[2, 3, 4, 5, 6]])
    model = CausalLanguageModel(
        vocabulary_size=16,
        padding_id=0,
        model_dimension=16,
        number_of_heads=4,
        number_of_layers=1,
        feed_forward_dimension=32,
        dropout=0.0,
        maximum_sequence_length=len(TOKENS),
    )
    model.eval()

    with torch.inference_mode():
        logits, attention_by_layer = model(
            token_ids,
            return_attention=True,
        )

    causal_mask = create_causal_attention_mask(len(TOKENS))
    first_head_weights = attention_by_layer[0][0, 0]
    assert_no_future_attention(first_head_weights)
    save_mask_visualization(
        causal_mask=causal_mask,
        attention_weights=first_head_weights,
        output_path=arguments.output,
    )

    print("Five-token causal permission matrix:")
    print(format_boolean_matrix(causal_mask))
    print(f"\nDecoder logits shape: {tuple(logits.shape)}")
    print("Verification passed: all future-token probabilities are zero.")
    print(f"Saved visualization to: {arguments.output}")


if __name__ == "__main__":
    main()

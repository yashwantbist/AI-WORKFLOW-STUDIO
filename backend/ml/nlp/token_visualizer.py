"""Terminal and HTML visualizations for tokenization results."""

from __future__ import annotations

from collections.abc import Sequence
from html import escape
from pathlib import Path
from typing import Protocol


class ResultLike(Protocol):
    """Attributes required by the visualization functions."""

    strategy: str
    text: str
    tokens: Sequence[str]
    token_ids: Sequence[int]
    token_count: int
    vocabulary_size: int
    context_window: int
    remaining_tokens: int
    fits_context: bool
    context_percentage: float


def visible_token(token: str) -> str:
    """Make whitespace and control characters readable."""
    return (
        token.replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace("\t", "\\t")
        .replace("\r", "\\r")
        .replace(" ", "␠")
    )


def _table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
) -> str:
    widths = [
        max(
            len(headers[column]),
            *(
                len(row[column])
                for row in rows
            ),
        )
        for column in range(len(headers))
    ]

    def format_row(row: Sequence[str]) -> str:
        return " | ".join(
            value.ljust(width)
            for value, width in zip(row, widths)
        )

    separator = "-+-".join("-" * width for width in widths)
    return "\n".join(
        [
            format_row(headers),
            separator,
            *(format_row(row) for row in rows),
        ]
    )


def token_rows(result: ResultLike) -> list[list[str]]:
    """Create display-ready rows for each token."""
    return [
        [
            str(index),
            visible_token(token),
            str(token_id),
            str(len(token)),
        ]
        for index, (token, token_id) in enumerate(
            zip(result.tokens, result.token_ids),
        )
    ]


def print_token_table(result: ResultLike) -> None:
    """Print token details and context usage."""
    rows = token_rows(result)
    print(f"Strategy: {result.strategy}")
    print(f"Original text: {result.text!r}")
    print(
        _table(
            ("Index", "Token", "Local ID", "Characters"),
            rows,
        )
        if rows
        else "(No tokens were produced.)"
    )

    status = "fits" if result.fits_context else "does not fit"
    print(f"\nToken count: {result.token_count}")
    print(f"Local vocabulary size: {result.vocabulary_size}")
    print(
        "Context usage: "
        f"{result.token_count}/{result.context_window} "
        f"({result.context_percentage:.2f}%) — {status}"
    )
    print(f"Remaining token capacity: {result.remaining_tokens}")


def print_comparison_table(results: Sequence[ResultLike]) -> None:
    """Print one summary row per tokenizer."""
    rows = [
        [
            result.strategy,
            str(result.token_count),
            str(result.vocabulary_size),
            f"{result.context_percentage:.2f}%",
            "yes" if result.fits_context else "no",
        ]
        for result in results
    ]
    print(
        _table(
            (
                "Strategy",
                "Tokens",
                "Vocabulary",
                "Context used",
                "Fits",
            ),
            rows,
        )
    )


def save_html_visualization(
    result: ResultLike,
    output_path: Path,
) -> Path:
    """Save a standalone HTML page containing colored token cards."""
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    token_cards = "\n".join(
        (
            '<span class="token">'
            f'<span class="token-text">{escape(visible_token(token))}</span>'
            f'<span class="token-id">ID {token_id}</span>'
            "</span>"
        )
        for token, token_id in zip(result.tokens, result.token_ids)
    )

    fit_text = "Fits inside the context window" if result.fits_context else (
        "Exceeds the context window"
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Tokenization Explorer</title>
  <style>
    body {{
      max-width: 960px;
      margin: 40px auto;
      padding: 0 20px;
      font-family: system-ui, sans-serif;
      line-height: 1.5;
      background: #f7f7f8;
      color: #202124;
    }}
    .panel {{
      background: white;
      border: 1px solid #dedede;
      border-radius: 16px;
      padding: 24px;
    }}
    .tokens {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin: 20px 0;
    }}
    .token {{
      display: inline-flex;
      flex-direction: column;
      padding: 8px 10px;
      border: 1px solid #b8b8b8;
      border-radius: 10px;
      background: #eef2ff;
    }}
    .token-text {{
      font-family: ui-monospace, monospace;
      font-weight: 700;
    }}
    .token-id {{
      font-size: 0.75rem;
      color: #555;
    }}
    code {{
      overflow-wrap: anywhere;
    }}
  </style>
</head>
<body>
  <main class="panel">
    <h1>Tokenization Explorer</h1>
    <p><strong>Strategy:</strong> {escape(result.strategy)}</p>
    <p><strong>Original text:</strong> <code>{escape(result.text)}</code></p>
    <div class="tokens">{token_cards}</div>
    <p><strong>Token count:</strong> {result.token_count}</p>
    <p><strong>Local vocabulary size:</strong> {result.vocabulary_size}</p>
    <p>
      <strong>Context:</strong>
      {result.token_count}/{result.context_window}
      ({result.context_percentage:.2f}%) — {fit_text}
    </p>
  </main>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")
    return output_path

"""Deterministic token-to-ID vocabulary utilities for the tokenization lab."""

from collections import Counter
from collections.abc import Iterable, Sequence


PAD_TOKEN = "<pad>"
UNKNOWN_TOKEN = "<unk>"


class Vocabulary:
    """Map string tokens to stable integer IDs.

    This vocabulary is intentionally small and local to the demo. The token IDs
    are not OpenAI, BERT, or SentencePiece production token IDs.
    """

    def __init__(self, token_to_id: dict[str, int]) -> None:
        if token_to_id.get(PAD_TOKEN) != 0:
            raise ValueError(f"{PAD_TOKEN} must have token ID 0")
        if token_to_id.get(UNKNOWN_TOKEN) != 1:
            raise ValueError(f"{UNKNOWN_TOKEN} must have token ID 1")

        expected_ids = set(range(len(token_to_id)))
        if set(token_to_id.values()) != expected_ids:
            raise ValueError("Vocabulary IDs must be contiguous")

        self._token_to_id = dict(token_to_id)
        self._id_to_token = {
            token_id: token for token, token_id in self._token_to_id.items()
        }

    @classmethod
    def build(
        cls,
        token_sequences: Iterable[Sequence[str]],
        minimum_frequency: int = 1,
    ) -> "Vocabulary":
        """Build a deterministic vocabulary from one or more token sequences."""
        if minimum_frequency < 1:
            raise ValueError("minimum_frequency must be at least 1")

        counts = Counter(
            token
            for sequence in token_sequences
            for token in sequence
        )
        ordered_tokens = sorted(
            (
                token
                for token, count in counts.items()
                if count >= minimum_frequency
            ),
            key=lambda token: (-counts[token], token),
        )

        token_to_id = {
            PAD_TOKEN: 0,
            UNKNOWN_TOKEN: 1,
        }
        token_to_id.update(
            {
                token: token_id
                for token_id, token in enumerate(ordered_tokens, start=2)
            }
        )
        return cls(token_to_id)

    @property
    def padding_id(self) -> int:
        """Return the ID reserved for padding."""
        return self._token_to_id[PAD_TOKEN]

    @property
    def unknown_id(self) -> int:
        """Return the ID reserved for unknown tokens."""
        return self._token_to_id[UNKNOWN_TOKEN]

    def token_id(self, token: str) -> int:
        """Return a token ID, falling back to the unknown-token ID."""
        return self._token_to_id.get(token, self.unknown_id)

    def encode(self, tokens: Sequence[str]) -> list[int]:
        """Convert a token sequence to integer IDs."""
        return [self.token_id(token) for token in tokens]

    def decode(self, token_ids: Sequence[int]) -> list[str]:
        """Convert integer IDs back to token strings."""
        return [
            self._id_to_token.get(token_id, UNKNOWN_TOKEN)
            for token_id in token_ids
        ]

    def to_dict(self) -> dict[str, int]:
        """Return a copy of the token-to-ID mapping."""
        return dict(self._token_to_id)

    def __len__(self) -> int:
        return len(self._token_to_id)

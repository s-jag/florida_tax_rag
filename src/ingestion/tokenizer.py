"""Token counting utilities using tiktoken."""

from __future__ import annotations

import tiktoken

# Module-level encoder cache
_encoder: tiktoken.Encoding | None = None


def get_encoder() -> tiktoken.Encoding:
    """Get or create the tiktoken encoder.

    Uses cl100k_base encoding which is compatible with Claude and GPT-4.
    """
    global _encoder
    if _encoder is None:
        _encoder = tiktoken.get_encoding("cl100k_base")
    return _encoder


def count_tokens(text: str) -> int:
    """Count the number of tokens in a text string.

    Args:
        text: The text to tokenize

    Returns:
        Number of tokens
    """
    if not text:
        return 0
    return len(get_encoder().encode(text))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to a maximum number of tokens.

    Args:
        text: The text to truncate
        max_tokens: Maximum number of tokens

    Returns:
        Truncated text
    """
    if not text:
        return ""

    encoder = get_encoder()
    tokens = encoder.encode(text)

    if len(tokens) <= max_tokens:
        return text

    return encoder.decode(tokens[:max_tokens])

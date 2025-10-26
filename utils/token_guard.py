from __future__ import annotations

"""Utilities for preventing runaway token costs."""

max_context: int = 128_000


class TokenLimitError(Exception):
    """Raised when a request would exceed the model's context window."""


def estimate_token_count(text: str) -> int:
    """Return the number of tokens for ``text`` using tiktoken for GPT-3.5."""
    try:
        import tiktoken
    except ModuleNotFoundError:
        return len(text.split())

    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(text))


def enforce_buffer(scene_tokens: int, buffer_tokens: int, max_context: int) -> None:
    """Raise :class:`TokenLimitError` if total tokens exceed the limit."""

    total = scene_tokens + buffer_tokens
    if total >= max_context:
        raise TokenLimitError(
            f"{total} tokens would exceed context window of {max_context}"
        )

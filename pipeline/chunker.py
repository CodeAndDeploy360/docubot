"""Token-aware chunking with overlap (tiktoken)."""

from __future__ import annotations

from typing import Any


def _tiktoken():
    try:
        import tiktoken
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The 'tiktoken' package is required for chunking. "
            "Install dependencies: pip install -r requirements.txt"
        ) from exc
    return tiktoken


def get_encoder(model: str = "gpt-4o-mini") -> Any:
    tiktoken = _tiktoken()
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def chunk_text_by_tokens(
    text: str,
    *,
    max_tokens: int = 500,
    overlap_tokens: int = 50,
    model: str = "gpt-4o-mini",
) -> list[str]:
    enc = get_encoder(model)
    tokens = enc.encode(text or "")
    if not tokens:
        return []
    chunks: list[str] = []
    start = 0
    step = max(1, max_tokens - overlap_tokens)
    while start < len(tokens):
        piece = tokens[start : start + max_tokens]
        chunks.append(enc.decode(piece))
        start += step
    return chunks

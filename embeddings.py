"""Chunking and embedding for weather narrative text.

The encoder is a module-level singleton. Loading a SentenceTransformer costs
seconds and hundreds of megabytes, so per-request loading would make every
search slow and hold several copies of the weights at once.

sentence-transformers is imported lazily, inside _load_model, so importing
this module - which the Flask app does at startup - does not pull in torch
until something actually needs to embed. That keeps /healthz answering while
the model is still warming.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Sequence

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# Kept identical to the existing news pipeline so both vector tables share a
# distance operator and a query encoder. Unknown models raise rather than
# defaulting, because a guessed width fails much later and much less clearly.
MODEL_DIMENSIONS = {
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-MiniLM-L12-v2": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": 768,
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
}


def model_dimension(model_name: str) -> int:
    """Output width for a known model."""
    try:
        return MODEL_DIMENSIONS[model_name]
    except KeyError:
        raise ValueError(
            f"Unknown embedding model {model_name!r} - add its output dimension "
            f"to MODEL_DIMENSIONS. Known: {sorted(MODEL_DIMENSIONS)}"
        ) from None


EMBEDDING_DIM = model_dimension(EMBEDDING_MODEL)

# 800 characters is roughly 200 tokens, comfortably inside all-MiniLM-L6-v2's
# 256-token input limit. A larger window would be silently truncated by the
# model, so the tail of every long chunk would never reach the vector - a
# failure that produces plausible embeddings and no error at all.
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "100"))

# Unit-length vectors. Cosine distance ignores magnitude either way, so this
# does not change ranking; it keeps stored vectors directly comparable by dot
# product too, and avoids drift if an inner-product operator is used later.
NORMALIZE = True

_encoder: "Encoder | None" = None
_encoder_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


# A boundary is only worth taking if the window is already mostly full.
# Otherwise a full stop 20 characters in would "win" and emit a near-empty
# chunk, trading a clean break for a useless one.
_MIN_BOUNDARY_FILL = 0.5

_SENTENCE_ENDS = (". ", ".\n", "! ", "!\n", "? ", "?\n")


def _is_break(char: str) -> bool:
    return char.isspace()


def _last_break(text: str, low: int, high: int) -> int:
    """Index of the last whitespace strictly inside (low, high), else -1."""
    for index in range(high - 1, low, -1):
        if _is_break(text[index]):
            return index
    return -1


def _last_paragraph_break(text: str, low: int, high: int) -> int:
    """End index of the last blank-line break inside the window, else -1."""
    position = text.rfind("\n\n", low, high)
    return position if position > low else -1


def _last_sentence_break(text: str, low: int, high: int) -> int:
    """End index just after the last sentence terminator, else -1.

    NWS text is hard-wrapped, so a sentence can end at ".\n" as readily as
    at ". " - both forms are checked.
    """
    best = -1
    for terminator in _SENTENCE_ENDS:
        position = text.rfind(terminator, low, high)
        if position > best:
            best = position
    # Include the terminator itself, exclude the whitespace that follows.
    return best + 1 if best > low else -1


def _choose_break(text: str, start: int, end: int) -> int:
    """Pick where to end a window: paragraph, then sentence, then word.

    Where a chunk ends is a presentation decision as much as a tokenization
    one - chunk_text is what gets shown to the reader as the retrieved
    passage, so a window ending mid-clause is a visible defect, not an
    internal detail.
    """
    floor = start + int((end - start) * _MIN_BOUNDARY_FILL)

    for finder in (_last_paragraph_break, _last_sentence_break):
        boundary = finder(text, start, end)
        if boundary > floor:
            return boundary

    return _last_break(text, start, end)


def _word_start(text: str, position: int, floor: int) -> int:
    """Walk back to the start of the word containing `position`."""
    index = position
    while index > floor and not _is_break(text[index - 1]):
        index -= 1
    return index


def chunk_text(
    text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """Split text into overlapping, word-aligned windows.

    Windows end at a whitespace boundary rather than at an exact character
    count, so a chunk never ends mid-word. That matters twice: the model sees
    whole tokens, and the chunk is also what gets shown to the reader as the
    retrieved passage, so a fragment is a visible defect rather than an
    internal detail.

    Overlap exists so a sentence straddling a boundary is still retrievable -
    without it, text at a seam is split across two vectors and matches neither
    query well.

    Most NWS forecasts fit in a single window and come back unchanged. Alerts,
    where description and instruction are joined, routinely need several.
    """
    if not isinstance(text, str):
        raise TypeError(f"chunk_text expects str, got {type(text).__name__}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if overlap < 0:
        raise ValueError(f"overlap must not be negative, got {overlap}")
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size}); "
            "otherwise the window makes no forward progress"
        )

    cleaned = text.strip()
    if not cleaned:
        return []
    if len(cleaned) <= chunk_size:
        return [cleaned]

    chunks: list[str] = []
    length = len(cleaned)
    start = 0

    while start < length:
        end = min(start + chunk_size, length)

        if end < length:
            boundary = _choose_break(cleaned, start, end)
            # A boundary of -1 means this window holds no break at all - a
            # single token longer than the window, such as a bare URL. Cut it
            # hard: refusing to split would never terminate.
            if boundary > start:
                end = boundary

        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= length:
            break

        # Step back by the overlap, then align to a word start so the next
        # window does not begin mid-token.
        next_start = _word_start(cleaned, max(end - overlap, start + 1), start + 1)
        start = next_start if next_start > start else end

    return chunks


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def _load_model(model_name: str):
    """Import and construct the SentenceTransformer.

    Separated out so tests can substitute a fake without importing torch, and
    so the import cost is paid on first use rather than at module import.
    """
    from sentence_transformers import SentenceTransformer

    cache_folder = os.environ.get("HF_HOME", "/tmp/.cache/huggingface")
    logger.info("Loading embedding model %s", model_name)
    return SentenceTransformer(model_name, cache_folder=cache_folder)


class Encoder:
    """Wraps a loaded model and guarantees plain-float output."""

    def __init__(self, model_name: str, model):
        self.model_name = model_name
        self.model = model

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        raw = self.model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=NORMALIZE,
            show_progress_bar=False,
        )

        vectors = raw.tolist() if hasattr(raw, "tolist") else [list(row) for row in raw]
        # Convert to built-in floats here. repository.to_vector_literal renders
        # values with str(), and a numpy scalar can stringify into something
        # pgvector's parser rejects.
        vectors = [[float(value) for value in vector] for vector in vectors]

        for vector in vectors:
            if len(vector) != EMBEDDING_DIM:
                raise ValueError(
                    f"Model {self.model_name!r} returned {len(vector)}-dimensional "
                    f"vectors, but this project stores VECTOR({EMBEDDING_DIM}). "
                    "Change EMBEDDING_MODEL back, or migrate the column and "
                    "re-embed everything."
                )
        return vectors


def get_encoder(model_name: str | None = None) -> Encoder:
    """Return the shared encoder, loading it on first use (R19)."""
    global _encoder
    name = model_name or EMBEDDING_MODEL

    if _encoder is None or _encoder.model_name != name:
        with _encoder_lock:
            if _encoder is None or _encoder.model_name != name:
                _encoder = Encoder(name, _load_model(name))
    return _encoder


def reset_encoder() -> None:
    """Drop the cached encoder. Used by tests."""
    global _encoder
    with _encoder_lock:
        _encoder = None


def embed_texts(texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
    """Embed a batch of texts."""
    if not texts:
        return []
    return get_encoder().encode(texts, batch_size=batch_size)


def embed_query(text: str) -> list[float]:
    """Embed a single search query with the same model the documents used.

    Using a different model here would produce cosine scores that are
    arithmetically valid and semantically meaningless - the two sets of
    vectors do not share a space.
    """
    return embed_texts([text])[0]


def chunk_document(narrative_text: str, content_hash: str) -> list[dict]:
    """Chunk one document's narrative into rows ready for embedding."""
    return [
        {"chunk_index": index, "chunk_text": chunk, "content_hash": content_hash}
        for index, chunk in enumerate(chunk_text(narrative_text))
    ]


__all__ = [
    "CHUNK_OVERLAP",
    "CHUNK_SIZE",
    "EMBEDDING_DIM",
    "EMBEDDING_MODEL",
    "Encoder",
    "MODEL_DIMENSIONS",
    "NORMALIZE",
    "chunk_document",
    "chunk_text",
    "embed_query",
    "embed_texts",
    "get_encoder",
    "model_dimension",
    "reset_encoder",
]

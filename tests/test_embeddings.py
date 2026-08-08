"""Tests for chunking and the query/document encoder (R12, R13, R19)."""

from __future__ import annotations

import pytest

import embeddings

pytestmark = pytest.mark.fast


@pytest.fixture(autouse=True)
def _isolated_encoder():
    """The encoder is a module-level singleton. Reset it around every test so
    one test's fake model cannot leak into the next."""
    embeddings.reset_encoder()
    yield
    embeddings.reset_encoder()


# --------------------------------------------------------------------------
# R12 - chunking
# --------------------------------------------------------------------------


def test_text_shorter_than_the_window_is_one_chunk():
    text = "Mostly sunny, with a high near 82. South wind around 10 mph."
    assert embeddings.chunk_text(text) == [text]


def test_text_is_stripped_before_chunking():
    assert embeddings.chunk_text("   Sunny today.   ") == ["Sunny today."]


@pytest.mark.parametrize("text", ["", "   ", "\n\n\t "])
def test_empty_text_produces_no_chunks(text):
    assert embeddings.chunk_text(text) == []


def test_non_string_input_is_rejected():
    with pytest.raises(TypeError):
        embeddings.chunk_text(None)


def test_long_text_is_split_into_multiple_chunks():
    text = " ".join(f"word{i}" for i in range(500))  # well over 800 chars
    chunks = embeddings.chunk_text(text, chunk_size=200, overlap=50)
    assert len(chunks) > 1


def test_no_chunk_exceeds_the_window():
    text = " ".join(f"word{i}" for i in range(500))
    for chunk in embeddings.chunk_text(text, chunk_size=200, overlap=50):
        assert len(chunk) <= 200


def test_chunks_never_split_a_word():
    """A window that ends mid-token gives the model a fragment to embed and
    puts a fragment in front of the reader as the retrieved passage."""
    text = " ".join(f"word{i:04d}" for i in range(400))
    chunks = embeddings.chunk_text(text, chunk_size=201, overlap=40)

    for chunk in chunks:
        for token in chunk.split():
            assert token.startswith("word"), f"fragment produced: {token!r}"
            assert len(token) == 8, f"truncated token: {token!r}"


def test_consecutive_chunks_overlap():
    """Overlap is what stops a sentence that straddles a boundary from being
    unretrievable in both directions."""
    text = " ".join(f"word{i:04d}" for i in range(400))
    chunks = embeddings.chunk_text(text, chunk_size=200, overlap=60)

    first_tokens = set(chunks[0].split())
    second_tokens = set(chunks[1].split())
    assert first_tokens & second_tokens, "adjacent chunks share no text"


def test_every_word_survives_chunking():
    """Coverage: no content may be dropped between windows."""
    words = [f"word{i:04d}" for i in range(400)]
    chunks = embeddings.chunk_text(" ".join(words), chunk_size=200, overlap=50)

    seen = set()
    for chunk in chunks:
        seen.update(chunk.split())
    assert set(words) == seen


def test_a_single_token_longer_than_the_window_is_hard_cut():
    """Pathological input must still terminate. A URL with no spaces can
    exceed the window on its own, and refusing to split it would loop."""
    text = "x" * 500
    chunks = embeddings.chunk_text(text, chunk_size=100, overlap=20)

    assert len(chunks) > 1
    assert all(len(chunk) <= 100 for chunk in chunks)
    assert "".join(dict.fromkeys("".join(chunks))) == "x"


def test_chunking_is_deterministic():
    text = " ".join(f"word{i}" for i in range(300))
    assert embeddings.chunk_text(text) == embeddings.chunk_text(text)


@pytest.mark.parametrize(
    "chunk_size,overlap",
    [(0, 0), (-1, 0), (100, 100), (100, 150), (100, -1)],
)
def test_invalid_window_settings_are_rejected(chunk_size, overlap):
    """overlap >= chunk_size makes no forward progress."""
    with pytest.raises(ValueError):
        embeddings.chunk_text("some text here", chunk_size=chunk_size, overlap=overlap)


def test_chunks_prefer_to_break_at_a_paragraph():
    """chunk_text is shown to the reader as the retrieved passage, so where a
    window ends is a presentation decision, not just a tokenization one."""
    first = "A" * 300
    second = "B" * 300
    third = "C" * 300
    chunks = embeddings.chunk_text(
        f"{first}\n\n{second}\n\n{third}", chunk_size=700, overlap=50
    )
    assert chunks[0].endswith("B" * 300), "should have ended at the paragraph break"


def test_chunks_prefer_to_break_at_a_sentence():
    sentences = " ".join(f"This is sentence number {i:03d}." for i in range(40))
    chunks = embeddings.chunk_text(sentences, chunk_size=300, overlap=40)

    # Every chunk but the last should end at a full stop rather than mid-clause.
    for chunk in chunks[:-1]:
        assert chunk.endswith("."), f"chunk ended mid-sentence: ...{chunk[-40:]!r}"


def test_a_sentence_break_too_early_in_the_window_is_not_used():
    """Preferring a boundary must not produce a nearly empty chunk. A full stop
    20 characters in is a worse break than a word boundary at 780."""
    text = "Hi. " + " ".join(f"word{i:04d}" for i in range(200))
    chunks = embeddings.chunk_text(text, chunk_size=400, overlap=50)
    assert len(chunks[0]) > 200, f"first chunk collapsed to {len(chunks[0])} chars"


def test_word_boundaries_still_apply_when_there_is_no_punctuation():
    text = " ".join(f"word{i:04d}" for i in range(300))
    for chunk in embeddings.chunk_text(text, chunk_size=250, overlap=40):
        for token in chunk.split():
            assert len(token) == 8, f"fragment produced: {token!r}"


def test_real_alert_text_chunks_as_expected(alerts_ca):
    """The live-shaped case: a real NWS alert with description + instruction."""
    properties = alerts_ca["features"][0]["properties"]
    narrative = f"{properties['description']}\n\n{properties['instruction']}"

    chunks = embeddings.chunk_text(narrative)
    assert len(chunks) >= 2, "a real alert should exercise the chunker"
    assert all(chunk.strip() for chunk in chunks)
    assert all(len(chunk) <= embeddings.CHUNK_SIZE for chunk in chunks)


def test_default_window_matches_the_model_input_limit():
    """all-MiniLM-L6-v2 truncates at 256 tokens, roughly 1000 characters.
    A window above that would be silently cut by the model, so the tail of
    every long chunk would never reach the vector."""
    assert embeddings.CHUNK_SIZE <= 1000
    assert 0 < embeddings.CHUNK_OVERLAP < embeddings.CHUNK_SIZE


# --------------------------------------------------------------------------
# R13, R19 - the encoder
# --------------------------------------------------------------------------


class FakeModel:
    """Stands in for SentenceTransformer without downloading 90MB of weights."""

    def __init__(self, dimension=embeddings.EMBEDDING_DIM):
        self.dimension = dimension
        self.encode_calls: list[list[str]] = []
        self.encode_kwargs: list[dict] = []

    def encode(self, texts, batch_size=32, normalize_embeddings=False, show_progress_bar=False):
        self.encode_calls.append(list(texts))
        self.encode_kwargs.append(
            {
                "batch_size": batch_size,
                "normalize_embeddings": normalize_embeddings,
                "show_progress_bar": show_progress_bar,
            }
        )
        return [[float(len(text) % 7)] * self.dimension for text in texts]


@pytest.fixture
def fake_model(monkeypatch) -> FakeModel:
    model = FakeModel()
    loads: list[str] = []

    def _load(name):
        loads.append(name)
        return model

    monkeypatch.setattr(embeddings, "_load_model", _load)
    model.loads = loads
    return model


def test_encoder_is_loaded_once_and_reused(fake_model):
    """R19 - loading the model per request would add seconds to every search
    and hold several copies of the weights in memory at once."""
    embeddings.get_encoder()
    embeddings.get_encoder()
    embeddings.embed_query("flooding")

    assert fake_model.loads == [embeddings.EMBEDDING_MODEL]


def test_embed_texts_returns_one_vector_per_input(fake_model):
    vectors = embeddings.embed_texts(["alpha", "beta", "gamma"])
    assert len(vectors) == 3
    assert all(len(vector) == embeddings.EMBEDDING_DIM for vector in vectors)


def test_embed_texts_returns_plain_python_floats(fake_model):
    """to_vector_literal renders values with str(). numpy scalars stringify
    in forms pgvector's parser rejects, so conversion has to happen here."""
    vector = embeddings.embed_texts(["alpha"])[0]
    assert all(type(value) is float for value in vector)


def test_embed_texts_with_no_input_does_not_touch_the_model(fake_model):
    assert embeddings.embed_texts([]) == []
    assert fake_model.encode_calls == []


def test_embed_query_returns_a_single_vector(fake_model):
    vector = embeddings.embed_query("flash flood risk this weekend")
    assert len(vector) == embeddings.EMBEDDING_DIM
    assert isinstance(vector[0], float)


def test_query_and_documents_use_the_same_model(fake_model):
    """A query embedded by a different model than the documents produces
    cosine scores that are arithmetically valid and semantically meaningless."""
    embeddings.embed_texts(["a document"])
    embeddings.embed_query("a query")
    assert fake_model.loads == [embeddings.EMBEDDING_MODEL]


def test_normalization_is_actually_requested_of_the_model(fake_model):
    """Assert on what the model was called with, not on the constant. A test
    that reads the setting it is meant to verify cannot fail."""
    embeddings.embed_texts(["alpha"])
    assert fake_model.encode_kwargs[0]["normalize_embeddings"] is True


def test_progress_bars_are_suppressed(fake_model):
    """A progress bar inside a Flask request writes junk to the app log."""
    embeddings.embed_query("alpha")
    assert fake_model.encode_kwargs[0]["show_progress_bar"] is False


def test_batch_size_reaches_the_model(fake_model):
    vectors = embeddings.embed_texts([f"text {i}" for i in range(70)], batch_size=16)
    assert len(vectors) == 70
    assert fake_model.encode_kwargs[0]["batch_size"] == 16


# --------------------------------------------------------------------------
# model / dimension agreement
# --------------------------------------------------------------------------


def test_known_model_dimension():
    assert embeddings.model_dimension("sentence-transformers/all-MiniLM-L6-v2") == 384


def test_unknown_model_raises_rather_than_guessing():
    """Guessing a dimension would let wrong-width vectors reach a VECTOR(384)
    column and fail much later, with a driver error naming neither."""
    with pytest.raises(ValueError, match="dimension"):
        embeddings.model_dimension("some/unlisted-model")


def test_configured_model_matches_the_declared_dimension():
    assert embeddings.model_dimension(embeddings.EMBEDDING_MODEL) == embeddings.EMBEDDING_DIM


def test_encoder_rejects_a_model_of_the_wrong_width(monkeypatch):
    """If the loaded model returns 768-dim vectors while the column is
    VECTOR(384), fail at load with a message naming both."""
    monkeypatch.setattr(embeddings, "_load_model", lambda name: FakeModel(dimension=768))
    monkeypatch.setattr(embeddings, "EMBEDDING_DIM", 384)

    with pytest.raises(ValueError, match="768"):
        embeddings.embed_texts(["alpha"])

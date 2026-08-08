"""Tests for the Lakebase data access layer (R9, R11, R14-R16, R18, R20).

These verify how the SQL is built - joins, casts, conflict targets, parameter
order. They deliberately do not verify that the SQL is valid Postgres; that is
what tests/test_live_lakebase.py does against a real instance.
"""

from __future__ import annotations

import json

import pytest

import repository
from weather_client import WeatherDocument

pytestmark = pytest.mark.fast


def make_document(doc_id="urn:oid:1", **overrides) -> WeatherDocument:
    fields = {
        "id": doc_id,
        "location": "Cook County",
        "latitude": 41.8781,
        "longitude": -87.6298,
        "source_type": "alert",
        "event": "Flash Flood Warning",
        "headline": "Flash Flood Warning issued...",
        "narrative_text": "Rivers are rising fast.",
        "issued_at": "2026-08-07T12:00:00-05:00",
        "effective_at": "2026-08-07T12:00:00-05:00",
        "expires_at": "2026-08-08T00:00:00-05:00",
        "severity": "Severe",
        "payload": {"properties": {"event": "Flash Flood Warning"}},
    }
    fields.update(overrides)
    return WeatherDocument(**fields)


# --------------------------------------------------------------------------
# vector literals
# --------------------------------------------------------------------------


def full_width(*leading: float) -> list[float]:
    """A correctly sized embedding beginning with the given values."""
    return list(leading) + [0.0] * (repository.VECTOR_DIM - len(leading))


def test_vector_literal_uses_pgvector_bracket_form():
    """pgvector's text input is '[1,2,3]'. A Postgres array literal
    '{1,2,3}' is a different type and will not cast."""
    literal = repository.to_vector_literal(full_width(0.1, -0.2, 0.3))

    assert literal.startswith("[0.1,-0.2,0.3,")
    assert literal.endswith("]")
    assert literal.count(",") == repository.VECTOR_DIM - 1


def test_vector_literal_accepts_numpy_style_values():
    """model.encode() returns numpy float32, not Python float. Left
    unconverted, str() on it yields values the vector parser rejects."""

    class Faux(float):
        def __str__(self):  # what numpy scalars do in some versions
            return f"np.float32({float(self)})"

    literal = repository.to_vector_literal([Faux(1.5)] + [Faux(0.0)] * (repository.VECTOR_DIM - 1))
    assert literal.startswith("[1.5,0.0,")
    assert "np.float32" not in literal


def test_vector_literal_rejects_the_wrong_width():
    """A 768-dim vector reaching a VECTOR(384) column fails deep inside a
    batch insert. Catching it here names the actual problem."""
    with pytest.raises(ValueError, match="384"):
        repository.to_vector_literal([0.0] * 768)


# --------------------------------------------------------------------------
# R10, S2 - document upsert
# --------------------------------------------------------------------------


def test_upsert_documents_writes_one_batch(fake_db):
    written = repository.upsert_documents([make_document("a"), make_document("b")])

    assert written == 2
    assert len(fake_db.execute_values_calls) == 1
    assert len(fake_db.last_write["rows"]) == 2


def test_upsert_documents_updates_on_conflict(fake_db):
    """Re-running sync must refresh a document, not fail and not duplicate."""
    repository.upsert_documents([make_document()])
    sql = fake_db.last_write["sql"].lower()

    assert "on conflict (id) do update" in sql
    assert "narrative_text = excluded.narrative_text" in sql
    assert "content_hash = excluded.content_hash" in sql


def test_upsert_documents_serializes_payload_as_jsonb(fake_db):
    repository.upsert_documents([make_document()])

    assert "::jsonb" in fake_db.last_write["template"]
    payload_values = [value for value in fake_db.last_write["rows"][0] if isinstance(value, str)]
    assert any(
        value.startswith("{") and json.loads(value).get("properties")
        for value in payload_values
    ), "payload must reach the driver as a JSON string"


def test_upsert_documents_deduplicates_ids_within_one_batch(fake_db):
    """Postgres raises "ON CONFLICT DO UPDATE command cannot affect row a
    second time" when one statement touches the same id twice. Collapsing
    duplicates here keeps a repeated id in the input from failing the batch."""
    first = make_document("dup", narrative_text="original text")
    second = make_document("dup", narrative_text="amended text")

    written = repository.upsert_documents([first, second])

    assert written == 1
    rows = fake_db.last_write["rows"]
    assert len(rows) == 1
    assert "amended text" in rows[0], "the later revision wins"


def test_upsert_documents_accepts_plain_dicts(fake_db):
    """The ingestion script works with rows, not WeatherDocument objects."""
    written = repository.upsert_documents([make_document().as_row()])
    assert written == 1


def test_upsert_documents_with_nothing_to_write_touches_no_connection(fake_db):
    assert repository.upsert_documents([]) == 0
    assert fake_db.connection.executed == []


# --------------------------------------------------------------------------
# R11 - finding what still needs embedding
# --------------------------------------------------------------------------


def test_fetch_unembedded_joins_on_id_hash_and_model(fake_db):
    """The join is what makes re-embedding correct. Matching on document_id
    alone would treat an amended alert as already done and keep serving the
    superseded text; ignoring model_name would treat vectors from a different
    model as interchangeable with this one's."""
    fake_db.queue([{"id": "a", "narrative_text": "x", "content_hash": "h"}])

    repository.fetch_unembedded_documents(model_name="test-model", limit=10)
    sql, params = fake_db.connection.find("left join", "weather_embeddings")

    assert "e.document_id = d.id" in sql.lower()
    assert "e.content_hash = d.content_hash" in sql.lower()
    assert "e.model_name = %s" in sql.lower()
    assert "where e.id is null" in sql.lower()
    assert params[0] == "test-model"
    assert params[-1] == 10


def test_fetch_unembedded_returns_rows(fake_db):
    fake_db.queue([{"id": "a", "narrative_text": "x", "content_hash": "h"}])
    rows = repository.fetch_unembedded_documents(model_name="m", limit=5)
    assert rows[0]["id"] == "a"


def test_fetch_unembedded_selects_the_text_it_must_embed(fake_db):
    fake_db.queue([])
    repository.fetch_unembedded_documents(model_name="m")
    sql, _ = fake_db.connection.find("left join")
    for column in ("narrative_text", "content_hash", "source_type"):
        assert column in sql.lower()


# --------------------------------------------------------------------------
# R14, R15 - writing vectors
# --------------------------------------------------------------------------


def make_chunks(document_id="a", count=2, content_hash="h"):
    return [
        {
            "document_id": document_id,
            "chunk_index": index,
            "chunk_text": f"chunk {index}",
            "embedding": [0.01 * index] * repository.VECTOR_DIM,
            "content_hash": content_hash,
        }
        for index in range(count)
    ]


def test_embeddings_are_cast_to_vector_on_insert(fake_db):
    """Written as real pgvector values on the first insert. The reference
    implementation stores double precision[] and requires a manual
    "UPDATE ... SET embedding = embedding::vector" afterwards - which leaves
    the table unqueryable, and the HNSW index unusable, in between."""
    repository.replace_document_embeddings("a", make_chunks(), model_name="m")

    assert "::vector" in fake_db.last_write["template"]
    assert "double precision[]" not in fake_db.last_write["template"]


def test_embedding_values_are_serialized_as_vector_literals(fake_db):
    repository.replace_document_embeddings("a", make_chunks(count=1), model_name="m")
    row = fake_db.last_write["rows"][0]
    assert any(isinstance(value, str) and value.startswith("[") for value in row)


def test_replace_deletes_existing_chunks_before_inserting(fake_db):
    """Amended text can produce FEWER chunks than the original. Upserting on
    (document_id, chunk_index) alone would leave the surplus high-index chunks
    behind, still indexed and still answering searches with deleted text."""
    repository.replace_document_embeddings("a", make_chunks(count=1), model_name="m")

    statements = [sql.lower() for sql in fake_db.connection.statements]
    delete_position = next(i for i, s in enumerate(statements) if s.startswith("delete"))
    insert_position = next(i for i, s in enumerate(statements) if "insert into weather_embeddings" in s)
    assert delete_position < insert_position

    delete_sql, delete_params = fake_db.connection.find("delete from weather_embeddings")
    assert "document_id = %s" in delete_sql.lower()
    assert delete_params[0] == "a"


def test_replace_with_no_chunks_still_clears_old_vectors(fake_db):
    """A document whose narrative became empty must not keep serving its old
    chunks."""
    repository.replace_document_embeddings("a", [], model_name="m")
    fake_db.connection.find("delete from weather_embeddings")
    assert fake_db.execute_values_calls == []


def test_embedding_row_ids_are_derived_from_document_and_chunk(fake_db):
    repository.replace_document_embeddings("doc-1", make_chunks(count=2), model_name="m")
    ids = [row[0] for row in fake_db.last_write["rows"]]
    assert ids == ["doc-1::0", "doc-1::1"]


def test_embedding_rows_carry_the_model_name(fake_db):
    repository.replace_document_embeddings("a", make_chunks(count=1), model_name="my-model")
    assert "my-model" in fake_db.last_write["rows"][0]


# --------------------------------------------------------------------------
# R18, R20, S4 - retrieval
# --------------------------------------------------------------------------


def search_row(**overrides):
    row = {
        "document_id": "urn:oid:1",
        "location": "Cook County",
        "headline": "Flash Flood Warning issued...",
        "event": "Flash Flood Warning",
        "source_type": "alert",
        "severity": "Severe",
        "chunk_index": 0,
        "chunk_text": "Rivers are rising fast.",
        "narrative_text": "Rivers are rising fast.",
        "effective_at": None,
        "expires_at": None,
        "similarity": 0.82,
    }
    row.update(overrides)
    return row


def test_search_orders_by_cosine_distance(fake_db):
    """ORDER BY must be the distance operator ascending, not similarity
    descending. Only the operator form can be answered by the HNSW index -
    "ORDER BY 1 - (...) DESC" is an expression the planner cannot match to it,
    so it silently degrades to a full scan."""
    fake_db.queue([search_row()])
    repository.search([0.1] * repository.VECTOR_DIM, top_k=5)
    sql, _ = fake_db.connection.find("from weather_embeddings")

    assert "order by e.embedding <=> %s::vector" in sql.lower()
    assert "order by similarity desc" not in sql.lower()


def test_search_returns_similarity_not_distance(fake_db):
    """<=> yields cosine distance in 0..2. The API contract is a similarity
    score where higher is better, so it is reported as 1 - distance."""
    fake_db.queue([search_row()])
    sql, _ = None, None
    results = repository.search([0.1] * repository.VECTOR_DIM, top_k=1)
    sql, _ = fake_db.connection.find("from weather_embeddings")

    assert "1 - (e.embedding <=> %s::vector) as similarity" in sql.lower()
    assert results[0]["similarity"] == 0.82


def test_search_joins_documents_for_context(fake_db):
    fake_db.queue([search_row()])
    repository.search([0.1] * repository.VECTOR_DIM)
    sql, _ = fake_db.connection.find("from weather_embeddings")

    assert "join weather_documents d on d.id = e.document_id" in sql.lower()
    for column in ("location", "headline", "chunk_text"):
        assert column in sql.lower()


def test_search_passes_top_k_as_the_limit(fake_db):
    fake_db.queue([search_row()])
    repository.search([0.1] * repository.VECTOR_DIM, top_k=7)
    _, params = fake_db.connection.find("from weather_embeddings")
    assert params[-1] == 7


def test_search_without_a_filter_adds_no_where_clause(fake_db):
    """Building the filter into the SQL only when asked keeps the unfiltered
    query - the common one - free of a predicate the planner has to reason
    about alongside the index scan."""
    fake_db.queue([search_row()])
    repository.search([0.1] * repository.VECTOR_DIM)
    sql, _ = fake_db.connection.find("from weather_embeddings")
    assert "where" not in sql.lower()


def test_search_filters_by_source_type(fake_db):
    fake_db.queue([search_row()])
    repository.search([0.1] * repository.VECTOR_DIM, source_type="forecast")
    sql, params = fake_db.connection.find("from weather_embeddings")

    assert "where d.source_type = %s" in sql.lower()
    assert "forecast" in params


def test_search_rejects_an_unknown_source_type(fake_db):
    with pytest.raises(ValueError, match="source_type"):
        repository.search([0.1] * repository.VECTOR_DIM, source_type="hurricane")


def test_search_on_an_empty_index_returns_nothing_rather_than_raising(fake_db):
    """R21 - nothing synced yet is a normal state, not a failure."""
    fake_db.queue([])
    assert repository.search([0.1] * repository.VECTOR_DIM) == []


def test_search_binds_the_same_vector_to_both_placeholders(fake_db):
    """The vector appears twice - once in the similarity projection, once in
    ORDER BY. Binding two different values would rank by one vector and score
    by another, producing results that look sorted wrongly."""
    fake_db.queue([search_row()])
    vector = [0.5] * repository.VECTOR_DIM
    repository.search(vector, top_k=3)
    _, params = fake_db.connection.find("from weather_embeddings")

    literals = [p for p in params if isinstance(p, str) and p.startswith("[")]
    assert len(literals) == 2
    assert literals[0] == literals[1]


# --------------------------------------------------------------------------
# schema verification
# --------------------------------------------------------------------------


def test_verify_schema_accepts_a_matching_vector_width(fake_db):
    fake_db.queue([{"column_type": "vector(384)"}])
    repository.verify_schema()  # must not raise


def test_verify_schema_rejects_a_mismatched_vector_width(fake_db):
    """Swapping the embedding model without migrating the column fails on the
    first insert, deep inside a batch. Checking at startup names the cause."""
    fake_db.queue([{"column_type": "vector(768)"}])
    with pytest.raises(repository.SchemaMismatchError, match="768"):
        repository.verify_schema()


def test_verify_schema_reports_a_missing_table(fake_db):
    fake_db.queue([])
    with pytest.raises(repository.SchemaMismatchError, match="schema.sql"):
        repository.verify_schema()

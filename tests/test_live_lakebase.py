"""Integration tests that execute the real SQL against a real Lakebase.

Marked `live` and skipped unless LAKEBASE_URL is set, so the default suite
stays hermetic and fast.

**Why these exist.** The unit tests verify how SQL is *built* - joins, casts,
conflict targets, parameter order - using a fake cursor that records
statements without parsing them. That catches a great deal, and it cannot
catch a statement that is not valid Postgres.

It missed exactly that, once. A `\\n` inside a Python string ended a `--` SQL
comment early, so the remainder of the comment became bare SQL. All unit tests
passed; the first real query failed with `syntax error at or near "ozone"`.
These tests close that gap by running every statement the application issues.

    pytest -m live
"""

from __future__ import annotations

import os

import pytest

import lakebase
import repository

pytestmark = pytest.mark.live

LIVE = bool(os.environ.get("LAKEBASE_URL"))
requires_lakebase = pytest.mark.skipif(
    not LIVE, reason="set LAKEBASE_URL to run integration tests"
)


@pytest.fixture(scope="module")
def query_vector() -> list[float]:
    """A well-formed vector. Deliberately not from the model - these tests are
    about SQL validity, and loading 90MB of weights to produce 384 floats
    would make them slow for no gain."""
    return [0.01] * repository.VECTOR_DIM


@requires_lakebase
def test_connection_works():
    assert lakebase.healthcheck() is True


@requires_lakebase
def test_pgvector_is_installed():
    row = lakebase.run_query_one(
        "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
    )
    assert row, "the vector extension is not installed - apply schema.sql"


@requires_lakebase
def test_schema_matches_the_embedding_width():
    repository.verify_schema()


@requires_lakebase
def test_hnsw_index_exists_with_cosine_ops():
    """An index built for a different distance function is silently ignored by
    the planner, which looks exactly like the index not helping."""
    row = lakebase.run_query_one(
        """
        SELECT indexdef FROM pg_indexes
        WHERE tablename = %s AND indexname = 'idx_weather_embeddings_hnsw'
        """,
        (repository.EMBEDDINGS_TABLE,),
    )
    assert row, "HNSW index missing"
    assert "vector_cosine_ops" in row["indexdef"]
    assert "hnsw" in row["indexdef"].lower()


# ---------------------------------------------------------------------------
# Every statement the application issues, actually executed
# ---------------------------------------------------------------------------


@requires_lakebase
def test_search_sql_is_valid_postgres(query_vector):
    """The regression test for the broken-comment bug. Correctness of the
    ranking is not the point here - parsing and executing is."""
    results = repository.search(query_vector, top_k=5)
    assert isinstance(results, list)


@requires_lakebase
@pytest.mark.parametrize("source_type", [None, "alert", "forecast"])
def test_search_sql_is_valid_with_every_filter(query_vector, source_type):
    """The filtered branch builds different SQL, so it needs its own execution."""
    repository.search(query_vector, top_k=3, source_type=source_type)


@requires_lakebase
def test_search_returns_the_contracted_columns(query_vector):
    results = repository.search(query_vector, top_k=1)
    if not results:
        pytest.skip("no embeddings stored yet")

    for column in (
        "document_id", "location", "headline", "event", "source_type",
        "chunk_text", "chunk_index", "similarity",
    ):
        assert column in results[0], f"search must return {column}"


@requires_lakebase
def test_search_results_are_distinct(query_vector):
    """The dedup pipeline collapses repeated county-zone advisories and
    multiple chunks of one document. Verified against real data, because the
    duplication being defended against is a property of the corpus."""
    import re

    results = repository.search(query_vector, top_k=10)
    if len(results) < 2:
        pytest.skip("not enough embeddings stored to test deduplication")

    documents = [row["document_id"] for row in results]
    assert len(documents) == len(set(documents)), "a document appeared twice"

    texts = [re.sub(r"\s", "", row["chunk_text"]) for row in results]
    assert len(texts) == len(set(texts)), "the same text appeared twice"


@requires_lakebase
def test_similarity_is_within_range(query_vector):
    results = repository.search(query_vector, top_k=5)
    for row in results:
        assert -1.0001 <= float(row["similarity"]) <= 1.0001


@requires_lakebase
def test_results_are_ordered_by_descending_similarity(query_vector):
    results = repository.search(query_vector, top_k=10)
    scores = [float(row["similarity"]) for row in results]
    assert scores == sorted(scores, reverse=True)


@requires_lakebase
def test_fetch_unembedded_sql_is_valid():
    rows = repository.fetch_unembedded_documents(model_name="test-model", limit=1)
    assert isinstance(rows, list)


@requires_lakebase
def test_stats_sql_is_valid():
    stats = repository.stats()
    for key in ("documents", "alerts", "forecasts", "embeddings", "embedded_documents"):
        assert key in stats
        assert isinstance(stats[key], int)


@requires_lakebase
def test_count_helpers_are_valid():
    assert repository.count_documents() >= 0
    assert repository.count_documents(source_type="alert") >= 0
    assert repository.count_embeddings() >= 0


@requires_lakebase
def test_the_application_role_needs_no_ddl_rights():
    """Applying the schema is an administrator's job, not the app's.

    `CREATE INDEX IF NOT EXISTS` checks ownership *before* it checks
    existence, so a role that does not own the tables cannot run schema.sql
    even when every object already exists. That is the correct boundary: the
    application reads and writes rows and never alters structure, so a bug in
    it cannot drop a table.

    Asserted rather than assumed, because the alternative - granting the app
    ownership so a convenience call stops failing - is a real temptation and a
    bad trade.
    """
    import psycopg2

    with pytest.raises(psycopg2.errors.InsufficientPrivilege):
        lakebase.apply_schema()


@requires_lakebase
def test_the_application_role_can_do_its_actual_job():
    """The flip side: no DDL, but full read/write on the rows it owns."""
    document_id = "_test_rw_probe"
    repository.upsert_documents(
        [
            {
                "id": document_id,
                "location": "Test County",
                "source_type": "forecast",
                "narrative_text": "read write probe",
                "payload": {},
                "content_hash": "probe-hash",
            }
        ]
    )
    row = lakebase.run_query_one(
        "SELECT narrative_text FROM weather_documents WHERE id = %s", (document_id,)
    )
    assert row["narrative_text"] == "read write probe"
    lakebase.run_write("DELETE FROM weather_documents WHERE id = %s", (document_id,))


@requires_lakebase
def test_source_type_check_constraint_is_enforced():
    """The CHECK is the last line of defence against a typo'd source_type
    reaching the table and quietly never matching a filter again."""
    import psycopg2

    with pytest.raises(psycopg2.errors.CheckViolation):
        repository.upsert_documents(
            [
                {
                    "id": "_test_bad_source_type",
                    "location": "Test",
                    "source_type": "hurricane",  # not in ('alert','forecast')
                    "narrative_text": "test",
                    "payload": {},
                    "content_hash": "x",
                }
            ]
        )


@requires_lakebase
def test_deleting_a_document_cascades_to_its_vectors():
    """Purging expired documents must not strand orphan vectors that still
    answer searches."""
    document_id = "_test_cascade_doc"
    repository.upsert_documents(
        [
            {
                "id": document_id,
                "location": "Test County",
                "source_type": "alert",
                "narrative_text": "cascade test narrative",
                "payload": {},
                "content_hash": "cascade-hash",
            }
        ]
    )
    repository.replace_document_embeddings(
        document_id,
        [
            {
                "document_id": document_id,
                "chunk_index": 0,
                "chunk_text": "cascade test chunk",
                "embedding": [0.02] * repository.VECTOR_DIM,
                "content_hash": "cascade-hash",
            }
        ],
        model_name="test-model",
    )

    before = lakebase.run_query_one(
        "SELECT COUNT(*) AS n FROM weather_embeddings WHERE document_id = %s",
        (document_id,),
    )
    assert before["n"] == 1

    lakebase.run_write("DELETE FROM weather_documents WHERE id = %s", (document_id,))

    after = lakebase.run_query_one(
        "SELECT COUNT(*) AS n FROM weather_embeddings WHERE document_id = %s",
        (document_id,),
    )
    assert after["n"] == 0, "vectors outlived their document"

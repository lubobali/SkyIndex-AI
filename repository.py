"""Data access for weather_documents and weather_embeddings.

All SQL lives here rather than in the Flask routes or the ingestion script, so
there is exactly one definition of how a document is upserted and one
definition of how retrieval is ranked.

On the write path: embeddings are inserted as real pgvector values, cast with
an explicit ``%s::vector`` in the execute_values template. No Spark, and no
intermediate ``double precision[]`` column needing a manual cast afterwards.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Iterable, Sequence

from psycopg2.extras import execute_values

import lakebase
from weather_client import SOURCE_TYPES, WeatherDocument

logger = logging.getLogger(__name__)

DOCUMENTS_TABLE = "weather_documents"
EMBEDDINGS_TABLE = "weather_embeddings"

# Must match VECTOR(n) in schema.sql and the embedding model's output width.
# verify_schema() checks the database agrees before any batch job starts.
VECTOR_DIM = 384

_DOCUMENT_COLUMNS = (
    "id", "location", "latitude", "longitude", "source_type", "event",
    "headline", "narrative_text", "issued_at", "effective_at", "expires_at",
    "severity", "payload", "content_hash",
)


class SchemaMismatchError(RuntimeError):
    """The database schema does not match what this code expects."""


def to_vector_literal(values: Sequence[float]) -> str:
    """Render an embedding in pgvector's text input format.

    pgvector parses '[1,2,3]'. A Postgres array literal '{1,2,3}' is a
    different type entirely and will not cast to vector, so the bracket form
    is not a stylistic choice.
    """
    if len(values) != VECTOR_DIM:
        raise ValueError(
            f"Expected a {VECTOR_DIM}-dimensional embedding, got {len(values)}. "
            "The embedding model and the VECTOR(n) column must agree."
        )
    return "[" + ",".join(str(float(value)) for value in values) + "]"


def _as_row(document: WeatherDocument | dict) -> dict:
    return document.as_row() if isinstance(document, WeatherDocument) else dict(document)


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


def upsert_documents(documents: Iterable[WeatherDocument | dict]) -> int:
    """Insert or refresh weather documents. Returns the number written.

    Duplicate ids within a single batch are collapsed, last one winning.
    Postgres rejects a statement that touches the same conflict target twice
    ("ON CONFLICT DO UPDATE command cannot affect row a second time"), so a
    repeated id in the input would otherwise fail the whole batch.
    """
    deduplicated: dict[str, dict] = {}
    for document in documents:
        row = _as_row(document)
        deduplicated[row["id"]] = row

    if not deduplicated:
        return 0

    values = [
        (
            row["id"],
            row["location"],
            row.get("latitude"),
            row.get("longitude"),
            row["source_type"],
            row.get("event"),
            row.get("headline"),
            row["narrative_text"],
            row.get("issued_at"),
            row.get("effective_at"),
            row.get("expires_at"),
            row.get("severity"),
            json.dumps(row.get("payload") or {}),
            row["content_hash"],
        )
        for row in deduplicated.values()
    ]

    updatable = [column for column in _DOCUMENT_COLUMNS if column != "id"]
    assignments = ", ".join(f"{column} = EXCLUDED.{column}" for column in updatable)

    sql = f"""
        INSERT INTO {DOCUMENTS_TABLE} ({", ".join(_DOCUMENT_COLUMNS)}, synced_at)
        VALUES %s
        ON CONFLICT (id) DO UPDATE SET {assignments}, synced_at = now()
    """
    template = "(" + ", ".join(["%s"] * 12) + ", %s::jsonb, %s, now())"

    with lakebase.get_connection() as conn:
        with conn.cursor() as cur:
            execute_values(cur, sql, values, template=template, page_size=200)

    logger.info("Upserted %s weather documents", len(values))
    return len(values)


def count_documents(source_type: str | None = None) -> int:
    sql = f"SELECT COUNT(*) AS n FROM {DOCUMENTS_TABLE}"
    params: tuple = ()
    if source_type:
        sql += " WHERE source_type = %s"
        params = (source_type,)
    row = lakebase.run_query_one(sql, params)
    return int(row["n"]) if row else 0


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


def fetch_unembedded_documents(model_name: str, limit: int = 500) -> list[dict]:
    """Documents with no current vectors for this model.

    The anti-join matches on document_id AND content_hash AND model_name, and
    each conjunct earns its place:

      document_id  - the obvious one.
      content_hash - NWS amends an alert in place, keeping its id. Without
                     this, an amended document counts as already embedded and
                     retrieval keeps serving the superseded text forever.
      model_name   - vectors from a different model are not interchangeable;
                     they are not even in the same space.
    """
    sql = f"""
        SELECT d.id,
               d.location,
               d.headline,
               d.event,
               d.source_type,
               d.narrative_text,
               d.content_hash
        FROM {DOCUMENTS_TABLE} d
        LEFT JOIN {EMBEDDINGS_TABLE} e
               ON e.document_id = d.id
              AND e.content_hash = d.content_hash
              AND e.model_name = %s
        WHERE e.id IS NULL
        ORDER BY d.synced_at DESC
        LIMIT %s
    """
    return lakebase.run_query(sql, (model_name, limit))


def replace_document_embeddings(
    document_id: str, chunks: Sequence[dict], model_name: str
) -> int:
    """Replace all vectors for one document, atomically.

    Delete-then-insert rather than upsert on (document_id, chunk_index):
    amended text can produce FEWER chunks than the original, and an upsert
    would leave the surplus high-index chunks in place - still indexed, still
    matching queries, still returning text that no longer exists upstream.

    Both statements share one connection, so the pool's context manager commits
    them together or rolls both back.
    """
    with lakebase.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {EMBEDDINGS_TABLE} WHERE document_id = %s", (document_id,)
            )

            if not chunks:
                return 0

            values = [
                (
                    f"{document_id}::{chunk['chunk_index']}",
                    document_id,
                    int(chunk["chunk_index"]),
                    chunk["chunk_text"],
                    to_vector_literal(chunk["embedding"]),
                    model_name,
                    chunk["content_hash"],
                )
                for chunk in chunks
            ]

            execute_values(
                cur,
                f"""
                INSERT INTO {EMBEDDINGS_TABLE}
                    (id, document_id, chunk_index, chunk_text, embedding,
                     model_name, content_hash)
                VALUES %s
                """,
                values,
                template="(%s, %s, %s, %s, %s::vector, %s, %s)",
                page_size=100,
            )

    return len(chunks)


def count_embeddings() -> int:
    row = lakebase.run_query_one(f"SELECT COUNT(*) AS n FROM {EMBEDDINGS_TABLE}")
    return int(row["n"]) if row else 0


def clear_embeddings() -> int:
    """Delete every vector, forcing a full re-embed on the next run.

    Needed because content_hash tracks the *document text*, not the chunking
    configuration. Change CHUNK_SIZE, CHUNK_OVERLAP, or the boundary rules and
    every stored vector is stale in a way the anti-join cannot detect: the
    documents are unchanged, so they look embedded.

    Documents are untouched - only the derived vectors go, and the ingestion
    script rebuilds them.
    """
    with lakebase.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f"DELETE FROM {EMBEDDINGS_TABLE}")
            deleted = cur.rowcount
    logger.info("Cleared %s embeddings", deleted)
    return deleted


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def search(
    embedding: Sequence[float], top_k: int = 5, source_type: str | None = None
) -> list[dict]:
    """Rank stored chunks by cosine similarity to a query embedding.

    ORDER BY is the distance operator ascending, not similarity descending.
    Only the bare operator form can be answered by the HNSW index; wrapping it
    in an expression like "ORDER BY 1 - (...) DESC" produces something the
    planner cannot match to the index, and the query silently degrades to a
    full scan that still returns correct results - the worst kind of
    regression, because nothing looks broken.
    """
    if source_type is not None and source_type not in SOURCE_TYPES:
        raise ValueError(
            f"Unknown source_type {source_type!r}, expected one of {SOURCE_TYPES}"
        )

    vector = to_vector_literal(embedding)

    where_clause = ""
    params: list[Any] = [vector]
    if source_type:
        where_clause = "WHERE d.source_type = %s"
        params.append(source_type)
    params.extend([vector, top_k])

    sql = f"""
        SELECT d.id AS document_id,
               d.location,
               d.headline,
               d.event,
               d.source_type,
               d.severity,
               d.effective_at,
               d.expires_at,
               d.narrative_text,
               e.chunk_index,
               e.chunk_text,
               1 - (e.embedding <=> %s::vector) AS similarity
        FROM {EMBEDDINGS_TABLE} e
        JOIN {DOCUMENTS_TABLE} d ON d.id = e.document_id
        {where_clause}
        ORDER BY e.embedding <=> %s::vector
        LIMIT %s
    """
    return lakebase.run_query(sql, tuple(params))


# ---------------------------------------------------------------------------
# Health and schema
# ---------------------------------------------------------------------------


def verify_schema() -> None:
    """Check the stored vector width matches the model this code embeds with.

    Without this, changing the embedding model fails at the first insert -
    partway through a batch job, with a driver-level type error that names
    neither the model nor the column.
    """
    row = lakebase.run_query_one(
        """
        SELECT format_type(a.atttypid, a.atttypmod) AS column_type
        FROM pg_attribute a
        WHERE a.attrelid = %s::regclass
          AND a.attname = 'embedding'
          AND NOT a.attisdropped
        """,
        (EMBEDDINGS_TABLE,),
    )

    if not row:
        raise SchemaMismatchError(
            f"{EMBEDDINGS_TABLE}.embedding not found. Apply schema.sql first."
        )

    column_type = str(row["column_type"])
    if column_type != f"vector({VECTOR_DIM})":
        raise SchemaMismatchError(
            f"{EMBEDDINGS_TABLE}.embedding is {column_type}, but this code "
            f"embeds at {VECTOR_DIM} dimensions. Re-apply schema.sql and "
            "re-embed, or change the model back."
        )


def stats() -> dict:
    """Counts for the health endpoint and the UI."""
    row = lakebase.run_query_one(
        f"""
        SELECT (SELECT COUNT(*) FROM {DOCUMENTS_TABLE}) AS documents,
               (SELECT COUNT(*) FROM {DOCUMENTS_TABLE} WHERE source_type = 'alert') AS alerts,
               (SELECT COUNT(*) FROM {DOCUMENTS_TABLE} WHERE source_type = 'forecast') AS forecasts,
               (SELECT COUNT(*) FROM {EMBEDDINGS_TABLE}) AS embeddings,
               (SELECT COUNT(DISTINCT document_id) FROM {EMBEDDINGS_TABLE}) AS embedded_documents
        """
    )
    return {key: int(value) for key, value in (row or {}).items()}


__all__ = [
    "DOCUMENTS_TABLE",
    "EMBEDDINGS_TABLE",
    "SchemaMismatchError",
    "VECTOR_DIM",
    "count_documents",
    "count_embeddings",
    "fetch_unembedded_documents",
    "replace_document_embeddings",
    "search",
    "stats",
    "to_vector_literal",
    "upsert_documents",
    "verify_schema",
]

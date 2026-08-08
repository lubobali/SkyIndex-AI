-- SkyIndex-AI schema for Lakebase (Databricks-managed Postgres + pgvector).
--
-- Idempotent by construction: every statement is IF NOT EXISTS, so applying
-- this file to an existing database is a no-op rather than an error. It is
-- executed as a single script by lakebase.apply_schema().
--
-- The vector width (384) is tied to sentence-transformers/all-MiniLM-L6-v2.
-- Changing the embedding model means changing this number and re-embedding
-- everything - repository.verify_schema() checks the two agree at startup so
-- a mismatch surfaces immediately instead of as an insert failure halfway
-- through a batch job.

CREATE EXTENSION IF NOT EXISTS vector;


-- ---------------------------------------------------------------------------
-- weather_documents - the raw document store
--
-- One row per retrievable weather narrative, from either source. payload keeps
-- the untouched API object so any field not promoted to a column can still be
-- recovered without re-fetching (alerts expire and disappear from the API).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS weather_documents (
    id              TEXT PRIMARY KEY,
    location        TEXT NOT NULL,
    latitude        DOUBLE PRECISION,
    longitude       DOUBLE PRECISION,
    source_type     TEXT NOT NULL CHECK (source_type IN ('alert', 'forecast')),
    event           TEXT,
    headline        TEXT,
    narrative_text  TEXT NOT NULL,
    issued_at       TIMESTAMPTZ,
    effective_at    TIMESTAMPTZ,
    expires_at      TIMESTAMPTZ,
    severity        TEXT,
    payload         JSONB NOT NULL,
    -- sha256 of narrative_text. NWS re-issues an alert under its original id
    -- with amended wording, so "have we embedded this document" is not a
    -- question about the id alone - it is a question about this revision of it.
    content_hash    TEXT NOT NULL,
    synced_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_weather_documents_source_type
    ON weather_documents (source_type);

CREATE INDEX IF NOT EXISTS idx_weather_documents_expires_at
    ON weather_documents (expires_at);


-- ---------------------------------------------------------------------------
-- weather_embeddings - the vector store
--
-- One row per chunk. chunk_text is stored next to its vector rather than
-- recomputed from the document at query time: recomputing would make the
-- retrieved passage depend on the chunker's *current* settings, so tuning
-- CHUNK_SIZE later would silently change what already-stored vectors claim
-- to represent.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS weather_embeddings (
    id            TEXT PRIMARY KEY,
    document_id   TEXT NOT NULL
                  REFERENCES weather_documents (id) ON DELETE CASCADE,
    chunk_index   INTEGER NOT NULL,
    chunk_text    TEXT NOT NULL,
    embedding     VECTOR(384) NOT NULL,
    model_name    TEXT NOT NULL,
    -- Carried from the document revision this vector was produced from, so a
    -- re-sync can tell "already embedded" from "text has changed since".
    content_hash  TEXT NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (document_id, chunk_index)
);

-- HNSW rather than IVFFlat: it can be built on an empty table (IVFFlat needs
-- representative data present to cluster on, which a fresh deploy has none of)
-- and it does not need rebuilding as rows accumulate.
--
-- vector_cosine_ops matches the <=> operator used at query time. An index
-- built for a different distance function is silently ignored by the planner,
-- which looks exactly like the index not helping.
CREATE INDEX IF NOT EXISTS idx_weather_embeddings_hnsw
    ON weather_embeddings
    USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_weather_embeddings_document_id
    ON weather_embeddings (document_id);

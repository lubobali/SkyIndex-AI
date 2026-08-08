# Databricks notebook source
# MAGIC %md
# MAGIC # Ingest Weather Narrative -> Vector Embeddings (Lakebase)
# MAGIC
# MAGIC Reads documents from `weather_documents` that have no current vectors,
# MAGIC splits their narrative text into overlapping windows, embeds each window
# MAGIC with `sentence-transformers/all-MiniLM-L6-v2`, and writes the results into
# MAGIC `weather_embeddings` as real `vector(384)` values.
# MAGIC
# MAGIC **Runs two ways, same code path:**
# MAGIC
# MAGIC - As a Databricks notebook (widgets supply the config)
# MAGIC - As a plain CLI script: `python notebooks/ingest_weather_embeddings.py`
# MAGIC
# MAGIC **Two deliberate departures from the pattern this was built from:**
# MAGIC
# MAGIC 1. **No Spark on the write path.** Writes go through psycopg2's
# MAGIC    `execute_values`. `spark.write.jdbc` is not reliable against Lakebase
# MAGIC    and cannot express `ON CONFLICT` or write pgvector types.
# MAGIC 2. **Vectors are cast inline with `%s::vector`, on the first insert.**
# MAGIC    That pattern stores `double precision[]` and then asks the
# MAGIC    operator to run `UPDATE ... SET embedding = embedding::vector` by hand
# MAGIC    afterwards. That leaves the table unqueryable and the HNSW index
# MAGIC    unusable in between, and it is a manual step that can be forgotten.

# COMMAND ----------

# DBTITLE 1,Install dependencies (notebook only)
# MAGIC %pip install -q sentence-transformers psycopg2-binary

# COMMAND ----------

# dbutils.library.restartPython()

# COMMAND ----------

from __future__ import annotations

import logging
import os
import sys
import time

# When running as a notebook from the repo's Git folder, the project modules
# live one directory up. Harmless when running as a CLI script from the root.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger("ingest-weather-embeddings")

# COMMAND ----------

# DBTITLE 1,Config
#
# Widgets when running as a notebook, environment variables when running as a
# script. One resolution helper so the body below does not care which it is.


def _config(name: str, default: str, label: str = "") -> str:
    try:
        dbutils.widgets.text(name, default, label or name)  # type: ignore[name-defined]
        return dbutils.widgets.get(name)  # type: ignore[name-defined]
    except Exception:
        return os.environ.get(name.upper(), default)


BATCH_LIMIT = int(_config("batch_limit", "500", "Documents per run"))
ENCODE_BATCH_SIZE = int(_config("encode_batch_size", "32", "Texts per encode call"))

# Set to "true" after changing CHUNK_SIZE, CHUNK_OVERLAP, or the chunk boundary
# rules. content_hash fingerprints the document text, not the chunking config,
# so a chunker change leaves every stored vector stale in a way the incremental
# anti-join cannot see - the documents did not change, so they look done.
RESET = _config("reset", "false", "Delete all vectors and re-embed").lower() in ("1", "true", "yes")

# COMMAND ----------

import embeddings
import repository

MODEL_NAME = embeddings.EMBEDDING_MODEL

logger.info("model      : %s (%s-dim)", MODEL_NAME, embeddings.EMBEDDING_DIM)
logger.info("chunking   : size=%s overlap=%s", embeddings.CHUNK_SIZE, embeddings.CHUNK_OVERLAP)

# Fail before doing any work if the stored vector width disagrees with the
# model. Otherwise the mismatch surfaces as a driver type error partway
# through the first batch, naming neither the model nor the column.
repository.verify_schema()
logger.info("schema     : verified")
logger.info("before     : %s", repository.stats())

# COMMAND ----------

# DBTITLE 1,Chunk, embed, and write
#
# Per document rather than one giant batch: a document's vectors are replaced
# atomically, so a failure midway leaves earlier documents correctly embedded
# and later ones simply still pending. The next run picks up exactly where this
# one stopped, because "pending" is derived from the data, not from a cursor.

started = time.monotonic()

if RESET:
    logger.info("reset       : clearing %s existing vectors", repository.count_embeddings())
    repository.clear_embeddings()

pending = repository.fetch_unembedded_documents(model_name=MODEL_NAME, limit=BATCH_LIMIT)
logger.info("pending    : %s documents", len(pending))

documents_written = 0
chunks_written = 0
skipped = 0

for position, document in enumerate(pending, start=1):
    chunks = embeddings.chunk_document(document["narrative_text"], document["content_hash"])
    if not chunks:
        skipped += 1
        continue

    vectors = embeddings.embed_texts(
        [chunk["chunk_text"] for chunk in chunks], batch_size=ENCODE_BATCH_SIZE
    )
    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector
        chunk["document_id"] = document["id"]

    chunks_written += repository.replace_document_embeddings(
        document["id"], chunks, model_name=MODEL_NAME
    )
    documents_written += 1

    if position % 25 == 0 or position == len(pending):
        logger.info("  %s/%s documents, %s chunks", position, len(pending), chunks_written)

elapsed = time.monotonic() - started

# COMMAND ----------

# DBTITLE 1,Report
logger.info("-" * 60)
logger.info("documents embedded : %s", documents_written)
logger.info("chunks written     : %s", chunks_written)
logger.info("skipped (no text)  : %s", skipped)
logger.info("elapsed            : %.1fs", elapsed)
logger.info("after              : %s", repository.stats())

remaining = repository.fetch_unembedded_documents(model_name=MODEL_NAME, limit=1)
if remaining:
    logger.info(
        "NOTE: more documents remain pending - this run hit the %s batch limit. "
        "Run again to continue.", BATCH_LIMIT,
    )
else:
    logger.info("all documents are embedded at the current revision")

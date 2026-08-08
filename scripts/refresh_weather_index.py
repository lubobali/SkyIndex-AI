"""Scheduled refresh: harvest new NWS narrative, then embed whatever is new.

This is the job the Databricks Workflow runs. It exists as one script rather
than two tasks because the two halves are meaningless apart - syncing without
embedding leaves documents that no search can reach, and embedding without
syncing has nothing new to work on.

Alerts are the reason this runs on a schedule at all. A Flash Flood Warning is
issued, amended, and expires inside a few hours; an index refreshed daily would
mostly answer questions about weather that has already happened.

    python scripts/refresh_weather_index.py
    python scripts/refresh_weather_index.py --locations "Chicago, IL" "Miami, FL"
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import embeddings  # noqa: E402
import repository  # noqa: E402
from weather_client import NWSClient, SOURCE_TYPES  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger("refresh-weather-index")

# Quiet the model loader before it is imported. sentence-transformers logs a
# hub HTTP request per config file it resolves - about 20 lines - plus progress
# bars. In a scheduled job that buries the only output that matters.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
for _noisy in (
    "httpx", "httpcore", "urllib3", "filelock", "sentence_transformers",
    "huggingface_hub", "huggingface_hub.utils._http", "transformers",
):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

DEFAULT_LOCATIONS = [
    location.strip()
    for location in os.environ.get(
        "DEFAULT_LOCATIONS", "Chicago, IL;Austin, TX;Denver, CO;Miami, FL;Seattle, WA"
    ).split(";")
    if location.strip()
]


def refresh(locations: list[str], limit: int, source_types: list[str]) -> dict:
    started = time.monotonic()

    repository.verify_schema()

    client = NWSClient()
    documents = client.harvest(locations, limit=limit, source_types=source_types)
    synced = repository.upsert_documents(documents)
    logger.info("synced %s documents from %s locations", synced, len(locations))

    for error in client.errors:
        # Logged individually rather than counted, so a location that fails
        # every run is visible in the job output instead of hiding inside a
        # total that still looks healthy.
        logger.warning("location failed: %s", error)

    model_name = embeddings.EMBEDDING_MODEL
    pending = repository.fetch_unembedded_documents(model_name=model_name, limit=1000)
    logger.info("%s documents need embedding", len(pending))

    chunks_written = 0
    for document in pending:
        chunks = embeddings.chunk_document(
            document["narrative_text"], document["content_hash"]
        )
        if not chunks:
            continue

        vectors = embeddings.embed_texts([chunk["chunk_text"] for chunk in chunks])
        for chunk, vector in zip(chunks, vectors):
            chunk["embedding"] = vector
            chunk["document_id"] = document["id"]

        chunks_written += repository.replace_document_embeddings(
            document["id"], chunks, model_name=model_name
        )

    summary = {
        "synced": synced,
        "embedded_documents": len(pending),
        "chunks_written": chunks_written,
        "failed_locations": len(client.errors),
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "totals": repository.stats(),
    }
    logger.info("refresh complete: %s", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locations", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--source-types", nargs="*", default=list(SOURCE_TYPES))
    args = parser.parse_args()

    refresh(
        locations=args.locations or DEFAULT_LOCATIONS,
        limit=args.limit,
        source_types=args.source_types,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""SkyIndex-AI - semantic search over live National Weather Service text.

Endpoints
    GET  /                 search UI
    GET  /healthz          liveness plus Lakebase reachability and row counts
    POST /weather/sync     harvest NWS narrative into weather_documents
    POST /weather/search   semantic search over weather_embeddings
    GET  /weather/search   same, plus a generated natural-language summary
    GET  /weather/documents recently synced documents

Run locally:
    python app.py
Deploy as a Databricks App using app.yaml.
"""

from __future__ import annotations

import logging
import os
import threading

from flask import Flask, jsonify, render_template, request

import embeddings
import lakebase
import rag
import repository
import validation
from validation import BadRequest
from weather_client import NWSClient, SOURCE_TYPES

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("skyindex")

app = Flask(__name__)

DEFAULT_LOCATIONS = [
    location.strip()
    for location in os.environ.get(
        "DEFAULT_LOCATIONS", "Chicago, IL;Austin, TX;Denver, CO;Miami, FL;Seattle, WA"
    ).split(";")
    if location.strip()
]

# Summaries are opt-in per deployment: model serving is limited on some tiers,
# and a search endpoint should not depend on it being present.
SUMMARY_ENABLED = os.environ.get("SUMMARY_ENABLED", "true").lower() not in ("0", "false", "no")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


@app.errorhandler(BadRequest)
def handle_bad_request(err: BadRequest):
    """Client errors carry a message written to be shown to the caller."""
    return jsonify({"error": str(err)}), 400


@app.errorhandler(Exception)
def handle_exception(err: Exception):
    """Every unhandled error returns JSON, never an HTML error page.

    The UI parses every response as JSON. A Flask HTML error page would make
    the frontend fail while parsing, hiding the real error behind a
    SyntaxError about an unexpected '<'.
    """
    logger.exception("Unhandled exception while processing request")
    status = getattr(err, "code", 500)
    if not isinstance(status, int):
        status = 500
    return jsonify({"error": str(err)}), status


# ---------------------------------------------------------------------------
# Health and UI
# ---------------------------------------------------------------------------


def _health_payload() -> tuple[dict, int]:
    """Liveness, plus whether Lakebase is reachable and how much is indexed.

    Deliberately does NOT touch the embedding model. Loading it takes seconds,
    and a health check that waits on a model load reports "unhealthy" for a
    perfectly healthy app that simply has not served a search yet.
    """
    database_ok = lakebase.healthcheck()
    payload = {"status": "ok" if database_ok else "degraded", "database": database_ok}

    if database_ok:
        try:
            payload["counts"] = repository.stats()
        except Exception:
            logger.warning("Could not read stats", exc_info=True)

    return payload, (200 if database_ok else 503)


@app.route("/healthz")
def healthz():
    """Health check for platform probes."""
    payload, status = _health_payload()
    return jsonify(payload), status


@app.route("/api/stats")
def api_stats():
    """The same payload, on a path no hosting platform will claim.

    `/healthz` is a de facto reserved path - Cloud Run and several other
    platforms intercept it for their own liveness probing and never let the
    request reach the application. The UI asking for its counts there gets the
    platform's answer instead of ours, which surfaces as "stats unavailable"
    on a perfectly healthy app.

    Health checks keep /healthz because that is what probes look for. The UI
    reads from here.
    """
    payload, status = _health_payload()
    return jsonify(payload), status


@app.route("/")
def index():
    return render_template("index.html", locations=DEFAULT_LOCATIONS)


# ---------------------------------------------------------------------------
# Part 1 - harvest
# ---------------------------------------------------------------------------


@app.route("/weather/sync", methods=["POST"])
def sync_weather():
    """Harvest NWS narrative for a set of locations into Lakebase.

    Body (all optional):
        {"locations": ["Chicago, IL", "39.7,-104.9"],
         "limit": 50,
         "source_types": ["alert", "forecast"]}

    Upserts on the document id, so re-running refreshes rather than
    duplicating. Documents are stored here but not embedded - run the
    ingestion script for that. Splitting them keeps this endpoint fast and
    keeps the app free of the model on the request path.
    """
    body = request.get_json(silent=True) or {}

    locations = validation.clean_locations(body.get("locations"), DEFAULT_LOCATIONS)
    limit = validation.clean_sync_limit(body.get("limit"))
    source_types = validation.clean_source_types(body.get("source_types"))

    client = NWSClient()
    documents = client.harvest(locations, limit=limit, source_types=source_types)
    written = repository.upsert_documents(documents)

    counts = {source: 0 for source in SOURCE_TYPES}
    for document in documents:
        counts[document.source_type] = counts.get(document.source_type, 0) + 1

    response = {
        "synced": written,
        "by_source_type": counts,
        "locations": locations,
        "source_types": source_types,
    }
    # A location that failed is reported rather than silently missing from the
    # totals - "synced: 40" reads like success even when half the batch failed.
    if client.errors:
        response["errors"] = client.errors

    return jsonify(response)


@app.route("/weather/documents")
def list_documents():
    """Recently synced documents, for inspection and for the UI."""
    limit = validation.clean_sync_limit(request.args.get("limit"), default=25)
    source_type = validation.clean_source_type(request.args.get("source_type"))

    sql = """
        SELECT id, location, source_type, event, headline, severity,
               effective_at, expires_at, synced_at,
               length(narrative_text) AS narrative_chars
        FROM weather_documents
    """
    params: list = []
    if source_type:
        sql += " WHERE source_type = %s"
        params.append(source_type)
    sql += " ORDER BY synced_at DESC, effective_at DESC NULLS LAST LIMIT %s"
    params.append(limit)

    return jsonify(lakebase.run_query(sql, tuple(params)))


# ---------------------------------------------------------------------------
# Part 3 - retrieval
# ---------------------------------------------------------------------------


def _run_search(query: str, top_k: int, source_type: str | None) -> list[dict]:
    """Embed the query and rank stored chunks against it."""
    vector = embeddings.embed_query(query)
    return repository.search(vector, top_k=top_k, source_type=source_type)


def _serialize(results: list[dict]) -> list[dict]:
    """Shape rows for the API response.

    Timestamps become strings here rather than relying on the JSON encoder, so
    the contract does not change if the driver's type mapping does.
    """
    payload = []
    for result in results:
        payload.append(
            {
                "document_id": result["document_id"],
                "location": result["location"],
                "headline": result.get("headline"),
                "event": result.get("event"),
                "source_type": result.get("source_type"),
                "severity": result.get("severity"),
                "chunk_index": result.get("chunk_index"),
                "chunk_text": result.get("chunk_text"),
                # The full document text alongside the matching chunk. The
                # chunk is what scored; the narrative is what it came from,
                # and a caller showing a passage usually wants to be able to
                # expand it without a second request.
                "narrative_text": result.get("narrative_text"),
                "similarity": round(float(result["similarity"]), 4),
                "effective_at": _isoformat(result.get("effective_at")),
                "expires_at": _isoformat(result.get("expires_at")),
            }
        )
    return payload


def _isoformat(value):
    return value.isoformat() if hasattr(value, "isoformat") else value


@app.route("/weather/search", methods=["POST"])
def search_weather():
    """Semantic search over the indexed weather narrative.

        {"query": "flash flood risk this weekend", "top_k": 5}

    Optional "source_type": "alert" or "forecast" to restrict the search.

    An empty index returns 200 with no results. Nothing synced yet is a normal
    state of a freshly deployed app, not a failure.
    """
    body = request.get_json(silent=True) or {}

    query = validation.clean_query(body.get("query"))
    top_k = validation.clean_top_k(body.get("top_k"))
    source_type = validation.clean_source_type(body.get("source_type"))

    results = _run_search(query, top_k, source_type)
    return jsonify(
        {
            "query": query,
            "top_k": top_k,
            "source_type": source_type,
            "count": len(results),
            "results": _serialize(results),
        }
    )


@app.route("/weather/search", methods=["GET"])
def search_weather_with_summary():
    """Search, plus a generated natural-language answer (basic RAG).

        GET /weather/search?query=flash+flood+risk&top_k=5

    The summary is additive: if the serving endpoint is unavailable, this
    returns the ranked results with "summary": null rather than failing.
    """
    query = validation.clean_query(request.args.get("query"))
    top_k = validation.clean_top_k(request.args.get("top_k"))
    source_type = validation.clean_source_type(request.args.get("source_type"))

    results = _run_search(query, top_k, source_type)
    serialized = _serialize(results)

    summary = None
    if SUMMARY_ENABLED and serialized:
        summary = rag.summarize(query, serialized)

    return jsonify(
        {
            "query": query,
            "top_k": top_k,
            "source_type": source_type,
            "count": len(serialized),
            "summary": summary,
            "summary_model": rag.SUMMARY_ENDPOINT if summary else None,
            "results": serialized,
        }
    )


def warm_encoder() -> None:
    """Load the embedding model in the background at startup.

    Without this the first search pays the full model load - about 1.4 seconds
    measured locally - and a demo's opening query is the one most likely to be
    watched. Loading in a daemon thread keeps /healthz answering immediately,
    and get_encoder() is guarded by a lock, so a search arriving mid-warmup
    waits for the same instance rather than starting a second load.
    """

    def _warm():
        try:
            embeddings.get_encoder()
            logger.info("Embedding model warm (%s)", embeddings.EMBEDDING_MODEL)
        except Exception:
            # A failed warmup must not stop the app: retrieval is broken either
            # way, but /healthz and the document endpoints still work, and the
            # real error surfaces on the first search with a proper traceback.
            logger.warning("Could not warm the embedding model", exc_info=True)

    threading.Thread(target=_warm, name="warm-encoder", daemon=True).start()


if os.environ.get("WARM_ENCODER", "true").lower() not in ("0", "false", "no"):
    warm_encoder()


if __name__ == "__main__":
    host = os.environ.get("FLASK_RUN_HOST", "0.0.0.0")
    port = int(os.environ.get("FLASK_RUN_PORT", "8000"))
    app.run(host=host, port=port, debug=os.environ.get("FLASK_DEBUG", "").lower() == "true")

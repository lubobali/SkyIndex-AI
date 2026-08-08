# SkyIndex-AI

**Semantic search over live National Weather Service text.**

[![tests](https://img.shields.io/badge/tests-196%20passing-4dd4a0)](#tests)
[![python](https://img.shields.io/badge/python-3.13-4da3ff)](#)
[![postgres](https://img.shields.io/badge/pgvector-0.8.0-4da3ff)](#)
[![licence](https://img.shields.io/badge/licence-MIT-8b9bb8)](LICENSE.txt)

The National Weather Service publishes a large, continuously changing body of
free text: watch and warning products, their protective-action instructions,
and multi-day forecast discussions. It is written for people, not machines.
There is no field you can filter on to answer *"is there a flash flood risk
near rivers this weekend"* — the answer is buried in prose.

SkyIndex-AI harvests that prose, embeds it, stores the vectors in Postgres with
pgvector, and serves it back as a retrieval API.

```
POST /weather/search  {"query": "flash flood risk this weekend"}
```

```json
{
  "count": 3,
  "results": [
    {
      "location": "El Paso, TX",
      "event": "Flash Flood Warning",
      "source_type": "alert",
      "chunk_text": "...that will experience flash flooding include East El Paso...",
      "similarity": 0.517
    }
  ]
}
```

---

## What it does that a keyword search cannot

Every one of these was answered from live data, with no keyword overlap doing
the work:

| Question asked | What came back | Score |
| --- | --- | --- |
| *"dangerous heat, what should I do"* | Heat Advisory — *"reschedule strenuous activities to early morning or evening"* | 0.664 |
| *"will it rain tomorrow morning"* | Forecast — *"a slight chance of showers and thunderstorms after midnight"* | 0.627 |
| *"flash flood risk this weekend"* | Flash Flood Warning, El Paso TX | 0.517 |

Note which source each question reached. Hazard questions surface **alerts**;
everyday questions surface **forecasts**. Nothing routes them — the two
document types live in one index and the embedding space separates them.

The second example is the one worth dwelling on. The forecast text never says
"rain". It says *"showers and thunderstorms"*. A keyword search returns nothing.

---

## Architecture

```
        api.weather.gov
   ┌──────────┴──────────┐
   │                     │
/alerts/active   /gridpoints/{office}/{x},{y}/forecast
   │                     │
   └──────────┬──────────┘
              │  weather_client.py
              │  · geocode → NWS gridpoint
              │  · normalize both shapes into one document
              │  · stable ids, content hashing
              ▼
     ┌───────────────────────┐
     │  weather_documents    │   raw narrative + JSONB provenance
     │  (Lakebase Postgres)  │   content_hash detects amendments
     └───────────┬───────────┘
                 │  ingest_weather_embeddings.py
                 │  · chunk 800/100, paragraph→sentence→word boundaries
                 │  · all-MiniLM-L6-v2, 384-dim
                 │  · psycopg2 execute_values, %s::vector
                 ▼
     ┌───────────────────────┐
     │  weather_embeddings   │   vector(384) + HNSW (vector_cosine_ops)
     └───────────┬───────────┘
                 │  repository.search()  ·  ORDER BY embedding <=> query
                 ▼
        Flask API + search UI
                 │
                 └── optional: generated answer over the retrieved passages
```

---

## Engineering decisions worth reading

**Vectors are written as vectors, on the first insert.** The pattern this was
built from stores embeddings as `double precision[]` and then asks the operator
to run `UPDATE ... SET embedding = embedding::vector` by hand afterwards. That
leaves the table unqueryable and the HNSW index unusable in the gap, and it is
a manual step that can be forgotten. Casting inline with `%s::vector` in the
`execute_values` template removes both the gap and the step.

**`content_hash` is what makes re-embedding correct.** NWS amends an alert in
place, under its original id. Deciding "has this been embedded" from the
document id alone would treat the amended alert as done and serve the
superseded text indefinitely. The anti-join matches on `document_id` **and**
`content_hash` **and** `model_name`, so an amendment is detected, a
model change invalidates everything, and unchanged text is skipped. A live
refresh synced 134 documents and re-embedded only the 29 that had actually
changed.

**Vectors are replaced, not upserted.** Amended text can produce *fewer* chunks
than the original. Upserting on `(document_id, chunk_index)` would leave the
surplus high-index chunks behind — still indexed, still matching queries, still
returning text that no longer exists upstream. Delete-then-insert inside one
transaction.

**`ORDER BY` is the bare distance operator.** Only that form can be answered by
the HNSW index. Writing the more readable `ORDER BY 1 - (...) DESC` produces an
expression the planner cannot match to the index, and the query silently
degrades to a full scan that still returns correct results — the worst kind of
regression, because nothing looks broken.

**Retrieval collapses the corpus's two kinds of repetition.** NWS issues the
same advisory separately for every county zone — one Colorado air quality alert
arrived 12 times, byte-identical, under 12 different ids. And a long alert
splits into several chunks that all score well on the same query. Untreated,
35% of stored chunks were redundant and a top-5 could be the same paragraph
five times. Retrieval over-fetches, then collapses to one result per document
and one per distinct text, so `top_k` means five different answers.

**Chunk boundaries are a presentation decision.** `chunk_text` is what a user
reads as the retrieved passage, so a window ending mid-clause is a visible
defect. The chunker prefers a paragraph break, then a sentence break, then a
word break, and only accepts one once the window is at least half full — so a
full stop 20 characters in cannot win and emit a near-empty chunk.

**Rounding coordinates saves a redirect.** `api.weather.gov` answers `301` for
`/points` coordinates carrying more than four decimal places, and `200` at
four. Rounding trades a few centimetres — far below the resolution of a 2.5 km
forecast grid square — for one fewer round trip per location.

---

## The HNSW benchmark

Full results: [`benchmarks/results/hnsw.md`](benchmarks/results/hnsw.md)

Two things make the naive version of this benchmark meaningless, and both are
addressed. **Client wall-clock measures the network** — a round trip to
us-west-2 is 70-80 ms, which swamps a sub-millisecond query — so every timing
comes from Postgres' own `EXPLAIN (ANALYZE)` execution time. And **a small
corpus cannot show an index winning**, so a scale sweep locates the crossover
instead of asserting it.

| vectors | HNSW p50 | seq scan p50 | speedup |
| --- | --- | --- | --- |
| 140 (real corpus) | 0.320 ms | 0.307 ms | 0.96x |
| 2,000 | 0.595 ms | 0.902 ms | 1.52x |
| 20,000 | 1.207 ms | 8.869 ms | **7.35x** |

At 140 vectors the planner declines the index, and it is right to — the table
fits in cache, so scanning costs less than walking a graph. An index the
planner refuses is not a broken index.

Latency alone would be a misleading result, because HNSW is approximate. So
recall is measured against exact search, and the knob that controls it is
swept:

| `hnsw.ef_search` | p50 | recall@5 | still faster than exact |
| --- | --- | --- | --- |
| 40 (default) | 1.188 ms | 0.180 | yes |
| 100 | 2.311 ms | 0.400 | yes |
| 200 | 3.711 ms | 0.660 | yes |
| 400 | 6.120 ms | 0.800 | yes |

Low recall from an HNSW index is a knob that has not been turned, not a defect.
Exact search on the same data costs 8.28 ms, so even at `ef_search=400` the
index is still ahead — with recall to spare on the real corpus, whose
embeddings have cluster structure that random test vectors deliberately lack.

---

## Tests

```
196 passing, 0.28s
```

| file | covers |
| --- | --- |
| `test_weather_client.py` | 53 — normalization, stable ids, pacing, retries, geocoding |
| `test_app.py` | 54 — endpoints, validation, error contracts, health |
| `test_repository.py` | 32 — SQL construction, casts, conflict targets, filters |
| `test_embeddings.py` | 37 — chunk boundaries, coverage, encoder lifecycle |
| `test_rag.py` | 20 — context building, every degradation path |

Only genuine external boundaries are mocked: outbound HTTP, the Postgres wire
protocol, the embedding model, the summarizer. Normalization, chunking, id
derivation, SQL construction and request handling all run for real.

The client fixtures are **recorded live responses** from `api.weather.gov`, not
hand-written dicts — so a passing test means the normalizer handles the shape
the API actually returns, including the fields nobody remembers, like an alert
whose `instruction` is null.

Two bugs came out of live testing that mocks could not have caught:

- `/alerts/active` rejects a `limit` parameter with a `400`. The failure is
  silent through a lenient parser: the error body has no `features` key, so
  naive code reports "no active alerts" for a state that has thirty.
- Coordinates with more than four decimal places get a `301` redirect.

---

## Running it

See [README_WEATHER.md](README_WEATHER.md) for setup, schema, and the full
sync → embed → search walkthrough.

```bash
pip install -r requirements.txt
export LAKEBASE_URL="postgresql://user:pass@host:5432/databricks_postgres?sslmode=require"
export NWS_USER_AGENT="(YourApp, you@example.com)"

python -c "import lakebase; lakebase.apply_schema()"
python app.py                                    # http://localhost:8000

curl -X POST localhost:8000/weather/sync -H 'Content-Type: application/json' \
     -d '{"locations": ["Chicago, IL", "Miami, FL"]}'
python notebooks/ingest_weather_embeddings.py
```

---

## Stack

Python 3.13 · Flask · psycopg2 · sentence-transformers (all-MiniLM-L6-v2, 384-dim)
· PostgreSQL 17 + pgvector 0.8.0 on Databricks Lakebase · HNSW / `vector_cosine_ops`
· data from [api.weather.gov](https://www.weather.gov/documentation/services-web-api) (public domain)

MIT licensed.

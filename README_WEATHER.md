# SkyIndex-AI — data source, schema, and how to run it

Technical companion to [README.md](README.md). Covers the source choice and why,
the schema decisions, and the end-to-end pipeline.

**Live app:** https://skyindex-ai-1352785079224954.aws.databricksapps.com
(Databricks workspace auth — screenshots in [README.md](README.md#it-running))
**Source:** https://github.com/lubobali/SkyIndex-AI

---

## 0. Where everything is

Every required behaviour, and where it lives.

### Part 1 — Harvest

| Required | Implemented in | Evidence |
| --- | --- | --- |
| Client module resolving locations to an NWS gridpoint via `GET /points/{lat},{lon}` | `weather_client.py::NWSClient.resolve` | screenshot 6 |
| Accepts city/state **and** lat/lon | `weather_client.py::parse_location` | `test_weather_client.py` (11 cases) |
| Fetches active alerts | `NWSClient.fetch_alerts` → `/alerts/active?area=` | screenshot 6 |
| Fetches forecast narrative | `NWSClient.fetch_forecast` → `/gridpoints/{office}/{x},{y}/forecast` | screenshot 6 |
| `id` — stable dedup key | alert URN; forecast `sha256(location\|start\|end)` | screenshot 8 |
| `location` | `weather_documents.location` | screenshot 8 |
| `source_type` — "alert" / "forecast" | `CHECK IN ('alert','forecast')` | screenshot 8 |
| `headline` / `event` | both stored as separate columns | screenshot 3 |
| `narrative_text` — the free text embedded | `weather_documents.narrative_text` | screenshot 3 |
| `issued_at` / `effective_at` | both, plus `expires_at` | screenshot 3 |
| `payload` — raw JSON for provenance | `JSONB NOT NULL` | screenshot 3 |
| `synced_at` | `TIMESTAMPTZ DEFAULT now()` | screenshot 3 |
| Written into `weather_documents` | `schema.sql`, `repository.upsert_documents` | screenshots 3, 8 |
| Same connection pattern as `lakebase.py` — `get_connection()` context manager, psycopg2 + `RealDictCursor` | `lakebase.py::get_connection` | — |
| `POST /weather/sync` with `{"locations": [...], "limit": 50}`, returns a count | `app.py::sync_weather` | screenshot 6 |

### Part 2 — Vectorize

| Required | Implemented in | Evidence |
| --- | --- | --- |
| Ingestion script, plain Python + psycopg2 | `notebooks/ingest_weather_embeddings.py` | screenshot 6 |
| **No `spark.write.jdbc`** | no Spark anywhere in the project | — |
| Reads unembedded rows via `get_connection()` | `repository.fetch_unembedded_documents` | screenshot 6 |
| Chunks `narrative_text`, 800 / 100 sliding window | `embeddings.chunk_text` | `test_embeddings.py` (17 cases) |
| Embeds with `sentence-transformers/all-MiniLM-L6-v2`, 384-dim | `embeddings.embed_texts` | screenshot 3 |
| `weather_embeddings.id` | `{document_id}::{chunk_index}` | screenshot 9 |
| `document_id` FK → `weather_documents.id` | `REFERENCES ... ON DELETE CASCADE` | screenshot 3 |
| `chunk_index` | `INTEGER NOT NULL` | screenshot 9 |
| `chunk_text` | `TEXT NOT NULL` | screenshot 9 |
| `embedding vector(384)` | real pgvector column | screenshots 3, 9 |
| `model_name` | `TEXT NOT NULL` | screenshot 3 |
| `created_at` | `TIMESTAMPTZ DEFAULT now()` | screenshot 3 |
| Written with `execute_values` | `repository.replace_document_embeddings` | — |
| Cast with `%s::vector` | in the `execute_values` template | `test_repository.py` |
| HNSW index `USING hnsw (embedding vector_cosine_ops)` | `schema.sql` | screenshot 3 |
| pgvector extension | `CREATE EXTENSION IF NOT EXISTS vector` | screenshot 3 |

### Part 3 — Retrieve

| Required | Implemented in | Evidence |
| --- | --- | --- |
| `POST /weather/search` with `{"query": ..., "top_k": 5}` | `app.py::search_weather` | screenshots 1, 2 |
| Query embedded with the **same** model | `embeddings.embed_query` | — |
| Model loaded once at module level, not per request | `embeddings.get_encoder`, warmed by `app.warm_encoder` | — |
| Cosine search via `<=>`, executed via psycopg2 | `repository.search` | screenshots 1, 2 |
| Returns `location`, `headline`, `chunk_text`, `similarity` | `app._serialize` (also `narrative_text`, `event`, `source_type`) | screenshot 2 |
| Edge case: empty `weather_embeddings` | 200 with `count: 0` | `test_app.py` |
| Edge case: malformed / missing query | 400 with a message | `test_app.py` (7 cases) |
| Edge case: `top_k` clamped 1–20 | `validation.clean_top_k` | `test_app.py` (9 cases) |

### Deliverables

| Required | File |
| --- | --- |
| `weather_client.py` | ✅ |
| Updated `app.py` with both endpoints | ✅ |
| DDL / migration for both tables | `schema.sql` + `lakebase.apply_schema()` |
| psycopg2 embedding ingestion script | `notebooks/ingest_weather_embeddings.py` |
| `README_WEATHER.md` — source + why, schema decisions, how to run, limitations | this file, sections 1–6 |

### Stretch goals — all five

| # | Goal | Implemented in | Evidence |
| --- | --- | --- | --- |
| 1 | `GET /weather/search?query=` with an LLM summary | `app.py::search_weather_with_summary`, `rag.py` | screenshot 1 |
| 2 | Upsert on `id` so re-sync does not duplicate | `repository.upsert_documents` — `ON CONFLICT (id) DO UPDATE` | screenshot 6 |
| 3 | Scheduled job re-syncing every N minutes | `resources/refresh_weather_index_job.yml` (30 min), `scripts/refresh_weather_index.py` | screenshot 6 |
| 4 | Two sources, retrieval filterable by `source_type` | alerts + forecasts; `repository.search(source_type=...)` | screenshot 2 |
| 5 | HNSW benchmark, with vs without the index | `benchmarks/hnsw_benchmark.py` | screenshot 7 |

Beyond spec: browser search UI (ARIA-labelled), pooled Lakebase connections,
221 tests, deployed on Databricks Apps.

---

## 1. Data source: the National Weather Service API

**Chosen:** `api.weather.gov`

**Why:**

- **No API key.** Zero auth plumbing, so the effort goes into the pipeline
  rather than into credential handling.
- **Genuinely unstructured.** An alert's `description` is 1000–2000 characters
  of `WHAT / WHERE / WHEN / IMPACTS` prose, and `instruction` is a separate
  block of protective-action text. Neither is reachable by any structured
  field. That is the property that makes this worth building: text where
  retrieval is the only way in.
- **Two document flavours from one source.** This satisfies the multi-source
  retrieval-filter goal without the incoherence of stitching two providers'
  schemas together:

  | `source_type` | endpoint | text embedded |
  | --- | --- | --- |
  | `alert` | `/alerts/active?area={state}` | `description` + `instruction` |
  | `forecast` | `/gridpoints/{office}/{x},{y}/forecast` | `detailedForecast` per period |

- **Live and always changing.** 329 alerts were active nationwide while this
  was being built, so there is always real data to demo against.
- **Public domain.** No licensing question about storing or serving it.

**Rejected:**

- **OpenWeatherMap** — needs a key, and its free text is a one-line
  `weather.description` ("light rain"). Not enough prose to embed meaningfully;
  the vectors would be near-duplicates of each other.
- **NOAA CPC discussion products** — excellent long-form text, but published as
  fixed-width plain-text files on a schedule rather than a JSON API. Parsing
  them is a text-munging exercise, not a pipeline exercise.

**One requirement of the API:** it rejects requests without a descriptive
`User-Agent` naming the application and a contact address. Set `NWS_USER_AGENT`.
`NWSClient` raises at construction if it is missing, rather than letting an
opaque `403` surface on the first call.

---

## 2. Schema

### `weather_documents`

| column | type | notes |
| --- | --- | --- |
| `id` | `TEXT PK` | stable dedup key, see below |
| `location` | `TEXT` | alert `areaDesc`, or the resolved "City, ST" |
| `latitude` / `longitude` | `DOUBLE PRECISION` | resolved gridpoint |
| `source_type` | `TEXT` | `CHECK IN ('alert','forecast')` |
| `event` | `TEXT` | "Flash Flood Warning" / "Tonight" |
| `headline` | `TEXT` | NWS headline, or forecast `shortForecast` |
| `narrative_text` | `TEXT NOT NULL` | **the text that gets embedded** |
| `issued_at` / `effective_at` / `expires_at` | `TIMESTAMPTZ` | |
| `severity` | `TEXT` | alerts only; forecasts carry no severity |
| `payload` | `JSONB NOT NULL` | untouched API object, for provenance |
| `content_hash` | `TEXT NOT NULL` | sha256 of `narrative_text` |
| `synced_at` | `TIMESTAMPTZ` | |

**Stable ids**, so re-running the sync updates rather than duplicates:

- **Alerts** use the NWS alert URN, which is already globally unique and
  versioned by the issuing office. Deriving our own would throw that away.
- **Forecasts** use `forecast:{office}/{x},{y}:{sha256(location|start|end)[:16]}`
  — keyed on the **time window**, never the period name. NWS relabels the same
  window as the day advances ("This Afternoon" is part of "Today" in an earlier
  issuance), so a name-keyed id would store the same forecast twice.

**`content_hash` is the column a naive design leaves out.** NWS
re-issues an alert under its original id with amended text. Keying "has this
been embedded" on the id alone would treat the amendment as done and serve the
superseded text forever. Hashing the narrative makes an amendment detectable
and an unchanged document skippable.

### `weather_embeddings`

| column | type | notes |
| --- | --- | --- |
| `id` | `TEXT PK` | `{document_id}::{chunk_index}` |
| `document_id` | `TEXT` | FK → `weather_documents(id)` `ON DELETE CASCADE` |
| `chunk_index` | `INTEGER` | |
| `chunk_text` | `TEXT` | the exact text that produced this vector |
| `embedding` | `VECTOR(384)` | |
| `model_name` | `TEXT` | |
| `content_hash` | `TEXT` | the document revision this vector came from |
| `created_at` | `TIMESTAMPTZ` | |
| | `UNIQUE (document_id, chunk_index)` | |
| | `INDEX USING hnsw (embedding vector_cosine_ops)` | |

`chunk_text` is stored beside its vector rather than recomputed at query time.
Recomputing would make the retrieved passage depend on the chunker's *current*
settings, so tuning `CHUNK_SIZE` later would silently change what
already-stored vectors claim to represent.

The FK cascades so purging expired documents cannot strand orphan vectors that
still answer searches.

**HNSW rather than IVFFlat:** it can be built on an empty table (IVFFlat needs
representative data to cluster on, which a fresh deploy has none of) and does
not need rebuilding as rows accumulate. `vector_cosine_ops` matches the `<=>`
operator used at query time — an index built for a different distance function
is silently ignored by the planner, which looks exactly like the index not
helping.

---

## 3. Chunking and the embedding model

| setting | value | why |
| --- | --- | --- |
| `CHUNK_SIZE` | 800 chars | ≈200 tokens, comfortably inside all-MiniLM-L6-v2's 256-token limit. A larger window is **silently truncated by the model**, so the tail of every long chunk would never reach the vector — plausible embeddings, no error. |
| `CHUNK_OVERLAP` | 100 chars | a sentence straddling a boundary stays retrievable; without overlap it is split across two vectors and matches neither query well |
| model | `sentence-transformers/all-MiniLM-L6-v2` | kept identical to the existing news pipeline so both vector tables share a distance operator and a query encoder |
| dimensions | 384 | must match `VECTOR(384)`; `repository.verify_schema()` checks the two agree before any batch job starts |

**Boundary preference: paragraph → sentence → word.** A window only breaks at a
boundary once it is at least half full, so a full stop 20 characters in cannot
win and emit a near-empty chunk. This matters because `chunk_text` is shown to
the user as the retrieved passage — a window ending mid-clause is a visible
defect, not an internal detail.

Most forecasts fit in one window unchanged. Alerts, where `description` and
`instruction` are joined, routinely need two or three.

An unknown model name **raises** rather than defaulting to a guessed
dimension — a wrong-width vector reaching a `VECTOR(384)` column fails deep
inside a batch insert, with a driver error naming neither the model nor the
column.

> This is the one place the project departs from an otherwise NVIDIA-first
> stack. Cross-pipeline vector compatibility is the binding constraint here.

---

## 4. Running the pipeline end to end

### Prerequisites

```bash
pip install -r requirements.txt

export LAKEBASE_URL="postgresql://<role>:<password>@<host>:5432/databricks_postgres?sslmode=require"
export NWS_USER_AGENT="(SkyIndex-AI, you@example.com)"
```

On Databricks the URL comes from the secret scope instead — run
`python setup_secrets.py` once, and `app.yaml` already points at it.

### Step 1 — schema

```bash
python -c "import lakebase; lakebase.apply_schema()"
```

Idempotent. Every statement is `IF NOT EXISTS`, so re-applying is a no-op.
`CREATE EXTENSION vector` runs first, so a missing extension fails immediately
rather than at the first vector insert.

### Step 2 — harvest

```bash
python app.py     # then, in another shell:

curl -X POST localhost:8000/weather/sync \
  -H 'Content-Type: application/json' \
  -d '{"locations": ["Chicago, IL", "Austin, TX", "39.7,-104.9"], "limit": 50}'
```

```json
{"synced": 111, "by_source_type": {"alert": 41, "forecast": 70},
 "locations": ["Chicago, IL", "Austin, TX", "39.7,-104.9"]}
```

Accepts `"City, ST"` and `"lat,lon"`. A bare city name is rejected — 
"Springfield" alone matches places in more than thirty states, and silently
guessing one would return confident weather for the wrong part of the country.

A location that fails is reported in an `errors` key rather than quietly
missing from the total: `"synced": 40` reads like success even when half the
batch failed.

### Step 3 — embed

```bash
python notebooks/ingest_weather_embeddings.py
```

```
pending    : 111 documents
documents embedded : 111
chunks written     : 140
elapsed            : 42.2s
```

Incremental. It reads documents with no *current* vectors, so a second run
does nothing and a run after a partial failure resumes exactly where it
stopped — "pending" is derived from the data, not from a stored cursor.

**After changing `CHUNK_SIZE`, `CHUNK_OVERLAP`, or the boundary rules, run with
`RESET=true`.** `content_hash` fingerprints the document *text*, not the
chunking config, so a chunker change leaves every vector stale in a way the
incremental anti-join cannot see — the documents did not change, so they look
done.

```bash
RESET=true python notebooks/ingest_weather_embeddings.py
```

The same file runs as a Databricks notebook, where widgets supply the config
instead of environment variables.

### Step 4 — search

```bash
curl -X POST localhost:8000/weather/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "flash flood risk this weekend", "top_k": 5}'
```

Or open `http://localhost:8000` for the UI.

`GET /weather/search?query=...` returns the same results plus a generated
natural-language answer over them.

### Step 5 — scheduled refresh (optional)

```bash
python scripts/refresh_weather_index.py      # sync + embed in one pass
```

As a Databricks Workflow:

```bash
databricks bundle deploy -t dev
databricks bundle run refresh_weather_index -t dev
```

**Deployed and running every 30 minutes.** Thirty rather than daily because
alerts are the perishable half of this corpus: a Flash Flood Warning is issued,
amended, and expires inside a few hours, so a daily index would mostly answer
questions about weather that has already happened.

It was run manually first, confirmed green, and only then put on the timer. The
bundle definition still ships `pause_status: PAUSED` so that deploying it into
a fresh workspace does not start firing before anyone has verified it there.

Three things had to be solved to make it run on Databricks at all, none of
which show up when running the same script locally:

1. **`__file__` is not defined.** Serverless executes a job's Python file via
   `exec(compile(...))`, so the usual `os.path.abspath(__file__)` raises
   `NameError` before the first import. `_project_root()` resolves the path
   without it.
2. **`psycopg2` cannot be installed.** The runtime already ships psycopg2
   2.9.11 in a directory pip refuses to uninstall from. Installing
   `psycopg2-binary` on top leaves two builds of the same native extension in
   one interpreter and `import psycopg2` aborts the process with SIGABRT — not
   an exception. Use the runtime's copy; see `requirements-databricks.txt`.
3. **A script task cannot restart its own interpreter.** `sentence-transformers`
   still needs installing at runtime, and any pip install into a live kernel
   needs a restart afterwards. Hence a notebook task wrapping the same
   `refresh()` function rather than a `spark_python_task`.

The click-path equivalent, for anyone without the CLI: **Workflows → Create Job
→ Python script task → `scripts/refresh_weather_index.py` → Add trigger →
Scheduled → cron `0 0,30 * * * ?`**.

---

## 5. API reference

| method | path | purpose |
| --- | --- | --- |
| `GET` | `/` | search UI |
| `GET` | `/healthz` | liveness, Lakebase reachability, row counts |
| `POST` | `/weather/sync` | harvest into `weather_documents` |
| `POST` | `/weather/search` | semantic search |
| `GET` | `/weather/search?query=` | search + generated answer |
| `GET` | `/weather/documents` | recently synced documents |

**Edge cases handled:**

| case | behaviour |
| --- | --- |
| empty `weather_embeddings` | `200` with `count: 0` — nothing synced yet is a normal state of a fresh deploy, not a failure |
| missing / blank / non-string `query` | `400` with a message naming the problem |
| `top_k` out of range | clamped to 1–20; a caller asking for 500 wants "as many as you can", and failing that helps nobody |
| unknown `source_type` | `400` listing the valid values |
| summarizer unavailable | `200`, results served, `"summary": null` |
| one location fails mid-batch | the rest still sync; the failure is reported in `errors` |
| any unhandled error | JSON, never an HTML error page — the UI parses every response as JSON |

---

## 6. Known limitations, and what I would do next

**The corpus is small enough that the index does not pay for itself yet.** At
140 vectors the planner correctly declines the HNSW index. The benchmark shows
the crossover arrives around a few thousand vectors. Syncing the full national
alert set and hourly forecasts for a few hundred gridpoints would put it well
past that; the code already supports it, it just needs more locations and more
API budget.

**Near-duplicate text still repeats in results.** Retrieval collapses exact
duplicates and limits results to one per document, which fixed the worst of it.
But NWS sometimes issues the same advisory for adjacent zones with a word or
two different — same meaning, different bytes — and those survive exact
matching. Catching them needs either a similarity threshold between result
vectors (drop a result within ~0.02 cosine of one already shown) or clustering
at ingest time. Both are real work and neither is free: too aggressive and
genuinely distinct alerts for neighbouring counties get suppressed.

**No time filtering in retrieval.** An expired Flash Flood Warning is still
indexed and can still be returned. `expires_at` is stored and indexed, so the
fix is a `WHERE expires_at > now()` predicate, but combining a filter with an
HNSW scan needs care — a filter applied after the graph walk can under-fill
`top_k`, which is why it is not a one-line change.

**No expiry sweep.** Documents accumulate forever. The FK cascades, so a
delete of expired documents cleans up their vectors, but nothing runs it. That
belongs in the scheduled job.

**Recall is untuned on the real corpus.** The `ef_search` sweep was run on
synthetic vectors, which is the hardest case. Measuring recall on real weather
embeddings needs a labelled relevance set — a few dozen queries with
hand-marked correct answers — which is the honest way to pick an `ef_search`
value rather than guessing from the synthetic curve.

**The generated answer is unverified locally.** Databricks Free Edition blocks
personal access token creation, so the AI Gateway call could not be exercised
from a laptop. Every code path around it is tested with mocks, including all
six failure modes, and it degrades to `"summary": null`. It resolves
credentials automatically when deployed as a Databricks App.

**Geocoding depends on a third party.** `"City, ST"` resolution uses
Open-Meteo. Coordinates skip it entirely, so the dependency is avoidable, but a
place name will not resolve if that service is down.

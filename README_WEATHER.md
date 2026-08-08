# SkyIndex-AI — data source, schema, and how to run it

Technical companion to [README.md](README.md). Covers the source choice and why,
the schema decisions, and the end-to-end pipeline.

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

Ships **PAUSED**, on a 30-minute schedule. Run it once by hand and confirm the
output before unpausing — an unattended job that has never succeeded is just a
scheduled way to fail. Thirty minutes rather than daily because alerts are the
perishable half of this corpus: a Flash Flood Warning is issued, amended, and
expires inside a few hours.

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

"""Print the state of the deployed pipeline in one screen.

Everything a reviewer needs to confirm the pipeline is real: the pgvector
extension, both tables with their column types, the HNSW index and its
operator class, row counts, and a live retrieval with scores.

    python scripts/verify_pipeline.py
"""

from __future__ import annotations

import os
import sys

def _project_root() -> str:
    """Locate the project root without depending on __file__.

    Databricks serverless runs a job's Python file through
    exec(compile(f.read(), filename, "exec")), which never binds __file__ in
    the executing namespace - so os.path.abspath(__file__) raises NameError
    before a single import has run. Notebooks have the same gap.

    Tries, in order: __file__, sys.argv[0], SKYINDEX_ROOT, then walks up from
    the working directory. A candidate counts only if repository.py is
    actually in it, so a wrong guess is rejected rather than silently used.
    """
    candidates = []

    try:
        candidates.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    except NameError:
        pass

    if sys.argv and sys.argv[0]:
        script = os.path.abspath(sys.argv[0])
        candidates.append(os.path.dirname(os.path.dirname(script)))
        candidates.append(os.path.dirname(script))

    if os.environ.get("SKYINDEX_ROOT"):
        candidates.append(os.environ["SKYINDEX_ROOT"])

    probe = os.getcwd()
    for _ in range(8):
        candidates.append(probe)
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent

    for candidate in candidates:
        if candidate and os.path.isfile(os.path.join(candidate, "repository.py")):
            return candidate

    raise RuntimeError(
        "Could not locate the SkyIndex-AI project root. Set SKYINDEX_ROOT to "
        f"the directory containing repository.py. Tried: {candidates}"
    )


_PROJECT_ROOT = _project_root()
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Quiet the model loader before it is imported. Its progress bars and hub
# warnings are written to stderr mid-report, which interleaves with the output
# below and makes the result hard to read.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import logging  # noqa: E402

os.environ.setdefault("HF_HUB_VERBOSITY", "error")
for _noisy in (
    "httpx", "httpcore", "urllib3", "filelock", "sentence_transformers",
    "huggingface_hub", "huggingface_hub.utils._http", "transformers",
):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

import embeddings  # noqa: E402
import lakebase  # noqa: E402
import repository  # noqa: E402

RULE = "=" * 72


def heading(text: str) -> None:
    print(f"\n{RULE}\n  {text}\n{RULE}")


def main() -> int:
    heading("1. pgvector extension")
    row = lakebase.run_query_one(
        "SELECT extname, extversion FROM pg_extension WHERE extname = 'vector'"
    )
    print(f"  {row['extname']} {row['extversion']}" if row else "  NOT INSTALLED")

    heading("2. Schema")
    rows = lakebase.run_query(
        """
        SELECT c.relname AS table_name,
               a.attname AS column_name,
               format_type(a.atttypid, a.atttypmod) AS data_type,
               a.attnotnull AS not_null
        FROM pg_attribute a
        JOIN pg_class c ON c.oid = a.attrelid
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE c.relname IN ('weather_documents', 'weather_embeddings')
          AND a.attnum > 0 AND NOT a.attisdropped AND n.nspname = 'public'
        ORDER BY c.relname, a.attnum
        """
    )
    current = None
    for row in rows:
        if row["table_name"] != current:
            current = row["table_name"]
            print(f"\n  {current}")
        flag = " NOT NULL" if row["not_null"] else ""
        marker = "  <-- pgvector" if row["data_type"].startswith("vector") else ""
        print(f"    {row['column_name']:<16} {row['data_type']}{flag}{marker}")

    heading("3. Indexes")
    for row in lakebase.run_query(
        """
        SELECT indexname, indexdef FROM pg_indexes
        WHERE tablename IN ('weather_documents', 'weather_embeddings')
        ORDER BY tablename, indexname
        """
    ):
        method = row["indexdef"].split("USING", 1)[-1].strip()
        marker = "  <-- vector index" if "hnsw" in method.lower() else ""
        print(f"  {row['indexname']:<42} {method}{marker}")

    heading("4. Row counts")
    stats = repository.stats()
    for key, value in stats.items():
        print(f"  {key:<20} {value}")

    # Load the model BEFORE the section header, so any remaining loader
    # chatter lands above the results rather than inside them.
    embeddings.get_encoder()

    heading("5. Live retrieval")
    query = "flash flood risk this weekend"
    print(f"  query: {query!r}")
    print(f"  model: {embeddings.EMBEDDING_MODEL} ({embeddings.EMBEDDING_DIM}-dim)")
    print(f"  chunking: size={embeddings.CHUNK_SIZE} overlap={embeddings.CHUNK_OVERLAP}\n")

    for position, hit in enumerate(
        repository.search(embeddings.embed_query(query), top_k=5), start=1
    ):
        text = " ".join((hit["chunk_text"] or "").split())
        print(
            f"  #{position}  {hit['similarity']:.3f}  [{hit['source_type']:8}] "
            f"{(hit['event'] or '')[:28]}"
        )
        print(f"        {hit['location'][:64]}")
        print(f"        {text[:64]}...")

    print(f"\n{RULE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Measure what the HNSW index actually buys, and what it costs.

Two things make a naive version of this benchmark meaningless, and both are
addressed here.

**Client wall-clock measures the network, not the index.** Lakebase is in
us-west-2; a round trip from a laptop is 70-80 ms, which swamps a query that
executes in under a millisecond. Every timing below comes from
`EXPLAIN (ANALYZE)`'s own Execution Time, so it reports what Postgres spent,
not what the WiFi cost.

**A small corpus cannot show an index winning.** HNSW walks a proximity graph
instead of scanning. That trade only pays once scanning is the dominant cost,
which is far more vectors than a handful of cities produce. So the script also
runs a scale sweep against a synthetic table, clearly separated from the real
measurement, to locate the crossover rather than assert it.

Latency alone would still be a misleading result: HNSW is approximate, so it
can miss neighbours an exact scan would find. Recall against exact search is
reported alongside every speed number.

Run:
    python benchmarks/hnsw_benchmark.py
    python benchmarks/hnsw_benchmark.py --scale 1000,10000,100000
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import embeddings  # noqa: E402
import lakebase  # noqa: E402
import repository  # noqa: E402

# Deliberately varied: short and long, hazard and everyday, wording that
# appears verbatim in NWS text and wording that does not. A benchmark run on
# one query shape measures that shape, not the index.
QUERIES = [
    "flash flood risk this weekend",
    "dangerous heat, what should I do",
    "will it rain tomorrow morning",
    "high wind and fire danger",
    "is it safe to drive tonight",
    "coastal flooding and storm surge",
    "thunderstorms with hail",
    "freezing conditions overnight",
    "air quality is bad",
    "should I evacuate",
]

SCALE_TABLE = "benchmark_vectors"


# ---------------------------------------------------------------------------
# Server-side timing
# ---------------------------------------------------------------------------


def execution_ms(cursor, sql: str, params: tuple) -> float:
    """Run a query under EXPLAIN ANALYZE and return Postgres' own timing.

    Client-side wall clock would be dominated by the round trip to us-west-2 -
    two orders of magnitude larger than the thing being measured.
    """
    cursor.execute(f"EXPLAIN (ANALYZE, FORMAT JSON) {sql}", params)
    row = cursor.fetchone()
    plan = row["QUERY PLAN"] if isinstance(row, dict) else row[0]
    if isinstance(plan, str):
        plan = json.loads(plan)
    return float(plan[0]["Execution Time"])


def plan_text(cursor, sql: str, params: tuple) -> str:
    cursor.execute(f"EXPLAIN {sql}", params)
    return " ".join(str(dict(row) if isinstance(row, dict) else row) for row in cursor.fetchall())


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
    return ordered[index]


def _measure(cursor, sql: str, vectors: list[str], top_k: int, runs: int) -> list[float]:
    timings = []
    for _ in range(runs):
        for vector in vectors:
            timings.append(execution_ms(cursor, sql, (vector, vector, top_k)))
    return timings


def _ids(cursor, sql: str, vector: str, top_k: int) -> set[str]:
    cursor.execute(sql, (vector, vector, top_k))
    return {str(row["id"]) for row in cursor.fetchall()}


# ---------------------------------------------------------------------------
# The real corpus
# ---------------------------------------------------------------------------


def benchmark_real(runs: int, top_k: int) -> dict:
    total = repository.count_embeddings()
    if not total:
        raise SystemExit("weather_embeddings is empty - run sync and ingest first.")

    sql = """
        SELECT e.id, 1 - (e.embedding <=> %s::vector) AS similarity
        FROM weather_embeddings e
        ORDER BY e.embedding <=> %s::vector
        LIMIT %s
    """

    print(f"encoding {len(QUERIES)} queries...")
    vectors = [repository.to_vector_literal(embeddings.embed_query(q)) for q in QUERIES]

    with lakebase.get_connection() as conn:
        with conn.cursor() as cur:
            execution_ms(cur, sql, (vectors[0], vectors[0], top_k))  # warm caches

            uses_index = "idx_weather_embeddings_hnsw" in plan_text(
                cur, sql, (vectors[0], vectors[0], top_k)
            )
            indexed = _measure(cur, sql, vectors, top_k, runs)
            approximate = [_ids(cur, sql, v, top_k) for v in vectors]

            cur.execute("SET enable_indexscan = off")
            cur.execute("SET enable_bitmapscan = off")
            sequential = _measure(cur, sql, vectors, top_k, runs)
            exact = [_ids(cur, sql, v, top_k) for v in vectors]
            cur.execute("RESET enable_indexscan")
            cur.execute("RESET enable_bitmapscan")

    recalls = [
        len(truth & found) / len(truth) if truth else 1.0
        for truth, found in zip(exact, approximate)
    ]

    return {
        "vectors": total,
        "runs": runs,
        "top_k": top_k,
        "index_used": uses_index,
        "indexed_p50": statistics.median(indexed),
        "indexed_p95": _percentile(indexed, 0.95),
        "sequential_p50": statistics.median(sequential),
        "sequential_p95": _percentile(sequential, 0.95),
        "recall_mean": statistics.mean(recalls),
        "recall_min": min(recalls),
        "perfect": sum(1 for r in recalls if r == 1.0),
    }


# ---------------------------------------------------------------------------
# Synthetic scale sweep
# ---------------------------------------------------------------------------


def benchmark_scale(sizes: list[int], runs: int, top_k: int, seed: int = 17) -> list[dict]:
    """Find where the index starts winning, using generated vectors.

    Synthetic and labelled as such. The point is the shape of the curve - how
    each access path responds to corpus size - which random unit vectors show
    just as well as real ones. Recall is still measured against exact search,
    because random data is the *hardest* case for an approximate index: with
    no cluster structure to exploit, the graph has nothing to prune on.
    """
    rng = random.Random(seed)
    dimensions = repository.VECTOR_DIM

    def random_vector() -> str:
        values = [rng.gauss(0.0, 1.0) for _ in range(dimensions)]
        norm = sum(value * value for value in values) ** 0.5 or 1.0
        return "[" + ",".join(str(value / norm) for value in values) + "]"

    sql = f"""
        SELECT b.id, 1 - (b.embedding <=> %s::vector) AS similarity
        FROM {SCALE_TABLE} b
        ORDER BY b.embedding <=> %s::vector
        LIMIT %s
    """
    probes = [random_vector() for _ in range(len(QUERIES))]
    results = []

    with lakebase.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {SCALE_TABLE}")
            cur.execute(
                f"CREATE TABLE {SCALE_TABLE} "
                f"(id BIGSERIAL PRIMARY KEY, embedding VECTOR({dimensions}) NOT NULL)"
            )

            inserted = 0
            for size in sorted(sizes):
                needed = size - inserted
                print(f"  populating to {size:,} vectors (+{needed:,})...")
                batch = []
                for _ in range(needed):
                    batch.append((random_vector(),))
                    if len(batch) >= 2000:
                        _insert_batch(cur, batch)
                        batch = []
                if batch:
                    _insert_batch(cur, batch)
                inserted = size

                # Build the index fresh at this size, then ANALYZE so the
                # planner has statistics to choose with - without it the
                # planner works from defaults and may pick the wrong path for
                # reasons that have nothing to do with the index.
                cur.execute(f"DROP INDEX IF EXISTS idx_{SCALE_TABLE}_hnsw")
                started = time.perf_counter()
                cur.execute(
                    f"CREATE INDEX idx_{SCALE_TABLE}_hnsw ON {SCALE_TABLE} "
                    "USING hnsw (embedding vector_cosine_ops)"
                )
                build_seconds = time.perf_counter() - started
                cur.execute(f"ANALYZE {SCALE_TABLE}")

                execution_ms(cur, sql, (probes[0], probes[0], top_k))
                uses_index = f"idx_{SCALE_TABLE}_hnsw" in plan_text(
                    cur, sql, (probes[0], probes[0], top_k)
                )
                indexed = _measure(cur, sql, probes, top_k, runs)
                approximate = [_ids(cur, sql, v, top_k) for v in probes]

                cur.execute("SET enable_indexscan = off")
                cur.execute("SET enable_bitmapscan = off")
                sequential = _measure(cur, sql, probes, top_k, runs)
                exact = [_ids(cur, sql, v, top_k) for v in probes]
                cur.execute("RESET enable_indexscan")
                cur.execute("RESET enable_bitmapscan")

                recalls = [
                    len(t & f) / len(t) if t else 1.0 for t, f in zip(exact, approximate)
                ]
                results.append(
                    {
                        "size": size,
                        "index_used": uses_index,
                        "build_seconds": build_seconds,
                        "indexed_p50": statistics.median(indexed),
                        "sequential_p50": statistics.median(sequential),
                        "recall_mean": statistics.mean(recalls),
                    }
                )
                print(
                    f"    indexed {results[-1]['indexed_p50']:.3f} ms | "
                    f"seq {results[-1]['sequential_p50']:.3f} ms | "
                    f"recall {results[-1]['recall_mean']:.3f}"
                )

            cur.execute(f"DROP TABLE IF EXISTS {SCALE_TABLE}")

    return results


def benchmark_ef_search(size: int, settings: list[int], runs: int, top_k: int, seed: int = 17) -> list[dict]:
    """Trace recall and latency against hnsw.ef_search.

    Low recall from an HNSW index is not a defect to report, it is a knob that
    has not been turned. ef_search sets how many candidates the graph walk
    keeps in flight: raise it and the search explores more of the graph,
    recovering neighbours a narrow walk missed, at the cost of time. This is
    the curve that says which trade a given deployment should take.
    """
    rng = random.Random(seed)
    dimensions = repository.VECTOR_DIM

    def random_vector() -> str:
        values = [rng.gauss(0.0, 1.0) for _ in range(dimensions)]
        norm = sum(value * value for value in values) ** 0.5 or 1.0
        return "[" + ",".join(str(value / norm) for value in values) + "]"

    sql = f"""
        SELECT b.id, 1 - (b.embedding <=> %s::vector) AS similarity
        FROM {SCALE_TABLE} b
        ORDER BY b.embedding <=> %s::vector
        LIMIT %s
    """
    probes = [random_vector() for _ in range(len(QUERIES))]
    results = []

    with lakebase.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {SCALE_TABLE}")
            cur.execute(
                f"CREATE TABLE {SCALE_TABLE} "
                f"(id BIGSERIAL PRIMARY KEY, embedding VECTOR({dimensions}) NOT NULL)"
            )
            print(f"  populating {size:,} vectors...")
            batch = []
            for _ in range(size):
                batch.append((random_vector(),))
                if len(batch) >= 2000:
                    _insert_batch(cur, batch)
                    batch = []
            if batch:
                _insert_batch(cur, batch)

            cur.execute(
                f"CREATE INDEX idx_{SCALE_TABLE}_hnsw ON {SCALE_TABLE} "
                "USING hnsw (embedding vector_cosine_ops)"
            )
            cur.execute(f"ANALYZE {SCALE_TABLE}")

            # Ground truth, once: exact neighbours with the index disabled.
            cur.execute("SET enable_indexscan = off")
            cur.execute("SET enable_bitmapscan = off")
            exact = [_ids(cur, sql, v, top_k) for v in probes]
            sequential = statistics.median(_measure(cur, sql, probes, top_k, runs))
            cur.execute("RESET enable_indexscan")
            cur.execute("RESET enable_bitmapscan")

            for ef_search in settings:
                cur.execute(f"SET hnsw.ef_search = {int(ef_search)}")
                timings = _measure(cur, sql, probes, top_k, runs)
                found = [_ids(cur, sql, v, top_k) for v in probes]
                recalls = [
                    len(t & f) / len(t) if t else 1.0 for t, f in zip(exact, found)
                ]
                results.append(
                    {
                        "ef_search": ef_search,
                        "p50": statistics.median(timings),
                        "recall": statistics.mean(recalls),
                        "sequential_p50": sequential,
                    }
                )
                print(
                    f"    ef_search={ef_search:<4} p50 {results[-1]['p50']:.3f} ms  "
                    f"recall {results[-1]['recall']:.3f}"
                )

            cur.execute("RESET hnsw.ef_search")
            cur.execute(f"DROP TABLE IF EXISTS {SCALE_TABLE}")

    return results


def _insert_batch(cursor, rows) -> None:
    from psycopg2.extras import execute_values

    execute_values(
        cursor,
        f"INSERT INTO {SCALE_TABLE} (embedding) VALUES %s",
        rows,
        template="(%s::vector)",
        page_size=500,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def report(real: dict, scale: list[dict] | None, ef: list[dict] | None = None, ef_size: int = 0) -> str:
    speedup = real["sequential_p50"] / real["indexed_p50"] if real["indexed_p50"] else 0.0

    lines = [
        "# HNSW index benchmark",
        "",
        "All timings are Postgres' own `EXPLAIN (ANALYZE)` execution time, not",
        "client wall clock. A round trip from a laptop to Lakebase in us-west-2",
        "is 70-80 ms, which would otherwise swamp a sub-millisecond query and",
        "turn this into a measurement of the network.",
        "",
        f"## Real corpus ({real['vectors']} vectors)",
        "",
        "| metric | HNSW index | sequential scan |",
        "| --- | --- | --- |",
        f"| p50 | {real['indexed_p50']:.3f} ms | {real['sequential_p50']:.3f} ms |",
        f"| p95 | {real['indexed_p95']:.3f} ms | {real['sequential_p95']:.3f} ms |",
        f"| speedup (p50) | {speedup:.2f}x | 1.00x |",
        "",
        f"- recall@{real['top_k']}: mean {real['recall_mean']:.3f}, "
        f"worst {real['recall_min']:.3f} ({real['perfect']}/{len(QUERIES)} perfect)",
        f"- planner chose the HNSW index: **{real['index_used']}**",
        "",
    ]

    if not real["index_used"]:
        lines += [
            "The planner declined the index, and it was right to. At this size the",
            "whole table is a few hundred kilobytes and already in cache, so a scan",
            "costs less than walking a proximity graph. An index the planner refuses",
            "is not a broken index - it is a planner correctly reading its own",
            "statistics.",
            "",
        ]

    if scale:
        lines += [
            "## Scale sweep (synthetic vectors)",
            "",
            "Random unit vectors, not weather text. The shape of the curve is the",
            "point, and random data is the *hardest* case for an approximate index:",
            "with no cluster structure to exploit, the graph has nothing to prune on,",
            "so the recall below is a lower bound rather than a typical figure.",
            "",
            "| vectors | HNSW p50 | seq scan p50 | speedup | recall@k | index used | build |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for row in scale:
            gain = row["sequential_p50"] / row["indexed_p50"] if row["indexed_p50"] else 0.0
            lines.append(
                f"| {row['size']:,} | {row['indexed_p50']:.3f} ms | "
                f"{row['sequential_p50']:.3f} ms | {gain:.2f}x | "
                f"{row['recall_mean']:.3f} | {row['index_used']} | "
                f"{row['build_seconds']:.1f}s |"
            )
        lines.append("")

    if ef:
        baseline = ef[0]["sequential_p50"]
        lines += [
            f"## Tuning recall with `hnsw.ef_search` ({ef_size:,} synthetic vectors)",
            "",
            "Low recall from an HNSW index is a knob that has not been turned, not a",
            "defect. `ef_search` sets how many candidates the graph walk keeps in",
            "flight; raising it explores more of the graph and recovers neighbours a",
            "narrow walk missed, at the cost of time. Exact search on the same data",
            f"takes {baseline:.3f} ms, which is the ceiling this is trading against.",
            "",
            "| ef_search | p50 | recall@k | still faster than exact |",
            "| --- | --- | --- | --- |",
        ]
        for row in ef:
            lines.append(
                f"| {row['ef_search']} | {row['p50']:.3f} ms | {row['recall']:.3f} | "
                f"{'yes' if row['p50'] < baseline else 'no'} |"
            )
        lines.append("")

    lines += [
        "## Why the index is still correct to create",
        "",
        "A sequential scan is O(n) in the corpus. This table grows every time the",
        "sync runs, so the scan's cost grows with it while the index's barely",
        "moves. Creating the index only once the scan has become a problem means",
        "building it under load, on a table that is already too slow.",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=15, help="repetitions per query")
    parser.add_argument("--top-k", type=int, default=5, help="neighbours to retrieve")
    parser.add_argument(
        "--scale",
        default="",
        help="comma-separated corpus sizes for the synthetic sweep, e.g. 1000,10000,50000",
    )
    parser.add_argument(
        "--ef-search",
        default="",
        help="comma-separated hnsw.ef_search values to sweep, e.g. 40,100,200,400",
    )
    parser.add_argument(
        "--ef-size", type=int, default=20000, help="corpus size for the ef_search sweep"
    )
    parser.add_argument("--out", help="write the markdown report to this file")
    args = parser.parse_args()

    real = benchmark_real(runs=args.runs, top_k=args.top_k)

    scale = None
    if args.scale:
        sizes = [int(value) for value in args.scale.split(",") if value.strip()]
        print(f"\nscale sweep: {sizes}")
        scale = benchmark_scale(sizes, runs=args.runs, top_k=args.top_k)

    ef = None
    if args.ef_search:
        settings = [int(value) for value in args.ef_search.split(",") if value.strip()]
        print(f"\nef_search sweep at {args.ef_size:,} vectors: {settings}")
        ef = benchmark_ef_search(args.ef_size, settings, runs=args.runs, top_k=args.top_k)

    markdown = report(real, scale, ef, args.ef_size)
    print("\n" + markdown)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(markdown + "\n")
        print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()

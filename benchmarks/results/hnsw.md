# HNSW index benchmark

All timings are Postgres' own `EXPLAIN (ANALYZE)` execution time, not
client wall clock. A round trip from a laptop to Lakebase in us-west-2
is 70-80 ms, which would otherwise swamp a sub-millisecond query and
turn this into a measurement of the network.

## Real corpus (140 vectors)

| metric | HNSW index | sequential scan |
| --- | --- | --- |
| p50 | 0.320 ms | 0.307 ms |
| p95 | 0.341 ms | 0.332 ms |
| speedup (p50) | 0.96x | 1.00x |

- recall@5: mean 1.000, worst 1.000 (10/10 perfect)
- planner chose the HNSW index: **False**

The planner declined the index, and it was right to. At this size the
whole table is a few hundred kilobytes and already in cache, so a scan
costs less than walking a proximity graph. An index the planner refuses
is not a broken index - it is a planner correctly reading its own
statistics.

## Scale sweep (synthetic vectors)

Random unit vectors, not weather text. The shape of the curve is the
point, and random data is the *hardest* case for an approximate index:
with no cluster structure to exploit, the graph has nothing to prune on,
so the recall below is a lower bound rather than a typical figure.

| vectors | HNSW p50 | seq scan p50 | speedup | recall@k | index used | build |
| --- | --- | --- | --- | --- | --- | --- |
| 2,000 | 0.595 ms | 0.902 ms | 1.52x | 0.760 | True | 0.6s |
| 20,000 | 1.207 ms | 8.869 ms | 7.35x | 0.280 | True | 5.4s |

## Tuning recall with `hnsw.ef_search` (20,000 synthetic vectors)

Low recall from an HNSW index is a knob that has not been turned, not a
defect. `ef_search` sets how many candidates the graph walk keeps in
flight; raising it explores more of the graph and recovers neighbours a
narrow walk missed, at the cost of time. Exact search on the same data
takes 8.284 ms, which is the ceiling this is trading against.

| ef_search | p50 | recall@k | still faster than exact |
| --- | --- | --- | --- |
| 40 | 1.188 ms | 0.180 | yes |
| 100 | 2.311 ms | 0.400 | yes |
| 200 | 3.711 ms | 0.660 | yes |
| 400 | 6.120 ms | 0.800 | yes |

## Why the index is still correct to create

A sequential scan is O(n) in the corpus. This table grows every time the
sync runs, so the scan's cost grows with it while the index's barely
moves. Creating the index only once the scan has become a problem means
building it under load, on a table that is already too slow.

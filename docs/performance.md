# Performance

Docker stack (one uvicorn process, one worker, Postgres 16) after the demo: 10,001 compounds, 746 measurements, two model versions.
Reproduce with `make demo`, then `make load-test`.

## Reads: the target's evidence page, one compound's trace, the job list, in rotation, closed loop, 10 s per level

| Concurrency | Requests | Throughput | p50 | p95 | p99 | Errors |
|---|---|---|---|---|---|---|
| 1 | 575 | 57.4 req/s | 16.2 ms | 18.0 ms | 20.2 ms | 0.00% |
| 8 | 879 | 87.0 req/s | 36.6 ms | 221.9 ms | 236.8 ms | 0.00% |
| 32 | 643 | 62.4 req/s | 170.6 ms | 918.7 ms | 979.2 ms | 0.00% |

Target from the blueprint: p95 under 500 ms for reads: met at 8 concurrent readers, not at 32. The trace runs the five-model
ensemble for the compound and once more for each of its fingerprint bits (one batch), about 15 ms of CPU, and a single process
serialises that. A first version predicted one row at a time (135 calls) and let OpenMP threads fight: p95 was 13 seconds at 8
readers. Batching and `OMP_NUM_THREADS=1` brought it here; a second worker process would take it the rest of the way.

## Compute (in-process)

| Work | Time |
|---|---|
| Generate the library: 10,000 molecules canonicalised, fingerprinted, described | 3.6 s |
| Train one model version (five fits plus a held-out probe), 300 to 400 compounds | 7 to 10 s |
| Predict the whole library | 0.4 s |
| Select a plate (10,000 candidates, Tanimoto diversity) | about 1 s |
| Evaluation: three libraries, nine plates, twelve model fits | about 2 minutes |

Not measured: libraries of millions, which is where fingerprint search and training both need different engineering.

# Latest Benchmark Result

Run date: 2026-08-23

The checked-in JSON is the authoritative machine-readable result. The main comparison from this run is:

| System | Avg. segments | Avg. documents scored | Recall@10 | MRR | Est. reranker tokens | Local latency |
|---|---:|---:|---:|---:|---:|---:|
| BM25 all segments | 12.0 | 600.0 | 1.000 | 1.000 | 452.32 | 0.3671 ms |
| Standard Bloom + BM25 | 12.0 | 600.0 | 1.000 | 1.000 | 452.32 | 0.4067 ms |
| Counting Bloom + BM25 | 12.0 | 600.0 | 1.000 | 1.000 | 452.32 | 0.3993 ms |
| Certainty cascade + BM25 | 1.0 | 50.0 | 1.000 | 1.000 | 226.58 | 0.3604 ms |

On this synthetic benchmark, the cascade reduced average segments and documents scored by 91.7%, reduced the estimated reranker payload by 49.9%, and preserved perfect held-out Recall@10. Local search latency was 11.4% lower than the standard Bloom baseline.

The tradeoff was memory: the standard packed Bloom filters required 384 logical bytes, while 16-bit counting filters plus the classifier required 6,032 logical bytes—about 15.7 times more. Standard filters did not skip segments because common query terms occurred throughout the collection. Counting filters alone produced the same routing behavior; the gain came from combining collision certainty with the learned relevance stage and a smaller reranking budget.

These findings are evidence only for the included synthetic workload. A real collection, human relevance judgments, repeated timing trials, confidence intervals, query-distribution shift, and adversarial evaluation are required before making production claims.

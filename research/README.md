# Certainty-Aware Cascaded BloomSearch Research

## Question

Can a certainty-aware learned Bloom-filter cascade reduce BM25 computation and estimated hosted-model token usage while preserving Recall@10?

## Systems compared

1. BM25 over every segment.
2. Standard term Bloom filters followed by BM25.
3. Counting Bloom filters followed by BM25.
4. Counting Bloom filters, collision-certainty features, a learned logistic segment classifier, BM25, and a five-candidate reranker budget.

## Reproduce

```bash
python -m bloom_search benchmark
```

This writes `research/results/latest.json`.

## Benchmark limits

The included benchmark is deterministic and synthetic: 600 documents across 12 topical segments, with 300 training and 300 held-out test queries. It intentionally includes common terms across segments to stress fixed Bloom-filter routing. Results demonstrate implementation behavior, not production effectiveness or statistical generalization.

No hosted-model API is called. Reranker tokens are estimated as payload characters divided by four. Logical filter memory assumes packed bits for the standard filter and 16-bit counters for the counting filter; it excludes Python object overhead. Latency measures local Python execution and should be rerun on the target hardware.

## NLP completion evaluation

The automatic single-signal versus combined-signal ablation, paired confidence
intervals, significance tests, and local timing are documented in
[`NONHUMAN_COMPLETION_EVALUATION.md`](NONHUMAN_COMPLETION_EVALUATION.md). The
completed two-participant evaluation of intent preservation, clarification
usefulness, perceived task success, trust, and action agreement is documented
in [`HUMAN_STUDY_RESULTS.md`](HUMAN_STUDY_RESULTS.md).

The authoritative synthesis for the four official NLP research questions is
[`FINAL_RESEARCH_QUESTION_RESULTS.md`](FINAL_RESEARCH_QUESTION_RESULTS.md).
It includes recomputed QReCC Recall@10 and NDCG@10, SciFact retrieval results,
the hierarchical action-policy evaluation, call/token/latency trade-offs, and
the remaining paired human baseline required for the exact semantic RQ3 claim.


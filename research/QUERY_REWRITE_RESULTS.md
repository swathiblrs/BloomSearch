# Conversational Query-Rewriting Results

## Current production status

BloomSearch performs query rewriting only through a user-supplied hosted API.
The joint action policy decides whether rewriting is appropriate, and the
intent guard rejects unsafe or unchanged rewrites. Without API credentials,
the application retains the original query and continues with BM25.

No live hosted-provider benchmark has been completed yet. The measurements
below are retained as historical offline experiments and must not be described
as hosted-model results.

## Historical QReCC experiments

| Experiment | Queries | Result |
|---|---:|---|
| Final offline rewrite pipeline | 2,803 | Hit@5: 33.39% original, 42.35% rewritten |
| Final offline rewrite pipeline | 2,803 | MRR@5: 0.2425 original, 0.2863 rewritten |
| Selective rewrite policy | 2,803 | Hit@10: 34.25% keep-all, 58.65% selective, 73.56% rewrite-all |
| Rewrite-benefit classifier | 2,755 held-out examples | Macro-F1: 0.5885 |

The selective policy recommended rewriting for 47.59% of evaluated queries.
It reduced rewriting relative to the rewrite-all baseline but also lost 14.91
percentage points of Hit@10. This is a quality/cost trade-off, not evidence that
the selective predictor outperformed unconditional rewriting.

Aggregate machine-readable values are stored in
`research/results/query_rewriting/summary.json`. Raw QReCC files, generated
indexes, trained joblib artifacts, and per-query outputs are intentionally not
tracked. See `data/qrecc/README.md` for reproduction instructions.

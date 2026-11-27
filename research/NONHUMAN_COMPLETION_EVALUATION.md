# Non-Human Research-Question Completion Evaluation

## Scope

This evaluation completes the automatic work that can be supported by the
checked-in QReCC and SciFact artifacts without making a new hosted-model call.
It evaluates the two NLP layers separately because the available rewriting and
reranking experiments use different public datasets. It does not combine their
scores into an invalid cross-dataset number.

Reproduce the experiment after installing the `research` extras:

```bash
python research/evaluate_nonhuman_completion.py --measure-latency
```

The machine-readable result is
`research/results/nonhuman_completion.json`.

The historical QReCC Hit@10 summary below is retained for the original action
ablation. The authoritative RQ2 retrieval values were later recomputed from
the checked-in deduplicated test index and mapping over 2,765 present targets;
see [`FINAL_RESEARCH_QUESTION_RESULTS.md`](FINAL_RESEARCH_QUESTION_RESULTS.md).

## Layer 1: QReCC Query Processing

The quality analysis uses 2,803 held-out QReCC queries. When a policy selects
`rewrite`, the replay uses the human gold rewrite. This tests whether the
action was selected for a useful query; it does not measure a newly generated
hosted rewrite.

### Retrieval comparison

| Method | Hit@10 | Rewrite rate | Clarification rate |
|---|---:|---:|---:|
| Original query with BM25 | 34.25% | 0% | 0% |
| Always use gold rewrite | 73.56% | 100% | 0% |
| Current joint policy | **62.86%** | 56.01% | 0.04% |

The joint policy improved Hit@10 over original-query BM25 by 28.61 percentage
points. The paired 95% bootstrap interval was +26.94 to +30.32 points, and the
exact McNemar test gave `p < 0.001`. The result is statistically clear for this
offline gold-rewrite replay.

### Single-signal versus combined-signal ablation

All learned comparisons use the same protocol: 5-fold stratified
cross-validation repeated ten times. The target is whether the gold rewrite
improves Hit@10.

| Feature family | ROC-AUC | Positive F1 | Macro-F1 |
|---|---:|---:|---:|
| Linguistic only | 0.6745 | 0.5894 | 0.6252 |
| BM25 only | 0.5239 | 0.4067 | 0.5146 |
| Bloom coverage only | 0.5015 | 0.5735 | 0.2897 |
| Linguistic + BM25 | 0.6741 | 0.5883 | 0.6252 |
| All signals | **0.6748** | 0.5891 | **0.6258** |

The combined feature set has the highest mean AUC and Macro-F1, but the gain
over linguistic features alone is negligible. Bloom coverage is almost a
constant on this collection and performs at chance by AUC. Therefore, this
experiment does **not** establish that Bloom evidence improves query-action
prediction on QReCC. It establishes that the current query-action signal comes
primarily from linguistic features.

This negative result is useful: a future Bloom contribution requires a dataset
or segment representation with meaningful variation in collision certainty or
coverage, followed by an ablation of that richer signal.

### Local latency

Latency was measured on a deterministic sample of 300 queries. It includes
local Python decision and BM25 computation but excludes hosted generation.

| Method | Mean | Median | p95 |
|---|---:|---:|---:|
| BM25 original query | 78.88 ms | 76.21 ms | 131.19 ms |
| Joint policy with gold-rewrite replay | 178.80 ms | 176.56 ms | 269.78 ms |

The Layer 1 joint path adds approximately 100 ms of local work and is about
2.27 times the BM25-only local latency. The project should not claim that Layer
1 is faster. Its benefit is better retrieval and selective hosted usage, with
additional local computation.

## Layer 2: SciFact Selective Reranking

The Layer 2 evaluation replays stored hosted-reranker outputs for all 300
SciFact test queries.

| Method | Retrieved-pool NDCG@10 | Reranking calls |
|---|---:|---:|
| BM25 only | 0.6725 | 0 |
| Rerank every query | 0.7127 | 300 |
| Selective margin rule | **0.7189** | 130 |

Selective reranking improved retrieved-pool NDCG@10 over BM25 by 0.0464. The
paired 95% bootstrap interval was +0.0285 to +0.0650, and the paired
randomization test gave `p < 0.001`.

Selective reranking was 0.0062 above reranking every query, but its confidence
interval was -0.0083 to +0.0215 and `p = 0.413`. Therefore, it should be
described as statistically comparable to reranking every query, not
significantly better.

The published standard-NDCG context remains BM25 0.6635, rerank-all 0.7015,
and held-out selective reranking 0.7023 at a 43% call rate.

### Efficiency

| Metric | Result |
|---|---:|
| Reranking-call reduction | **56.67%** |
| Token reduction | **57.05%** |
| Average hosted latency across all queries | 1.14 s |
| Average latency per selected reranking call | 2.64 s |

This supports the Layer 2 claim: selective reranking preserves the quality of
always-on reranking while avoiding more than half of its calls and tokens.

## Research-Question Status Before Human Evaluation

### RQ1: Joint action selection

**Automatically answered in part.** The ablation is complete for rewrite
benefit, and it shows that all signals do not materially outperform linguistic
features on QReCC. The human three-class keep/rewrite/clarify labels are still
required for the complete action-selection result.

### RQ2: Retrieval effectiveness

**Automatically answered at the layer level.** BM25, always-on, and selective
methods are compared on identical queries within QReCC and within SciFact,
with paired confidence intervals and tests. A single end-to-end score is not
claimed because no available dataset contains the stored outputs required for
both layers.

### RQ3: Intent-preserving rewriting

**Not fully answered automatically.** Retrieval improvements and degradation
proxies are available, but semantic intent preservation requires the planned
human judgments.

### RQ4: Calls, tokens, latency, and task success

**The automatic portion is answered.** Calls, tokens, local latency, and stored
hosted latency are measured. User task success remains part of the human study.

## Work Remaining at This Evaluation Stage

After this evaluation, the remaining required work is human annotation and
analysis for:

1. Three-class keep/rewrite/clarify Macro-F1.
2. Semantic harmful-rewrite rate.
3. Clarification usefulness and task success.
4. Trust and participant agreement.

The final paper should also discuss the negative Bloom ablation result and
avoid claiming that Bloom certainty improves NLP decisions until it is tested
with a more informative certainty feature on a suitable collection.

## Subsequent completion

P01 and P02 later completed the planned human evaluation. The final human
metrics and research-question conclusions are reported in
[`HUMAN_STUDY_RESULTS.md`](HUMAN_STUDY_RESULTS.md). No P03 response is required
by the two-participant pilot protocol.

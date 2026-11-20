# Integrated JointActionPolicy Evaluation

## Scope

This is a zero-new-API-cost offline replay of the integrated policy. Layer 1
uses 2,803 held-out QReCC queries and the historical QReCC search index. Layer 2
uses 300 SciFact queries and the stored outputs from the completed hosted
reranking experiment.

The layers are reported separately because QReCC and SciFact are different
datasets. No single combined end-to-end quality score is claimed.

## Layer 1: keep, rewrite, or clarify

When the integrated policy selected `rewrite`, the replay used the human QReCC
gold rewrite. This isolates action-selection potential but does not measure the
quality of a newly generated hosted rewrite. A clarification was counted as
unresolved because QReCC does not provide a simulated user response.

| Method | Hit@10 | Rewrite-call rate |
|---|---:|---:|
| Keep original query | 34.25% | 0% |
| Previous selective rewriting | 58.65% | 47.59% |
| Integrated policy with gold-rewrite replay | **62.86%** | 56.01% |
| Always use gold rewrite | 73.56% | 100% |

The integrated policy selected 1,232 keep actions, 1,570 rewrite actions, and
one clarification. Rewrite-benefit precision was 0.5197, recall was 0.7253,
positive-class F1 was **0.6056**, and macro-F1 was **0.6202**.

Compared with the previous selective policy, Hit@10 increased by **4.21
percentage points**, but the rewrite-call rate increased by **8.42 percentage
points**. The result is better for retrieval quality, but not unconditionally
better for efficiency.

## Layer 2: return BM25 or rerank

The current runtime threshold (`BM25 score margin <= 3.84`) selected 130 of 300
queries for reranking.

| Metric | Result |
|---|---:|
| Reranking-call rate | 43.33% |
| Reranking-call reduction | **56.67%** |
| Token reduction | **57.05%** |
| Rerank-benefit F1 | 0.4560 |
| Rerank-benefit macro-F1 | 0.5990 |
| Average latency per selected call | 2.64 s |
| Average latency across all queries | 1.14 s |

For the stored retrieved-pool NDCG values, BM25 scored 0.6725, reranking every
query scored 0.7127, and the integrated selective rule scored **0.7189**. The
selective value is higher because the rule avoids some queries on which the
stored reranker harmed the ordering.

The published standard-NDCG context remains: BM25 0.6635, rerank-all 0.7015,
and the earlier held-out selective result 0.7023 at a 43% call rate. These
standard-NDCG figures should be used in the paper when comparing with other
systems.

## Conclusion

The integrated policy is promising, not conclusively finished. Layer 1 improves
Hit@10 over the previous selective implementation, while Layer 2 preserves the
quality benefit with approximately 57% fewer calls and tokens. However, the
rewrite-benefit F1 of 0.6056 and rerank-benefit F1 of 0.4560 show that action
selection still needs calibration. A final hosted evaluation should replace
gold rewrites with generated rewrites and should measure all actions on a
shared dataset with clarification responses.

The complete machine-readable results are stored in
`research/results/integrated_policy_replay.json`. Reproduce the replay with:

```bash
python research/evaluate_integrated_policy.py
```

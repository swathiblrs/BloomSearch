# Final Research-Question Evaluation

This report evaluates the four official BloomSearch research questions. It
combines reproducible QReCC retrieval, SciFact reranking, signal ablations,
stored hosted-model usage, and the P01/P02 human study. No new hosted-model
call was made.

## RQ1: Action Selection

> To what extent can query ambiguity, BM25 retrieval confidence, and
> Bloom-filter certainty predict whether a search query should be kept,
> rewritten, clarified, or reranked?

BloomSearch implements this as a two-stage decision because rewriting and
reranking are not mutually exclusive. Layer 1 selects `keep`, `rewrite`, or
`clarify`; Layer 2 independently selects `return BM25` or `rerank`.

### Layer 1

| Evaluation | Result |
|---|---:|
| Human three-class Macro-F1 | 0.3818 |
| Keep F1 | 0.3043 |
| Rewrite F1 | 0.7805 |
| Clarify F1 | 0.0606 |
| Participant exact agreement | 56.67% |

The repeated-cross-validation rewrite-benefit ablation produced:

| Signal family | ROC-AUC | Macro-F1 |
|---|---:|---:|
| Linguistic ambiguity/context only | 0.6745 | 0.6252 |
| BM25 confidence only | 0.5239 | 0.5146 |
| Bloom coverage only | 0.5015 | 0.2897 |
| Linguistic + BM25 | 0.6741 | 0.6252 |
| All signals | **0.6748** | **0.6258** |

The combined model is numerically highest, but its gain over linguistic
features alone is negligible. Bloom coverage is effectively at chance.

### Layer 2

The rerank-benefit rule reached Macro-F1 0.5990 and positive-class F1 0.4560.
BM25 score features were the strongest tested predictor (ROC-AUC 0.7047),
ahead of query features (0.6227) and the tested combined set (0.6811).

**Answer:** The current policy predicts rewriting better than clarification or
rerank benefit. Linguistic features drive Layer 1, BM25 scores drive Layer 2,
and the current Bloom-certainty proxy adds no material predictive value.

## RQ2: Retrieval Effectiveness

> How does the proposed selective NLP policy affect Recall@10 and NDCG@10
> compared with BM25 retrieval and always-on LLM processing?

### Layer 1: QReCC

Metrics were recomputed from the checked-in test index and query-to-document
mapping. Thirty-eight stale target mappings removed by index deduplication were
excluded, leaving 2,765 reproducible queries. Each query has one designated
relevant passage, so Recall@10 equals Hit@10.

| Method | Recall@10 | NDCG@10 | Rewrite calls |
|---|---:|---:|---:|
| BM25 original query | 0.3996 | 0.2889 | 0 |
| Always use gold rewrite | **0.8882** | **0.6011** | 2,765 |
| Selective gold-rewrite replay | 0.7617 | 0.5185 | 1,549 |

Selective processing improved Recall@10 over BM25 by 0.3620 (95% bootstrap
CI [0.3443, 0.3801], McNemar `p < 0.001`) and NDCG@10 by 0.2296 (95% CI
[0.2175, 0.2417], paired randomization `p < 0.001`).

Compared with always rewriting, selective processing lost 0.1266 Recall@10
(95% CI [-0.1392, -0.1143], `p < 0.001`) and 0.0826 NDCG@10 (95% CI
[-0.0917, -0.0741], `p < 0.001`).

### Layer 2: SciFact

| Method | Recall@10 | Standard NDCG@10 |
|---|---:|---:|
| BM25 only | 0.7805 | 0.6635 |
| Rerank every query | 0.7805 | 0.7015 |
| Selective reranking | 0.7805 | 0.7023 |

Recall is identical because reranking changes only the order of the same top-10
documents. The prior paired experiment found selective reranking statistically
comparable to reranking every query.

**Answer:** Selective NLP substantially improves over BM25. At Layer 1 it
trades some always-rewrite quality for fewer calls; at Layer 2 it preserves
always-rerank quality and Recall@10.

## RQ3: Intent Preservation

> Does selective query rewriting reduce the harmful-rewrite rate compared with
> rewriting every query?

The stored 2,803-query generated-rewrite experiment provides an automatic
retrieval-harm comparison:

| Condition | Lost Hit@5 | Reduced reciprocal rank |
|---|---:|---:|
| Always rewrite | 17 (0.61% of queries) | 34 (1.21%) |
| Selective rewrite | 17 (0.61% of queries) | 34 (1.21%) |

Every observed retrieval-harm event occurred among rewrites accepted by the
selective policy. Therefore, selection did not reduce retrieval-harm counts in
this stored run.

The same deterministic intent guard was also applied to both rewrite policies
on the 20 QReCC items in the blinded pilot:

| Condition | Rewrites checked | Guard-rejected | Automatic harmful-rewrite rate |
|---|---:|---:|---:|
| Always rewrite | 20 | 5 | **25.00%** |
| Selective rewrite | 10 | 5 | **50.00%** |

The improved guard rejected Q01, Q10, Q11, Q18, and Q25 for dropping important
topic terms, removing continuation meaning, reversing semantic roles, or
removing constraints. These are the same five failure patterns identified in
the participant study. The selective rate was 25 percentage points higher, so
this common automatic semantic proxy also does not support the hypothesis that
selection reduces harmful rewrites.

P01/P02 judged 45% of the shown selected rewrites semantically harmful. They
were not originally shown rewrites for the policy's keep cases, so a human
always-rewrite semantic rate cannot be reconstructed from completed responses.

**Answer:** No. The hypothesis is not supported by either retrieval harm or the
common automatic intent-safety proxy. Bansari's P01/P02 analysis further shows
that selected rewrites still have a high human-rated semantic-harm rate. A
paired human always-rewrite comparison would strengthen this conclusion, but
it is an optional extension rather than missing evidence for the automatic
answer. No P03 is required.

## RQ4: Efficiency

> To what extent does the proposed policy reduce hosted-model calls, input
> tokens, and search latency while preserving retrieval effectiveness?

| Layer and measure | Result versus always-on |
|---|---:|
| Layer 1 rewrite-call reduction | 43.98% |
| Layer 1 estimated input-token reduction | 43.81% |
| Layer 2 reranking-call reduction | 56.67% |
| Layer 2 recorded token reduction | 57.05% |
| Layer 2 average hosted-latency reduction | 60.94% |

Layer 1 input tokens are provider-neutral estimates based on query and context
characters; fixed provider prompts are excluded. Layer 2 uses recorded hosted
token counts.

The latency result is mixed. Layer 2 reduced average hosted latency from 2.93
seconds under always-rerank to 1.14 seconds per query under selective reranking.
Layer 1 local processing increased from 78.88 ms for BM25 to 178.80 ms for the
joint decision and gold-rewrite replay.

**Answer:** Selective NLP reduces hosted calls and tokens by roughly 44% at
Layer 1 and 57% at Layer 2. It preserves Layer 2 retrieval quality, but it does
not make Layer 1 local processing faster and it sacrifices some Layer 1 quality
relative to always rewriting.

## Completion status

All automatic experiments and the original P01/P02 pilot are complete. All
four research questions now have empirical answers. RQ3 is a negative result:
selection did not reduce harm under either automatic definition, while the
human pilot found substantial harm among selected rewrites. Participant
packets and an analyzer remain available for an optional paired-human
replication.

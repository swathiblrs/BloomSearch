# Reranking decision experiment (Task 3)


This folder tests the Layer 2 decision in BloomSearch: after BM25 returns its top 10 results, should the system call the hosted LLM reranker or return the BM25 list as it is?

## Summary of results

All numbers are NDCG@10 on the 300 test queries of BEIR SciFact (5,183 documents).

| Method | API calls | NDCG@10 |
|---|---|---|
| BM25 only | 0 | 0.6635 |
| Rerank every query | 300 | 0.7015 |
| Random 40% of queries reranked (expected) | 120 | 0.6787 |
| Smallest-margin 40% of queries reranked | 120 | 0.7042 |
| Oracle (rerank only when it helps) | 63 | 0.7347 |

- Reranking every query improves NDCG@10 by +0.0381 (95% bootstrap interval +0.0142 to +0.0619).
- Reranking only the queries where BM25's top two scores are closest gives about the same quality as reranking everything, with far fewer API calls.
- On held-out queries (see "Evaluation protocol"), the margin rule scores 0.7023 against 0.7048 for reranking everything, while calling the reranker for 43% of queries. That keeps about 94% of the reranking gain with 57% fewer calls.
- A trained logistic regression predictor (ROC-AUC 0.705) does no better than this one-feature rule.

Other BM25 metrics on the 300 queries: Recall@10 = 0.7805, MRR@10 = 0.6329. Recall@10 does not change after reranking, because the reranker only reorders the same 10 documents.

## Where reranking helps and where it hurts

| Group of queries | Count | What reranking did |
|---|---|---|
| No relevant document in the BM25 top 10 | 60 | Nothing (it cannot help) |
| Relevant documents already in the best order | 158 | Made 22 worse |
| Relevant document present but ranked too low | 82 | Improved 63, made 8 worse |

Overall the reranker helped 63 queries, hurt 30 and left 207 unchanged. The mean gain when it helped was +0.3393 and the mean loss when it hurt was -0.3320.

## Setup

- **Dataset:** BEIR SciFact, test split. 5,183 documents, 300 queries with relevance judgments.
- **Index:** `python -m bloom_search build scifact_docs.json scifact.bloom-index.json --segment-size 50`, which gives 104 segments.
- **BM25:** `SearchIndex.search_bm25(query, limit=10)` from this repository.
- **Reranker:** `openai/gpt-oss-20b` on Groq, called through the OpenAI-compatible `/chat/completions` endpoint with temperature 0. For each query the model sees the 10 BM25 results (title cut to 80 characters, text cut to 150 characters) and returns a relevance score from 0 to 1 for each. Results are sorted by that score.
- **Cost:** 300 of 300 calls succeeded. 439,147 tokens in total, 1,464 per query on average. Average latency was 2.93 s per query, which includes waits caused by rate limits.

## Evaluation protocol

- **BM25 and reranking** involve no training. Both are evaluated on all 300 test queries.
- **Predictor.** Logistic regression with balanced class weights, predicting whether reranking improves NDCG@10 for a query. It is evaluated with 5-fold cross-validation repeated 20 times, with feature scaling fitted inside each fold. ROC-AUC: 0.705 with BM25 score features, 0.623 with query features, 0.681 with both. Chance level is 0.5.
- **Margin rule.** The margin is the BM25 score of result 1 minus the score of result 2. The rule reranks a query when its margin is at or below a cut-off.
- **Held-out check of the margin rule.** The 300 queries are split at random into 210 training and 90 test queries. The cut-off is chosen on the training queries only, as the smallest cut-off that matches the NDCG@10 of reranking all training queries. It is then applied to the test queries. This is repeated over 200 random splits. Average test results: BM25 only 0.6660, rerank every query 0.7048, margin rule 0.7023 with 43% of queries reranked. The difference between the margin rule and reranking everything is -0.0025 on average, and ranges from -0.0400 to +0.0170 across the middle 95% of splits.

## Limitations

- The idea of using the margin was settled after looking at results on all 300 queries, so the held-out check reduces but does not remove selection bias. A clean test would use new queries, for example from the SciFact training split.
- The held-out result varies a lot between splits, because 90 test queries is a small sample. The cut-off chosen in training ranged from 20% to 90% of queries.
- One dataset, one reranker model and one run. The reranker is not perfectly repeatable.
- The reranker sees only the first 150 characters of each document.
- Only the BM25 top 10 is reranked. A relevant document appears in the top 10 for 240 of 300 queries, in the top 50 for 264 and in the top 100 for 268, so reranking a deeper pool could help up to 28 more queries at a higher token cost.
- The per-query NDCG values stored in `bm25_results.json` and `reranking_progress_v2.json` are normalised by the best ordering of the 10 retrieved documents. `analyze_results.py` and `validate_margin_rule.py` convert them to standard NDCG, which normalises by all relevant documents in the judgments. All numbers in this README are standard NDCG.

## Files

| File | Purpose |
|---|---|
| `prepare_scifact.py` | Downloads SciFact and writes the documents, queries and judgments as JSON |
| `evaluate_bm25.py` | Runs BM25 on the 300 queries and saves the top 10 results and scores |
| `evaluate_reranking_v2.py` | Calls the hosted reranker, with retries, progress saving, and token and latency tracking |
| `analyze_results.py` | Computes all metrics, the predictor results and the cost versus quality comparison. No API calls |
| `validate_margin_rule.py` | Held-out check of the margin rule. No API calls |
| `bm25_results.json` | BM25 results for the 300 queries |
| `reranking_progress_v2.json` | Per-query reranking results, tokens and latency |
| `analysis_results.json` | Aggregate results |

## How to reproduce

Run these from this folder, after installing the repository with `python -m pip install -e .` and `pip install beir scikit-learn requests`.

```bash
python prepare_scifact.py
python -m bloom_search build scifact_docs.json scifact.bloom-index.json --segment-size 50
python evaluate_bm25.py

export BLOOMSEARCH_API_BASE="https://api.groq.com/openai/v1"
export BLOOMSEARCH_API_KEY="your-own-key"
export BLOOMSEARCH_MODEL="openai/gpt-oss-20b"
python evaluate_reranking_v2.py

python analyze_results.py
python validate_margin_rule.py
```

The reranking step is the only one that needs an API key. The key is read from the environment and is never written to a file. The reranking script saves progress every 10 queries and resumes when run again.



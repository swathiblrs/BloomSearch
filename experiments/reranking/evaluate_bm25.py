"""Run BM25 on all SciFact queries and compute NDCG@10."""

import json
import math
from bloom_search.index import SearchIndex

# --- Step 1: Load ---
print("Loading index...")
index = SearchIndex.load("scifact.bloom-index.json")

with open("scifact_queries.json") as f:
    queries = json.load(f)

with open("scifact_qrels.json") as f:
    qrels = json.load(f)

print(f"Queries: {len(queries)}")
print(f"Queries with relevance labels: {len(qrels)}")

# --- Step 2: NDCG functions ---

def dcg_at_k(relevances, k):
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        dcg += rel / math.log2(i + 2)
    return dcg

def ndcg_at_k(relevances, k):
    dcg = dcg_at_k(relevances, k)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = dcg_at_k(ideal_relevances, k)
    if idcg == 0:
        return 0.0
    return dcg / idcg

# --- Step 3: Run BM25 on all queries ---
K = 10
results_data = {}
ndcg_scores = []

print(f"\nRunning BM25 search on {len(qrels)} queries...")

count = 0
for query_id, query_text in queries.items():
    if query_id not in qrels:
        continue

    # search_bm25 returns (list of (Document, score), segments_checked)
    results, segments_checked = index.search_bm25(query_text, limit=K)

    relevances = []
    result_doc_ids = []
    bm25_scores = []

    for doc, score in results:
        rel = qrels[query_id].get(doc.id, 0)
        relevances.append(rel)
        result_doc_ids.append(doc.id)
        bm25_scores.append(score)

    ndcg = ndcg_at_k(relevances, K)
    ndcg_scores.append(ndcg)

    results_data[query_id] = {
        "query_text": query_text,
        "doc_ids": result_doc_ids,
        "bm25_scores": bm25_scores,
        "relevances": relevances,
        "ndcg": ndcg,
        "segments_checked": segments_checked
    }

    count += 1
    if count % 50 == 0:
        print(f"  Processed {count} queries...")

# --- Step 4: Results ---
avg_ndcg = sum(ndcg_scores) / len(ndcg_scores)

print(f"\n=== BM25 BASELINE RESULTS ===")
print(f"Queries evaluated: {len(ndcg_scores)}")
print(f"Average NDCG@{K}: {avg_ndcg:.4f}")
print(f"Min NDCG@{K}: {min(ndcg_scores):.4f}")
print(f"Max NDCG@{K}: {max(ndcg_scores):.4f}")

perfect = sum(1 for s in ndcg_scores if s == 1.0)
zero = sum(1 for s in ndcg_scores if s == 0.0)
partial = len(ndcg_scores) - perfect - zero

print(f"\nPerfect (NDCG=1.0): {perfect} queries")
print(f"Zero (NDCG=0.0): {zero} queries")
print(f"Partial: {partial} queries")

with open("bm25_results.json", "w") as f:
    json.dump(results_data, f, indent=2)

print(f"\nSaved detailed results to bm25_results.json")
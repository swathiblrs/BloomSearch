"""Rerank BM25 results - v2 with token/latency tracking."""

import json
import math
import os
import re
import time
import requests

# --- Config ---
API_BASE = os.environ["BLOOMSEARCH_API_BASE"]
API_KEY = os.environ["BLOOMSEARCH_API_KEY"]
MODEL = os.environ["BLOOMSEARCH_MODEL"]
K = 10
PROGRESS_FILE = "reranking_progress_v2.json"

# --- Load data ---
with open("bm25_results.json") as f:
    bm25_results = json.load(f)

with open("scifact_qrels.json") as f:
    qrels = json.load(f)

with open("scifact_docs.json") as f:
    all_docs = {doc["id"]: doc for doc in json.load(f)}

# --- Load progress ---
if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE) as f:
        reranked_data = json.load(f)
    print(f"Resuming from {len(reranked_data)}/{len(bm25_results)} queries")
else:
    reranked_data = {}
    print(f"Starting fresh: {len(bm25_results)} queries")

# --- NDCG functions ---
def dcg_at_k(relevances, k):
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        dcg += rel / math.log2(i + 2)
    return dcg

def ndcg_at_k(relevances, k):
    dcg = dcg_at_k(relevances, k)
    ideal = sorted(relevances, reverse=True)
    idcg = dcg_at_k(ideal, k)
    if idcg == 0:
        return 0.0
    return dcg / idcg

# --- Better JSON extraction ---
def extract_json(text):
    text = text.strip()
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = text.strip()
    if "```" in text:
        matches = re.findall(r'```(?:json)?\s*(.*?)```', text, re.DOTALL)
        if matches:
            text = matches[0].strip()
    try:
        parsed = json.loads(text)
        return parsed.get("results", [])
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{[^{}]*"results"\s*:\s*\[.*?\]\s*\}', text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            return parsed.get("results", [])
        except json.JSONDecodeError:
            pass
    match = re.search(r'\[\s*\{.*?"id".*?\}\s*\]', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None

# --- API call function ---
def call_reranker(query_text, documents):
    doc_list = ""
    for i, (doc_id, title, text) in enumerate(documents):
        short_text = text[:150].replace('"', "'").replace("\n", " ").replace("\\", " ")
        clean_title = title[:80].replace('"', "'").replace("\n", " ").replace("\\", " ")
        doc_list += f"\nDoc {i+1} (ID: {doc_id}): {clean_title}. {short_text}\n"
    
    prompt = (
        f'Search query: "{query_text}"\n\n'
        f'Score each document by relevance (0.0 to 1.0). '
        f'Return ONLY this JSON format, nothing else:\n'
        f'{{"results": [{{"id": "doc_id", "relevance_score": 0.95}}, ...]}}\n\n'
        f'Documents:\n{doc_list}'
    )

    for attempt in range(5):
        try:
            resp = requests.post(
                API_BASE + "/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}"},
                json={
                    "model": MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 2000,
                    "temperature": 0
                },
                timeout=60
            )
            if resp.status_code == 429:
                wait = (attempt + 1) * 15
                print(f"    Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            resp_json = resp.json()
            content = resp_json["choices"][0]["message"]["content"]
            usage = resp_json.get("usage", {})
            
            results = extract_json(content)
            if results is not None:
                return results, usage
            
            if attempt < 4:
                print(f"    Parse failed (attempt {attempt+1}), retrying...")
                time.sleep(3)
                continue
            else:
                print(f"    Parse failed after all attempts")
                return None, {}
                
        except requests.exceptions.HTTPError as e:
            print(f"    API error: {e}")
            return None, {}
        except Exception as e:
            print(f"    Error: {e}")
            if attempt < 4:
                time.sleep(3)
                continue
            return None, {}
    
    print(f"    Failed after 5 retries")
    return None, {}

def save_progress():
    with open(PROGRESS_FILE, "w") as f:
        json.dump(reranked_data, f, indent=2)

# --- Run reranking ---
count = 0
total = len(bm25_results)

for query_id, bm25_info in bm25_results.items():
    if query_id in reranked_data:
        count += 1
        continue
    
    query_text = bm25_info["query_text"]
    doc_ids = bm25_info["doc_ids"]
    bm25_ndcg = bm25_info["ndcg"]
    
    if not doc_ids:
        count += 1
        continue
    
    documents = []
    for doc_id in doc_ids:
        if doc_id in all_docs:
            doc = all_docs[doc_id]
            documents.append((doc_id, doc["title"], doc["text"]))
    
    if not documents:
        count += 1
        continue
    
    start_time = time.time()
    reranked, usage = call_reranker(query_text, documents)
    latency = time.time() - start_time
    
    if reranked is None:
        reranked_data[query_id] = {
            "bm25_ndcg": bm25_ndcg,
            "reranked_ndcg": bm25_ndcg,
            "rerank_helped": False,
            "bm25_scores": bm25_info["bm25_scores"],
            "latency_seconds": round(latency, 2),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "error": True
        }
    else:
        score_map = {}
        for r in reranked:
            rid = str(r.get("id", ""))
            try:
                rscore = float(r.get("relevance_score", 0))
            except (ValueError, TypeError):
                rscore = 0.0
            rscore = max(0.0, min(1.0, rscore))
            if rid in [str(d) for d in doc_ids]:
                score_map[rid] = rscore
        
        reranked_doc_ids = sorted(doc_ids, key=lambda d: score_map.get(str(d), 0), reverse=True)
        
        reranked_relevances = []
        for doc_id in reranked_doc_ids[:K]:
            rel = qrels[query_id].get(doc_id, 0)
            reranked_relevances.append(rel)
        
        reranked_ndcg = ndcg_at_k(reranked_relevances, K)
        
        reranked_data[query_id] = {
            "bm25_ndcg": bm25_ndcg,
            "reranked_ndcg": reranked_ndcg,
            "rerank_helped": reranked_ndcg > bm25_ndcg,
            "bm25_scores": bm25_info["bm25_scores"],
            "latency_seconds": round(latency, 2),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "error": False
        }
    
    count += 1
    
    if count % 10 == 0:
        save_progress()
        done = len(reranked_data)
        errs = sum(1 for d in reranked_data.values() if d["error"])
        good = done - errs
        if good > 0:
            bm25_avg = sum(d["bm25_ndcg"] for d in reranked_data.values() if not d["error"]) / good
            reranked_avg = sum(d["reranked_ndcg"] for d in reranked_data.values() if not d["error"]) / good
            total_tokens = sum(d["total_tokens"] for d in reranked_data.values() if not d["error"])
            avg_latency = sum(d["latency_seconds"] for d in reranked_data.values() if not d["error"]) / good
            print(f"  [{done}/{total}] BM25: {bm25_avg:.4f} | Reranked: {reranked_avg:.4f} | Errors: {errs}/{done} | Tokens: {total_tokens} | Avg latency: {avg_latency:.1f}s | SAVED")
        else:
            print(f"  [{done}/{total}] Errors: {errs}/{done} | SAVED")
    
    time.sleep(8)

# --- Final save ---
save_progress()

# --- Final Results ---
good_data = {k: v for k, v in reranked_data.items() if not v.get("error")}
errs = len(reranked_data) - len(good_data)

if good_data:
    avg_bm25 = sum(d["bm25_ndcg"] for d in good_data.values()) / len(good_data)
    avg_reranked = sum(d["reranked_ndcg"] for d in good_data.values()) / len(good_data)
    helped = sum(1 for d in good_data.values() if d["rerank_helped"])
    hurt = sum(1 for d in good_data.values() if d["reranked_ndcg"] < d["bm25_ndcg"])
    same = len(good_data) - helped - hurt
    total_tokens = sum(d["total_tokens"] for d in good_data.values())
    avg_latency = sum(d["latency_seconds"] for d in good_data.values()) / len(good_data)
    avg_tokens_per_query = total_tokens / len(good_data)

    print(f"\n=== RERANKING RESULTS ===")
    print(f"Total queries: {len(reranked_data)}")
    print(f"Successful: {len(good_data)}")
    print(f"Errors: {errs} ({errs/len(reranked_data)*100:.1f}%)")
    print(f"\nAverage BM25 NDCG@{K}:     {avg_bm25:.4f}")
    print(f"Average Reranked NDCG@{K}: {avg_reranked:.4f}")
    print(f"Improvement:                {avg_reranked - avg_bm25:+.4f}")
    print(f"\nReranking helped: {helped} queries ({helped/len(good_data)*100:.1f}%)")
    print(f"Reranking hurt:   {hurt} queries ({hurt/len(good_data)*100:.1f}%)")
    print(f"No change:        {same} queries ({same/len(good_data)*100:.1f}%)")
    print(f"\n=== COST & LATENCY ===")
    print(f"Total tokens used: {total_tokens}")
    print(f"Avg tokens per query: {avg_tokens_per_query:.0f}")
    print(f"Avg latency per query: {avg_latency:.2f}s")
    print(f"Model: {MODEL}")

with open("reranking_results.json", "w") as f:
    json.dump(reranked_data, f, indent=2)

print(f"\nSaved final results to reranking_results.json")
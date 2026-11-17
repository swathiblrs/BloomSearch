"""Full analysis of the reranking experiment. No API calls.

Reads:  bm25_results.json, scifact_qrels.json, scifact_queries.json,
        scifact_docs.json, reranking_progress_v2.json
Writes: analysis_results.json
"""

import json
import math
import re

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

K = 10
RERANK_FILE = "reranking_progress_v2.json"
SEED = 42


def load(path):
    with open(path) as f:
        return json.load(f)


def dcg(rels, k=K):
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels[:k]))


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def header(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


bm25 = load("bm25_results.json")
qrels = load("scifact_qrels.json")
docs = {d["id"]: d for d in load("scifact_docs.json")}
rerank_all = load(RERANK_FILE)
rerank = {q: v for q, v in rerank_all.items() if not v.get("error") and q in bm25}
out = {"k": K, "rerank_file": RERANK_FILE}

# ---------------------------------------------------------------------------
# 1. Corrected metrics
# The earlier scripts divided DCG by the ideal ordering of the 10 retrieved
# documents. Standard NDCG divides by the ideal ordering of ALL relevant
# documents in the qrels. Both runs ranked the same 10 documents, so the
# standard value is: old_value * IDCG(retrieved set) / IDCG(all relevant).
# ---------------------------------------------------------------------------
def idcg_set(qid):
    return dcg(sorted(bm25[qid]["relevances"], reverse=True))


def idcg_full(qid):
    return dcg(sorted(qrels.get(qid, {}).values(), reverse=True))


def correct(qid, old_value):
    full = idcg_full(qid)
    return old_value * idcg_set(qid) / full if full > 0 else 0.0


def bm25_metrics(qids):
    ndcg, recall, mrr = [], [], []
    for q in qids:
        rels = bm25[q]["relevances"][:K]
        n_rel = sum(1 for v in qrels.get(q, {}).values() if v > 0)
        ndcg.append(correct(q, bm25[q]["ndcg"]))
        recall.append(sum(1 for r in rels if r > 0) / n_rel if n_rel else 0.0)
        first = next((i for i, r in enumerate(rels) if r > 0), None)
        mrr.append(1.0 / (first + 1) if first is not None else 0.0)
    return float(np.mean(ndcg)), float(np.mean(recall)), float(np.mean(mrr))


header("1. BM25 BASELINE (standard NDCG, IDCG from all relevant documents)")
all_q = list(bm25.keys())
old_all = float(np.mean([bm25[q]["ndcg"] for q in all_q]))
n_all, r_all, m_all = bm25_metrics(all_q)
print(f"All {len(all_q)} queries")
print(f"  NDCG@{K} as first reported : {old_all:.4f}")
print(f"  NDCG@{K} standard          : {n_all:.4f}")
print(f"  Recall@{K}                 : {r_all:.4f}")
print(f"  MRR@{K}                    : {m_all:.4f}")
out["bm25_all"] = {"queries": len(all_q), "ndcg_first_reported": old_all,
                   "ndcg": n_all, "recall": r_all, "mrr": m_all}

# ---------------------------------------------------------------------------
# 2. Reranking on the evaluated subset
# ---------------------------------------------------------------------------
qids = list(rerank.keys())
n = len(qids)
bm = np.array([correct(q, rerank[q]["bm25_ndcg"]) for q in qids])
rr = np.array([correct(q, rerank[q]["reranked_ndcg"]) for q in qids])
diff = rr - bm
helped = diff > 1e-9
hurt = diff < -1e-9

header(f"2. RERANKING ON {n} EVALUATED QUERIES "
       f"({len(rerank_all) - n} errors excluded)")
print(f"  BM25 NDCG@{K}      : {bm.mean():.4f}")
print(f"  Reranked NDCG@{K}  : {rr.mean():.4f}")
print(f"  Difference         : {diff.mean():+.4f}")
print(f"  Helped / same / hurt: {helped.sum()} / {n - helped.sum() - hurt.sum()} / {hurt.sum()}")
if helped.any():
    print(f"  Mean gain when helped: {diff[helped].mean():+.4f}")
if hurt.any():
    print(f"  Mean loss when hurt  : {diff[hurt].mean():+.4f}")
print(f"  Recall@{K} is identical before and after: the reranker only reorders"
      f" the same {K} documents.")

rng = np.random.default_rng(SEED)
boot = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(10000)])
lo, hi = np.percentile(boot, [2.5, 97.5])
print(f"  95% bootstrap interval for the difference: [{lo:+.4f}, {hi:+.4f}]")
print("  " + ("Interval excludes zero: the gain is unlikely to be chance."
              if lo > 0 else "Interval includes zero: the gain is NOT yet established."))

tokens = np.array([rerank[q].get("total_tokens", 0) for q in qids], dtype=float)
lat = np.array([rerank[q].get("latency_seconds", 0) for q in qids], dtype=float)
print(f"  Tokens per reranked query: {tokens.mean():.0f}   "
      f"Latency per query: {lat.mean():.2f}s (includes rate-limit waits)")
out["reranking"] = {
    "queries": n, "errors": len(rerank_all) - n,
    "bm25_ndcg": float(bm.mean()), "reranked_ndcg": float(rr.mean()),
    "difference": float(diff.mean()), "ci95": [float(lo), float(hi)],
    "helped": int(helped.sum()), "hurt": int(hurt.sum()),
    "same": int(n - helped.sum() - hurt.sum()),
    "avg_tokens": float(tokens.mean()), "avg_latency_s": float(lat.mean()),
}

# ---------------------------------------------------------------------------
# 3. Headroom: where can reordering 10 documents help at all?
# ---------------------------------------------------------------------------
header("3. HEADROOM: WHERE CAN RERANKING THE TOP 10 HELP?")
no_rel = np.array([not any(r > 0 for r in bm25[q]["relevances"][:K]) for q in qids])
ideal_order = np.array([
    (not no_rel[i]) and abs(dcg(bm25[q]["relevances"]) - idcg_set(q)) < 1e-9
    for i, q in enumerate(qids)])
fixable = ~no_rel & ~ideal_order
print(f"  No relevant document in top {K} (cannot be helped)  : {no_rel.sum()}")
print(f"  Relevant documents already in best order (cannot help): {ideal_order.sum()}")
print(f"  Relevant document present but ranked too low (fixable): {fixable.sum()}")
print(f"  Of the fixable queries, reranking helped {int((helped & fixable).sum())}"
      f" and hurt {int((hurt & fixable).sum())}")
print(f"  Of the already-best queries, reranking hurt {int((hurt & ideal_order).sum())}")
out["headroom"] = {"no_relevant_in_top_k": int(no_rel.sum()),
                   "already_ideal": int(ideal_order.sum()),
                   "fixable": int(fixable.sum()),
                   "helped_of_fixable": int((helped & fixable).sum()),
                   "hurt_of_ideal": int((hurt & ideal_order).sum())}

try:
    from bloom_search.index import SearchIndex
    index = SearchIndex.load("scifact.bloom-index.json")
    queries = load("scifact_queries.json")
    depths = [10, 20, 50, 100]
    found = {d: 0 for d in depths}
    rec = {d: [] for d in depths}
    for q in all_q:
        results, _ = index.search_bm25(queries[q], limit=max(depths))
        ids = [doc.id for doc, _ in results]
        rel_ids = {d for d, v in qrels.get(q, {}).items() if v > 0}
        for d in depths:
            hit = len(rel_ids & set(ids[:d]))
            found[d] += 1 if hit else 0
            rec[d].append(hit / len(rel_ids) if rel_ids else 0.0)
    print(f"\n  BM25 candidate depth on all {len(all_q)} queries:")
    print(f"  {'Depth':<8}{'Queries with a relevant doc':<30}{'Recall':<10}")
    for d in depths:
        print(f"  {d:<8}{found[d]:<30}{np.mean(rec[d]):<10.4f}")
    out["depth"] = {str(d): {"queries_with_relevant": found[d],
                             "recall": float(np.mean(rec[d]))} for d in depths}
except Exception as e:  # analysis continues without the depth table
    print(f"\n  (Depth table skipped: {type(e).__name__}: {e})")

# ---------------------------------------------------------------------------
# 4. Predictor: can we tell in advance which queries to rerank?
# ---------------------------------------------------------------------------
header("4. PREDICTOR (5-fold CV repeated 20 times, scaling fitted inside each fold)")
N_docs = len(docs)
df = {}
doc_tokens = {}
for d_id, d in docs.items():
    toks = set(tokenize(d["title"] + " " + d["text"]))
    doc_tokens[d_id] = toks
    for t in toks:
        df[t] = df.get(t, 0) + 1


def idf(t):
    return math.log((N_docs - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5) + 1)


score_names = ["top_score", "margin_1_2", "margin_2_3", "score_range",
               "score_std", "top_concentration"]
query_names = ["query_len", "mean_idf", "max_idf", "coverage_top1", "coverage_gap"]
Xs, Xq, keep = [], [], []
for i, q in enumerate(qids):
    s = bm25[q]["bm25_scores"]
    if len(s) < 3:
        continue
    q_toks = tokenize(bm25[q]["query_text"])
    uniq = set(q_toks) or {""}
    ids = bm25[q]["doc_ids"]
    cov = [len(uniq & doc_tokens.get(d, set())) / len(uniq) for d in ids[:5]]
    idfs = [idf(t) for t in uniq]
    Xs.append([s[0], s[0] - s[1], s[1] - s[2], s[0] - s[-1],
               float(np.std(s)), s[0] / sum(s) if sum(s) else 0.0])
    Xq.append([len(q_toks), float(np.mean(idfs)), max(idfs), cov[0],
               cov[0] - float(np.mean(cov[1:])) if len(cov) > 1 else 0.0])
    keep.append(i)
keep = np.array(keep)
Xs, Xq = np.array(Xs), np.array(Xq)
y = helped[keep].astype(int)
print(f"  {len(y)} queries, {y.sum()} where reranking helped ({y.mean() * 100:.1f}%)")


def oof_probs(X, y):
    """Out-of-fold probabilities averaged over repeats, plus per-repeat AUC."""
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=SEED)
    total = np.zeros(len(y))
    aucs, aps = [], []
    fold_p = np.zeros(len(y))
    for j, (tr, te) in enumerate(cv.split(X, y)):
        model = make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=2000, class_weight="balanced"))
        model.fit(X[tr], y[tr])
        fold_p[te] = model.predict_proba(X[te])[:, 1]
        if (j + 1) % 5 == 0:
            aucs.append(roc_auc_score(y, fold_p))
            aps.append(average_precision_score(y, fold_p))
            total += fold_p
            fold_p = np.zeros(len(y))
    return total / 20, np.array(aucs), np.array(aps)


probs = {}
out["predictor"] = {"queries": int(len(y)), "positives": int(y.sum())}
if y.sum() >= 5 and (len(y) - y.sum()) >= 5:
    print(f"\n  {'Feature set':<28}{'ROC-AUC':<18}{'Avg precision':<18}")
    print(f"  {'Chance level':<28}{'0.500':<18}{y.mean():<18.3f}")
    for name, X in [("BM25 scores only", Xs), ("Query features only", Xq),
                    ("Scores + query features", np.hstack([Xs, Xq]))]:
        p, aucs, aps = oof_probs(X, y)
        probs[name] = p
        print(f"  {name:<28}{aucs.mean():.3f} +/- {aucs.std():.3f}   "
              f"{aps.mean():.3f} +/- {aps.std():.3f}")
        out["predictor"][name] = {"auc": float(aucs.mean()), "auc_std": float(aucs.std()),
                                  "avg_precision": float(aps.mean())}
    print("  ROC-AUC: 0.5 = guessing, 1.0 = perfect ranking of queries.")
else:
    print("  Too few examples in one class to train a predictor.")

# ---------------------------------------------------------------------------
# 5. Cost versus quality: the comparison that decides whether selection works
# ---------------------------------------------------------------------------
header("5. COST VERSUS QUALITY (same queries for every method)")
bm_k, rr_k, tok_k = bm[keep], rr[keep], tokens[keep]
gain = rr_k.mean() - bm_k.mean()
oracle = np.maximum(bm_k, rr_k)
rows = [("BM25 only", 0, bm_k.mean()),
        ("Rerank every query", len(y), rr_k.mean()),
        ("Oracle (rerank only if it helps)", int((rr_k > bm_k + 1e-9).sum()), oracle.mean())]
print(f"  {'Method':<36}{'API calls':<12}{'NDCG@' + str(K):<10}")
for name, calls, val in rows:
    print(f"  {name:<36}{calls:<12}{val:<10.4f}")
out["comparison"] = {name: {"calls": int(c), "ndcg": float(v)} for name, c, v in rows}


def at_budget(order, frac):
    chosen = order[:int(round(frac * len(order)))]
    val = bm_k.copy()
    val[chosen] = rr_k[chosen]
    return float(val.mean()), float(tok_k[chosen].sum())


margin_order = np.argsort(Xs[:, 1])            # smallest margin first
orders = {"Smallest-margin rule": margin_order}
for name, p in probs.items():
    orders["Predictor: " + name] = np.argsort(-p)

print(f"\n  NDCG@{K} when only a share of queries is reranked")
budgets = [0.2, 0.4, 0.6, 0.8]
print(f"  {'Selection method':<36}" + "".join(f"{int(b * 100)}%".ljust(9) for b in budgets))
rand = [bm_k.mean() + b * gain for b in budgets]
print(f"  {'Random selection (expected)':<36}" + "".join(f"{v:.4f}".ljust(9) for v in rand))
out["budget_curve"] = {"budgets": budgets, "Random selection": rand}
for name, order in orders.items():
    vals = [at_budget(order, b)[0] for b in budgets]
    print(f"  {name:<36}" + "".join(f"{v:.4f}".ljust(9) for v in vals))
    out["budget_curve"][name] = vals
print(f"  Rerank every query = {rr_k.mean():.4f}. A selection method is useful only if its"
      f"\n  row is clearly above the random row.")

with open("analysis_results.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nSaved analysis_results.json")
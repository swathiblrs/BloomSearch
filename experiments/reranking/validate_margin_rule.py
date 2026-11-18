"""Train/test check of the smallest-margin rule. No API calls."""
import json
import math
import numpy as np

K = 10
bm25 = json.load(open("bm25_results.json"))
qrels = json.load(open("scifact_qrels.json"))
rerank = json.load(open("reranking_progress_v2.json"))


def dcg(rels):
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels[:K]))


def standard(q, value):
    full = dcg(sorted(qrels[q].values(), reverse=True))
    found = dcg(sorted(bm25[q]["relevances"], reverse=True))
    return value * found / full if full else 0.0


qs = [q for q, v in rerank.items() if not v["error"] and len(bm25[q]["bm25_scores"]) >= 2]
bm = np.array([standard(q, rerank[q]["bm25_ndcg"]) for q in qs])
rr = np.array([standard(q, rerank[q]["reranked_ndcg"]) for q in qs])
margin = np.array([bm25[q]["bm25_scores"][0] - bm25[q]["bm25_scores"][1] for q in qs])
n = len(qs)
n_train = int(round(0.7 * n))


def choose_cutoff(train):
    """Smallest margin cut-off on the training queries that matches reranking all of them."""
    order = train[np.argsort(margin[train], kind="stable")]
    target = rr[train].mean()
    for share in np.arange(0.1, 1.01, 0.1):
        chosen = order[:int(round(share * len(train)))]
        val = bm[train].mean() + (rr[chosen] - bm[chosen]).sum() / len(train)
        if val >= target - 1e-12:
            return margin[chosen[-1]], share
    return margin[order[-1]], 1.0


def run(seed):
    perm = np.random.default_rng(seed).permutation(n)
    train, test = perm[:n_train], perm[n_train:]
    cutoff, share = choose_cutoff(train)
    call = margin[test] <= cutoff
    selective = np.where(call, rr[test], bm[test]).mean()
    return cutoff, share, call.mean(), selective, bm[test].mean(), rr[test].mean()


cutoff, share, calls, sel, b, r = run(42)
print(f"ONE SPLIT (seed 42): {n_train} training queries, {n - n_train} test queries")
print(f"  Chosen on training queries: rerank the {share * 100:.0f}% with margin <= {cutoff:.2f}")
print(f"  On the {n - n_train} test queries (never used to choose the cut-off):")
print(f"    BM25 only            NDCG@10 = {b:.4f}   API calls = 0%")
print(f"    Rerank every query   NDCG@10 = {r:.4f}   API calls = 100%")
print(f"    Margin rule          NDCG@10 = {sel:.4f}   API calls = {calls * 100:.0f}%")

res = np.array([run(s) for s in range(200)])
diff = res[:, 3] - res[:, 5]
lo, hi = np.percentile(diff, [2.5, 97.5])
print(f"\n200 DIFFERENT RANDOM SPLITS (averages on the test queries)")
print(f"  BM25 only            NDCG@10 = {res[:, 4].mean():.4f}")
print(f"  Rerank every query   NDCG@10 = {res[:, 5].mean():.4f}")
print(f"  Margin rule          NDCG@10 = {res[:, 3].mean():.4f}   API calls = {res[:, 2].mean() * 100:.0f}%")
print(f"  Margin rule minus rerank-all: {diff.mean():+.4f}  (middle 95% of splits: {lo:+.4f} to {hi:+.4f})")
print(f"  Share chosen in training: average {res[:, 1].mean() * 100:.0f}%, "
      f"from {res[:, 1].min() * 100:.0f}% to {res[:, 1].max() * 100:.0f}%")
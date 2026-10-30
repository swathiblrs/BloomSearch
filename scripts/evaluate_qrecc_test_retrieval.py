
import json
from pathlib import Path

from bloom_search.index import SearchIndex

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/qrecc"

with open(DATA_DIR / "test_3000.json") as f:
    test_data = json.load(f)

with open(DATA_DIR / "qrecc_test_mapping.json") as f:
    mapping = json.load(f)

index = SearchIndex.load(
    DATA_DIR / "qrecc_test_index.json"
)


def evaluate_query(query, target_id):
    results, _ = index.search_bm25(query, limit=5)

    for rank, (document, score) in enumerate(results, 1):
        if document.id == target_id:
            return 1, 1 / rank

    return 0, 0.0


original_hits = 0
rewrite_hits = 0

original_mrr = 0.0
rewrite_mrr = 0.0

evaluated = 0

for i, row in enumerate(test_data):
    target_id = mapping.get(str(i))

    if not target_id:
        continue

    original = row["original_query"]
    rewritten = row["gold_rewrite"]

    original_hit, original_rr = evaluate_query(
        original, target_id
    )

    rewrite_hit, rewrite_rr = evaluate_query(
        rewritten, target_id
    )

    original_hits += original_hit
    rewrite_hits += rewrite_hit

    original_mrr += original_rr
    rewrite_mrr += rewrite_rr

    evaluated += 1

    if evaluated % 100 == 0:
        print(f"Evaluated {evaluated} queries")

print("\nQReCC Retrieval Evaluation")
print("--------------------------")
print("Evaluated queries:", evaluated)

print(
    "Original Hit@5:",
    f"{original_hits / evaluated * 100:.2f}%"
)
print(
    "Rewritten Hit@5:",
    f"{rewrite_hits / evaluated * 100:.2f}%"
)

print(
    "Original MRR@5:",
    f"{original_mrr / evaluated:.4f}"
)
print(
    "Rewritten MRR@5:",
    f"{rewrite_mrr / evaluated:.4f}"
)

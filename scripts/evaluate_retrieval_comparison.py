
import json
from pathlib import Path

from bloom_search.index import SearchIndex
from bloom_search.rewrite import rewrite_query

ROOT = Path(__file__).resolve().parents[1]

INDEX_PATH = ROOT / "data/qrecc/qrecc_retrieval_index.json"
TEST_PATH = ROOT / "data/qrecc/test_3000.json"
OUTPUT_PATH = ROOT / "data/qrecc/retrieval_comparison.json"

LIMIT = 5
NUM_EXAMPLES = 20


def retrieve(index, query):
    results, _ = index.search_bm25(query, limit=LIMIT)

    return [
        {
            "document_id": document.id,
            "score": float(score),
        }
        for document, score in results
    ]


def main():
    index = SearchIndex.load(INDEX_PATH)

    with open(TEST_PATH, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = data.get("data", data.get("results", []))

    comparisons = []

    for i, row in enumerate(data[:NUM_EXAMPLES], start=1):
        original = row["original_query"]
        context = row.get("context", [])

        result = rewrite_query(original, context)

        rewritten = result.rewritten_query

        original_results = retrieve(index, original)
        rewritten_results = retrieve(index, rewritten)

        original_ids = {
            item["document_id"] for item in original_results
        }
        rewritten_ids = {
            item["document_id"] for item in rewritten_results
        }

        comparisons.append({
            "original_query": original,
            "rewritten_query": rewritten,
            "accepted": result.accepted,
            "original_results": original_results,
            "rewritten_results": rewritten_results,
            "overlap_at_5": len(original_ids & rewritten_ids),
        })

        print(f"\nExample {i}")
        print("Original:", original)
        print("Rewritten:", rewritten)
        print("Accepted:", result.accepted)
        print("Original retrieved:", len(original_results))
        print("Rewritten retrieved:", len(rewritten_results))
        print("Overlap@5:", len(original_ids & rewritten_ids))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(comparisons, f, indent=2)

    print("\nRetrieval comparison completed!")
    print("Results saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()

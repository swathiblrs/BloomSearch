

import json
from pathlib import Path

from bloom_search.index import SearchIndex
from bloom_search.rewrite import rewrite_query

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/qrecc"

TEST_PATH = DATA / "test_3000.json"
INDEX_PATH = DATA / "qrecc_test_index.json"
MAPPING_PATH = DATA / "qrecc_test_mapping.json"
OUTPUT_PATH = DATA / "system_rewrite_evaluation_final.json"

K = 5


def reciprocal_rank(results, relevant_id):
    for rank, (document, _) in enumerate(results, start=1):
        if document.id == relevant_id:
            return 1.0 / rank
    return 0.0


def main():
    test_data = json.loads(TEST_PATH.read_text())
    mapping = json.loads(MAPPING_PATH.read_text())
    index = SearchIndex.load(INDEX_PATH)

    total = 0
    original_hits = 0
    rewritten_hits = 0
    original_mrr = 0.0
    rewritten_mrr = 0.0
    accepted_count = 0
    changed_count = 0
    records = []

    for i, row in enumerate(test_data):
        relevant_id = mapping.get(str(i))

        if relevant_id is None:
            continue

        original = row["original_query"]
        context = row.get("context", [])

        result = rewrite_query(
            original,
            context,
            prompt_version="improved",
            )

        # Use the original query if the rewrite is rejected.
        rewritten = (
            result.rewritten_query
            if result.accepted
            else original
        )

        if result.accepted:
            accepted_count += 1

        if rewritten != original:
            changed_count += 1

        original_results, _ = index.search_bm25(
            original, limit=K
        )

        rewritten_results, _ = index.search_bm25(
            rewritten, limit=K
        )

        original_rr = reciprocal_rank(
            original_results, relevant_id
        )
        rewritten_rr = reciprocal_rank(
            rewritten_results, relevant_id
        )

        original_hits += int(original_rr > 0)
        rewritten_hits += int(rewritten_rr > 0)

        original_mrr += original_rr
        rewritten_mrr += rewritten_rr

        total += 1

        records.append({
            "query_index": i,
            "original_query": original,
            "rewritten_query": rewritten,
            "accepted": result.accepted,
            "problems": result.problems,
            "relevant_document_id": relevant_id,
            "original_hit": original_rr > 0,
            "rewritten_hit": rewritten_rr > 0,
            "original_rr": original_rr,
            "rewritten_rr": rewritten_rr,
        })

        if total % 100 == 0:
            print(f"Evaluated {total} queries")

    metrics = {
        "evaluated_queries": total,
        "accepted_rewrites": accepted_count,
        "changed_queries": changed_count,
        "original_hit_at_5": original_hits / total,
        "system_hit_at_5": rewritten_hits / total,
        "original_mrr_at_5": original_mrr / total,
        "system_mrr_at_5": rewritten_mrr / total,
    }

    OUTPUT_PATH.write_text(
        json.dumps({
            "metrics": metrics,
            "records": records
        }, indent=2)
    )

    print("\nSystem Rewrite Evaluation")
    print("-------------------------")
    print("Evaluated queries:", total)
    print("Accepted rewrites:", accepted_count)
    print("Changed queries:", changed_count)
    print(f"Original Hit@5: {metrics['original_hit_at_5']:.2%}")
    print(f"System Hit@5: {metrics['system_hit_at_5']:.2%}")
    print(f"Original MRR@5: {metrics['original_mrr_at_5']:.4f}")
    print(f"System MRR@5: {metrics['system_mrr_at_5']:.4f}")
    print("Results saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()

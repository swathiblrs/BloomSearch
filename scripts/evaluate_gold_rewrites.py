import json
from pathlib import Path

from bloom_search.index import Document, SearchIndex


ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "data" / "qrecc" / "test_3000.json"
OUTPUT_FILE = ROOT / "data" / "qrecc" / "test_3000_results.json"
TOP_K = 10


def hit_at_k(index: SearchIndex, query: str, target_id: str) -> bool:
    results, _ = index.search_bm25(query, limit=TOP_K)
    returned_ids = {document.id for document, _score in results}
    return target_id in returned_ids


def main() -> None:
    rows = json.loads(INPUT_FILE.read_text(encoding="utf-8"))

    documents = []
    usable_rows = []

    for row in rows:
        answer = row["answer"].strip()
        original = row["original_query"].strip()
        rewrite = row["gold_rewrite"].strip()

        if not answer or not original or not rewrite:
            continue

        document_id = f"{row['conversation_id']}-{row['turn_id']}"
        documents.append(
            Document(
                id=document_id,
                title=f"QReCC answer {document_id}",
                text=answer,
                url=row["answer_url"],
            )
        )
        usable_rows.append((row, document_id))

    index = SearchIndex.build(documents, segment_size=25)

    results = []
    original_hits = 0
    rewrite_hits = 0
    rewrite_wins = 0

    for row, document_id in usable_rows:
        original_hit = hit_at_k(index, row["original_query"], document_id)
        rewrite_hit = hit_at_k(index, row["gold_rewrite"], document_id)

        original_hits += original_hit
        rewrite_hits += rewrite_hit
        rewrite_wins += rewrite_hit and not original_hit

        results.append(
            {
                "conversation_id": row["conversation_id"],
                "turn_id": row["turn_id"],
                "original_query": row["original_query"],
                "gold_rewrite": row["gold_rewrite"],
                "original_hit_at_10": original_hit,
                "rewrite_hit_at_10": rewrite_hit,
                "rewrite_helped": rewrite_hit and not original_hit,
            }
        )

    total = len(results)
    report = {
        "examples": total,
        "top_k": TOP_K,
        "original_hit_at_10": original_hits / total if total else 0,
        "gold_rewrite_hit_at_10": rewrite_hits / total if total else 0,
        "rewrite_only_wins": rewrite_wins,
        "results": results,
    }

    OUTPUT_FILE.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Examples evaluated: {total}")
    print(f"Original Hit@10: {report['original_hit_at_10']:.1%}")
    print(f"Gold rewrite Hit@10: {report['gold_rewrite_hit_at_10']:.1%}")
    print(f"Rewrite-only wins: {rewrite_wins}")
    print(f"Saved report: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
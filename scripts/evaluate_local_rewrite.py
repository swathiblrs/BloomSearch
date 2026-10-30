
import json
import argparse
from pathlib import Path

from bloom_search.rewrite import rewrite_query

ROOT = Path(__file__).resolve().parents[1]

TEST_FILE = ROOT / "data" / "qrecc" / "test_3000.json"
OUTPUT_FILE = ROOT / "data" / "qrecc" / "local_rewrite_results.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    with TEST_FILE.open(encoding="utf-8") as f:
        data = json.load(f)

    # Handle a JSON list or a dictionary containing examples.
    if isinstance(data, dict):
        data = data.get("data", data.get("results", []))

    results = []

    for index, row in enumerate(data[:args.limit], start=1):
        original = row.get("original_query", row.get("question", ""))
        context = row.get("context", [])

        if isinstance(context, str):
            context = [context]

        if not original:
            continue

        try:
            result = rewrite_query(
                original,
                context=context,
                force_rewrite=True,
            )

            results.append({
                "original_query": original,
                "rewritten_query": result.rewritten_query,
                "accepted": result.accepted,
                "problems": result.problems,
            })

            print(f"\n[{index}]")
            print(f"Original: {original}")
            print(f"Context: {context}")
            print(f"Final: {result.rewritten_query}")
            print(f"Accepted: {result.accepted}")
            print(f"Problems: {result.problems}")

        except Exception as exc:
            print(f"[{index}] Error: {exc}", flush=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    total = len(results)
    accepted = sum(r["accepted"] for r in results)

    print("\nLocal Rewrite Evaluation")
    print("------------------------")
    print(f"Queries evaluated: {total}")
    print(f"Accepted rewrites: {accepted}")
    print(f"Rejected rewrites: {total - accepted}")
    print(f"Acceptance rate: {accepted / total:.2%}" if total else "No results")
    print(f"Saved results: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

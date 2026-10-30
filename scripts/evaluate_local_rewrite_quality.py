
import json
import re
from pathlib import Path

from bloom_search.rewrite import rewrite_query

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data/qrecc/test_3000.json"
OUTPUT_FILE = ROOT / "data/qrecc/local_rewrite_quality.json"

def tokens(text):
    return re.findall(r"\w+", text.lower())

def token_f1(predicted, reference):
    from collections import Counter

    p = Counter(tokens(predicted))
    r = Counter(tokens(reference))

    overlap = sum((p & r).values())
    if not overlap:
        return 0.0

    precision = overlap / sum(p.values())
    recall = overlap / sum(r.values())
    return 2 * precision * recall / (precision + recall)

def main():
    with open(DATA_FILE, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = data.get("data", data.get("results", []))

    results = []

    for i, row in enumerate(data[:100], 1):
        original = row["original_query"]
        gold = row["gold_rewrite"]
        context = row.get("context", [])

        result = rewrite_query(original, context)

        results.append({
            "original": original,
            "gold": gold,
            "predicted": result.rewritten_query,
            "accepted": result.accepted,
            "exact_match": (
                result.rewritten_query.strip().lower()
                == gold.strip().lower()
            ),
            "token_f1": token_f1(
                result.rewritten_query, gold
            ),
            "problems": result.problems,
        })

        print(f"Completed {i}/100")

    n = len(results)
    if not n:
        raise ValueError("No evaluation examples found.")

    print("\nLocal Rewrite Quality")
    print("---------------------")
    print(f"Examples: {n}")
    print(f"Acceptance rate: {sum(r['accepted'] for r in results)/n:.2%}")
    print(f"Exact match: {sum(r['exact_match'] for r in results)/n:.2%}")
    print(f"Average token F1: {sum(r['token_f1'] for r in results)/n:.2%}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    print(f"Saved results: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

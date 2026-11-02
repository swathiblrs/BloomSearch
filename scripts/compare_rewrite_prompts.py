
import json
import re
from collections import Counter
from pathlib import Path

from bloom_search.rewrite import rewrite_query

DATA_FILE = Path("data/qrecc/training_12000.json")

with DATA_FILE.open() as f:
    data = json.load(f)

if isinstance(data, dict):
    data = data.get("data", data.get("results", []))

examples = data[:30]


def tokenize(text):
    return re.findall(r"\w+", text.lower())


def token_f1(predicted, gold):
    p = Counter(tokenize(predicted))
    g = Counter(tokenize(gold))

    overlap = sum((p & g).values())

    if not p and not g:
        return 1.0

    if not p or not g or overlap == 0:
        return 0.0

    precision = overlap / sum(p.values())
    recall = overlap / sum(g.values())

    return 2 * precision * recall / (precision + recall)


def normalize(text):
    return " ".join(tokenize(text))


for version in ["baseline", "improved"]:
    accepted = 0
    exact_matches = 0
    f1_scores = []

    print(f"\nTesting prompt: {version}")

    for row in examples:
        result = rewrite_query(
            original_query=row["original_query"],
            context=row.get("context", []),
            force_rewrite=True,
            prompt_version=version,
        )

        predicted = result.rewritten_query
        gold = row["gold_rewrite"]

        accepted += int(result.accepted)
        exact_matches += int(
            normalize(predicted) == normalize(gold)
        )
        f1_scores.append(token_f1(predicted, gold))

    total = len(examples)

    print("Examples:", total)
    print("Accepted:", accepted)
    print(f"Acceptance rate: {accepted / total:.2%}")
    print(f"Exact match: {exact_matches / total:.2%}")
    print(f"Average token F1: {sum(f1_scores) / total:.2%}")


import json
from pathlib import Path

from bloom_search.index import Document, SearchIndex

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "qrecc"

INPUT_FILE = DATA_DIR / "qrecc-training.json"
OUTPUT_FILE = DATA_DIR / "qrecc_retrieval_index.json"

with open(INPUT_FILE, encoding="utf-8") as f:
    records = json.load(f)

documents = []
seen = set()

for row in records:
    answer = (row.get("Answer") or "").strip()
    url = (row.get("Answer_URL") or "").strip()

    if not answer or not url:
        continue

    # Avoid duplicate answer passages from the same URL.
    key = (url, answer)

    if key in seen:
        continue

    seen.add(key)

    documents.append(
        Document(
            id=f"qrecc-{len(documents)}",
            title=url,
            text=answer,
            url=url,
        )
    )

print("Documents prepared:", len(documents))

index = SearchIndex.build(documents, error_rate=0.01)
index.save(OUTPUT_FILE)

print("Index saved to:", OUTPUT_FILE)

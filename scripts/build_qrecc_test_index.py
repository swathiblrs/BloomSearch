
import json
from pathlib import Path

from bloom_search.index import Document, SearchIndex

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "data/qrecc/test_3000.json"
OUTPUT_FILE = ROOT / "data/qrecc/qrecc_test_index.json"
MAPPING_FILE = ROOT / "data/qrecc/qrecc_test_mapping.json"

with open(INPUT_FILE, encoding="utf-8") as f:
    data = json.load(f)

documents = []
mapping = {}

for i, row in enumerate(data):
    answer = row.get("answer", "").strip()

    if not answer:
        continue

    doc_id = f"test-{i}"

    documents.append(
        Document(
            id=doc_id,
            title=row.get("answer_url", ""),
            text=answer,
            url=row.get("answer_url", ""),
        )
    )

    mapping[str(i)] = doc_id

print("Documents prepared:", len(documents))

index = SearchIndex.build(documents, error_rate=0.01)
index.save(OUTPUT_FILE)

with open(MAPPING_FILE, "w", encoding="utf-8") as f:
    json.dump(mapping, f, indent=2)

print("Index saved to:", OUTPUT_FILE)
print("Mapping saved to:", MAPPING_FILE)

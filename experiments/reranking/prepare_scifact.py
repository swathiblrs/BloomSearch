
import json
import os
from beir import util
from beir.datasets.data_loader import GenericDataLoader

print("Downloading SciFact")
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
data_path = util.download_and_unzip(url, "datasets")

print("Loading data")
corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

print(f"Documents: {len(corpus)}")
print(f"Queries: {len(queries)}")
print(f"Queries with relevance labels: {len(qrels)}")

documents = []
for doc_id, doc in corpus.items():
    documents.append({
        "id": doc_id,
        "title": doc["title"],
        "text": doc["text"]
    })

with open("scifact_docs.json", "w") as f:
    json.dump(documents, f, indent=2)

with open("scifact_queries.json", "w") as f:
    json.dump(queries, f, indent=2)

with open("scifact_qrels.json", "w") as f:
    json.dump(qrels, f, indent=2)

print("\nDONE")
print(f"Saved {len(documents)} documents to scifact_docs.json")
print(f"Saved {len(queries)} queries to scifact_queries.json")
print(f"Saved qrels for {len(qrels)} queries to scifact_qrels.json")

sample_id = list(corpus.keys())[0]
print(f"\nSample document:")
print(f"  ID: {sample_id}")
print(f"  Title: {corpus[sample_id]['title'][:80]}")
print(f"  Text: {corpus[sample_id]['text'][:100]}...")
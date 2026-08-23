# 🔎 BloomSearch — Bloom Filter Accelerated AI Search Engine

An API-first intelligent search engine that uses Bloom filters to reduce unnecessary index work, BM25 for keyword retrieval, NLP tokenization, and remotely hosted LoRA/QLoRA-capable language models for query rewriting, semantic reranking, and result explanations.

## 🌟 Why This Project Exists

Search engines spend time revisiting duplicate content, scanning index segments that cannot contain the requested terms, and ranking documents that are only weakly related to the user's intent.

BloomSearch demonstrates how a memory-efficient probabilistic data structure can improve that pipeline without replacing the search index or trusting probabilistic matches as final answers.

The project can:

- Detect exact duplicate documents before indexing
- Divide documents into independently searchable index segments
- Use a Bloom filter to skip segments that definitely lack query terms
- Rank keyword matches with BM25
- Rewrite unclear queries through a hosted language-model API
- Rerank candidate documents through the same hosted API
- Generate a short relevance explanation for every returned result
- Connect to a provider-hosted LoRA or QLoRA fine-tuned model
- Persist indexes as portable JSON files
- Report Bloom-filter capacity, fill ratio, and estimated false-positive rate

## 🌸 What Is a Bloom Filter?

A Bloom filter is a compact data structure used to answer a membership question:

> **Have I probably seen this item before?**

Instead of storing every item, it stores a bit array containing only `0` and `1`. When an item is added, several hash functions convert it into positions in that array, and the bits at those positions are changed to `1`.

For example, start with a 10-bit array:

```text
Index: 0 1 2 3 4 5 6 7 8 9
Bits:  0 0 0 0 0 0 0 0 0 0
```

Suppose the hash positions for `apple` are `1`, `4`, and `7`. Adding it sets those bits:

```text
Index: 0 1 2 3 4 5 6 7 8 9
Bits:  0 1 0 0 1 0 0 1 0 0
```

To check another value, the same hash positions are calculated:

- If **any required bit is `0`**, the value is **definitely not present**.
- If **all required bits are `1`**, the value is **probably present**.

The answer is "probably present" because different items can share the same bit positions. An item that was never added may therefore appear present. This is called a **false positive**.

| Bloom-filter result | Meaning |
|---|---|
| Definitely not present | The item was not added to the filter |
| Probably present | The item may have been added and should be verified |
| False positive | The filter says "probably present," but the item is absent |
| False negative | Not expected in a standard Bloom filter when it is used correctly |

Bloom filters are useful when memory is limited and a fast negative answer can avoid expensive work. They are commonly applied to web crawling, databases, caches, storage systems, distributed systems, and search indexes.

### Why Use It in a Search Engine?

Imagine a search index split into 1,000 segments. Searching every segment for every query wastes time. BloomSearch gives each segment a small Bloom filter containing the terms found in that segment.

For a query such as `python bloom filter`, the engine checks the segment filters first:

- If none of those terms can exist in a segment, BloomSearch skips it.
- If one or more terms may exist, BloomSearch searches that segment with BM25.
- A false positive may cause an extra segment search, but it does not remove a correct result.

This makes the Bloom filter a safe optimization layer: it can eliminate definitely irrelevant work, while BM25 and the hosted model remain responsible for ranking actual documents.

## 🏗️ Architecture Overview

High-level search workflow:

```text
User query
    -> Hosted model API rewrites and expands the query
    -> Segment Bloom filters skip definitely irrelevant segments
    -> BM25 retrieves keyword-matched candidate documents
    -> Hosted model API reranks candidates by semantic relevance
    -> BloomSearch returns ranked results with explanations
```

Indexing workflow:

```text
JSON documents
    -> Normalize and fingerprint content
    -> Bloom-filter duplicate precheck
    -> Exact verification of probable duplicates
    -> NLP tokenization
    -> Build segmented inverted index and term Bloom filters
    -> Save portable search index
```

Core stack:

- Python 3.10+
- Custom Bloom filter backed by a compact `bytearray`
- BLAKE2b double hashing for deterministic Bloom-filter positions
- SHA-256 content fingerprints for document deduplication
- Regular-expression NLP tokenization
- Segmented inverted index
- BM25 keyword ranking
- OpenAI-compatible `/chat/completions` API protocol
- User-selected hosted model or hosted LoRA/QLoRA adapter
- FastAPI browser interface
- Docker and user-owned Google Cloud Run deployment
- Standard-library HTTP and JSON support
- `unittest` test suite with a fake hosted-model client

## 🤖 Key Features

### 🌸 Bloom-Filter Search Optimization

- Automatically calculates the bit-array size and hash count from capacity and target error rate
- Stores filter state compactly as bits rather than Python objects
- Assigns an independent term Bloom filter to every index segment
- Skips a segment when all rewritten query terms are definitely absent
- Uses Bloom filters only as a precheck because probable matches may be false positives
- Reports bit usage, fill ratio, hash count, and estimated current false-positive probability

### 🔤 NLP and BM25 Retrieval

- Normalizes text to lowercase searchable tokens
- Indexes document titles and content
- Records per-document term frequencies and document lengths
- Calculates inverse document frequency across all segments
- Applies BM25 term-frequency and length normalization
- Returns the strongest candidates before an API reranking call

### 🧠 API-Hosted LLM Reranking

- Sends no model weights or inference work to the local machine
- Rewrites informal queries while preserving user intent and named entities
- Scores BM25 candidates from `0` to `1` by semantic relevance
- Produces short result explanations
- Rejects invented document identifiers from model responses
- Supports JSON returned directly or inside a Markdown code fence
- Keeps the model provider configurable through environment variables

### 🧩 LoRA and QLoRA Support

BloomSearch does not train or load adapters locally. Users can fine-tune and deploy an adapter with their preferred cloud platform, then configure the deployed model ID.

Recommended fine-tuning tasks:

- **Query rewriting:** informal request → focused search query
- **Search reranking:** query and document → relevance score and explanation
- **Domain adaptation:** general search query → terminology appropriate for a selected document collection

At runtime, BloomSearch treats a hosted base model, LoRA adapter, and QLoRA-trained model in the same way: each is a remote model identifier behind an API.

## 💡 Example Use Cases

Example searches:

- "How can a crawler avoid visiting the same URL twice?"
- "What data structure can check membership without storing every object?"
- "How does BM25 rank keyword matches?"
- "Can a hosted LoRA model improve search-result ranking?"
- "Why can a Bloom filter say an absent term might exist?"

Useful project domains include:

| Domain | Search Collection | BloomSearch Contribution |
|---|---|---|
| Technical documentation | Manuals and API guides | Skips irrelevant index segments and reranks matching pages |
| Academic search | Paper titles and abstracts | Removes exact duplicates and improves query wording |
| News archives | Articles and summaries | Filters repeated content and ranks topical matches |
| Product catalogs | Product descriptions | Rewrites natural-language shopping queries |
| Educational search | Notes and course material | Explains why each lesson matches the question |

## 📂 Project Structure

```text
bloom_filter/
 ├── core.py              # Bloom filter implementation and persistence
 ├── cli.py               # Standalone Bloom-filter commands
 └── __main__.py          # python -m bloom_filter entry point

bloom_search/
 ├── api.py               # Provider-neutral hosted-model API client
 ├── cli.py               # Index build, statistics, and search commands
 ├── engine.py            # Query rewriting and candidate reranking workflow
 ├── index.py             # Segmented index, deduplication, tokenization, and BM25
 ├── web.py               # FastAPI routes and per-request user API configuration
 └── __main__.py          # python -m bloom_search entry point

examples/
 └── documents.json       # Small sample search collection

web/
 └── index.html           # Browser search interface

deploy/
 └── google-cloud-run.md  # User-owned Cloud Run deployment guide

tests/
 ├── test_bloom_filter.py # Data-structure and persistence tests
 └── test_bloom_search.py # Index, BM25, Bloom skipping, and API workflow tests
```

## 🚀 Getting Started

### ✅ Prerequisites

- Python 3.10 or newer
- An account with a compatible hosted model provider
- An API key
- A hosted model or adapter that can follow JSON output instructions
- An OpenAI-compatible `/chat/completions` endpoint

No local LLM server, GPU, PyTorch installation, tokenizer download, or model download is required.

### 🔧 Local Setup

Clone the repository:

```bash
git clone https://github.com/swathiblrs/Bloom-Filter.git
cd Bloom-Filter
```

Optionally install the package in editable mode:

```bash
python -m pip install -e .
```

Configure the hosted model API:

```bash
export BLOOMSEARCH_API_BASE="https://your-provider.example/v1"
export BLOOMSEARCH_API_KEY="your-api-key"
export BLOOMSEARCH_MODEL="your-hosted-model-or-adapter-id"
```

Do not commit API keys. The `.env` filename is ignored, but BloomSearch reads variables from the process environment rather than loading `.env` automatically.

The provider is intentionally not hard-coded. Any service may be used if it accepts OpenAI-style chat messages at `/chat/completions` and returns the standard `choices[0].message.content` response shape.

## 🔍 Run BloomSearch

Build an index from the included sample documents:

```bash
python -m bloom_search build \
  examples/documents.json \
  demo.bloom-index.json \
  --segment-size 2
```

Inspect the index and its segment Bloom filters:

```bash
python -m bloom_search stats demo.bloom-index.json
```

Search through the configured hosted model API:

```bash
python -m bloom_search search \
  demo.bloom-index.json \
  "how can crawlers avoid visiting the same URL"
```

Search is intentionally unavailable without API configuration. Index building and statistics do not require an API call.

## 🌐 Browser Interface

Install the web dependencies and start the application:

```bash
python -m pip install -e ".[web]"
uvicorn bloom_search.web:app --reload
```

Open `http://localhost:8000`. Each user enters their own compatible API base URL, API key, and hosted model ID. The credential is used only for that search request and is not persisted by BloomSearch.

## ☁️ User-Owned Google Cloud Deployment

This repository does not operate a shared paid cloud service. Anyone who wants a public deployment creates it inside their own Google Cloud project and pays their own infrastructure and hosted-model charges.

The included [Google Cloud Run deployment guide](deploy/google-cloud-run.md) uses Docker with:

- Minimum instances set to `0`
- Maximum instances capped at `1`
- No repository-owner API key
- No Secrets Manager dependency
- API credentials supplied by the user per search request
- Public HTTPS provider URLs only; private-network targets are rejected

These limits reduce accidental usage but do not guarantee that Google Cloud will charge the deploying user zero dollars. Nothing is deployed automatically by this repository.

## 📄 Input Document Format

Collections are JSON arrays. Every document needs an `id`, `title`, and `text`; `url` is optional.

```json
[
  {
    "id": "crawler",
    "title": "Avoiding Duplicate URLs in a Web Crawler",
    "text": "A crawler can use a Bloom filter to precheck visited URLs.",
    "url": "https://example.test/web-crawler"
  }
]
```

## 🔌 Hosted Model Contract

The query-rewrite call asks for:

```json
{
  "rewritten_query": "How do Bloom filters prevent duplicate crawler URLs?"
}
```

The reranking call asks for:

```json
{
  "results": [
    {
      "id": "crawler",
      "relevance_score": 0.98,
      "explanation": "This document directly describes visited-URL filtering."
    }
  ]
}
```

BloomSearch clamps relevance scores to the range `0`–`1`, ignores unknown IDs, and removes duplicate model results.

## 🧪 Testing

Run the complete automated test suite:

```bash
python -m unittest discover -s tests -v
```

The test suite validates:

- Bloom-filter sizing formulas
- No false negatives for inserted values
- String and byte membership behavior
- Filter serialization and restoration
- Parameter validation
- BM25 result ordering
- Bloom-filter index-segment skipping
- Exact duplicate-document removal
- Search-index persistence
- API-based query rewriting and reranking
- Hosted-model interactions through a fake client

Tests do not contact an external provider and do not run a local language model.

## 🧮 Standalone Bloom-Filter CLI

Run the original presentation example:

```bash
python -m bloom_filter demo
```

Create and query a persistent filter:

```bash
python -m bloom_filter create usernames.bloom.json --capacity 10000 --error-rate 0.01
python -m bloom_filter add usernames.bloom.json swathi disha
python -m bloom_filter check usernames.bloom.json swathi unknown-user
python -m bloom_filter stats usernames.bloom.json
```

## ⚡ Performance and Reliability

- Bloom filters can skip index segments without loading or scanning their document terms
- BM25 limits the number of candidates sent to the hosted model
- API requests have a configurable client timeout
- Invalid HTTP, network, and JSON responses produce explicit errors
- Probable Bloom-filter matches are never treated as authoritative duplicate decisions
- No API credentials are stored in the search index

## 🔮 Future Improvements

- Add a permitted web crawler with a visited-URL Bloom filter
- Create larger benchmark collections and relevance judgments
- Measure Precision@K, Recall@K, MRR, and NDCG
- Compare latency with and without segment Bloom filters
- Add configurable stop words, stemming, and language-specific tokenizers
- Add incremental index segments without rebuilding existing segments
- Add API retries with bounded exponential backoff
- Add support for provider-specific authentication headers through configuration
- Export query-rewrite and reranking datasets for hosted LoRA/QLoRA training

## 🙌 Acknowledgements

Built as an educational search-engine project combining Bloom filters, information retrieval, NLP, BM25 ranking, and API-hosted language-model intelligence. The design keeps probabilistic filtering deterministic and local while requiring all LLM inference and optional LoRA/QLoRA adaptation to remain with the user's selected provider.

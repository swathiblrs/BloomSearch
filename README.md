# BloomSearch

BloomSearch is an API-first intelligent search engine combining Bloom filters, BM25 retrieval, NLP tokenization, and remotely hosted LoRA/QLoRA-capable language models.

**No LLM is downloaded or run locally.** Every query rewrite and neural reranking request is sent to an API selected by the user.

## Architecture

```text
Query
  -> hosted LLM API: query rewriting
  -> per-segment Bloom filters: skip definitely irrelevant index segments
  -> BM25: retrieve candidates
  -> hosted LLM API: relevance reranking and explanations
  -> ranked results
```

Bloom filters are also used while indexing to precheck exact duplicate documents. A normal set verifies probable matches because Bloom filters can return false positives.

## Features

- Compact Bloom filter implementation with configurable false-positive rate.
- Segmented inverted index with a term Bloom filter per segment.
- BM25 keyword retrieval.
- Exact-content deduplication during indexing.
- Hosted-model query rewriting, reranking, and explanations.
- Provider-neutral OpenAI-compatible HTTP client using only the Python standard library.
- No local model, tokenizer, PyTorch, GPU, or model download.
- Persistent JSON index and command-line interface.

## Requirements

- Python 3.10+
- An API provider offering an OpenAI-compatible `/chat/completions` endpoint
- A hosted instruction model, including a provider-hosted LoRA/QLoRA fine-tuned model if desired

## Configure the model API

Set these variables in the terminal. Do not commit API keys.

```bash
export BLOOMSEARCH_API_BASE="https://your-provider.example/v1"
export BLOOMSEARCH_API_KEY="your-api-key"
export BLOOMSEARCH_MODEL="your-hosted-model-or-adapter-id"
```

The project deliberately does not prescribe a provider. Change these variables to use any compatible service. The endpoint must accept OpenAI-style chat messages and the hosted model must follow the JSON output instructions. Native provider JSON-mode support is not required.

## Run BloomSearch

Build the included example index. Index construction is deterministic and does not call an LLM:

```bash
python -m bloom_search build examples/documents.json demo.bloom-index.json --segment-size 2
python -m bloom_search stats demo.bloom-index.json
```

Search requires the configured remote API:

```bash
python -m bloom_search search demo.bloom-index.json "how can crawlers avoid visiting the same URL"
```

Run the tests. Tests use a fake API client and make no network requests:

```bash
python -m unittest discover -s tests -v
```

## LoRA and QLoRA

BloomSearch does not train adapters locally. Fine-tune and host the adapter using a cloud platform or model provider, then place its deployed model ID in `BLOOMSEARCH_MODEL`.

Recommended training tasks:

1. **Query rewriting:** train records mapping informal queries to concise search queries.
2. **Relevance reranking:** train records containing a query, candidate document, relevance score, and short explanation.

Example query-rewrite record:

```json
{
  "instruction": "Rewrite this request as an effective search query.",
  "input": "how bloom thing avoid same websites",
  "output": {"rewritten_query": "How do Bloom filters prevent duplicate URLs in web crawlers?"}
}
```

Provider-hosted LoRA is appropriate for a smaller reranker. Provider-hosted QLoRA is useful for fine-tuning a larger instruction model at lower training-memory cost. At runtime BloomSearch treats both simply as hosted model IDs.

## Original Bloom-filter CLI

The standalone data structure remains available:

```bash
python -m bloom_filter demo
python -m bloom_filter create usernames.bloom.json --capacity 10000 --error-rate 0.01
python -m bloom_filter add usernames.bloom.json swathi disha
python -m bloom_filter check usernames.bloom.json swathi unknown-user
```

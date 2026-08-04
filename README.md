# Bloom Filter in Python

A compact, dependency-free implementation of the probabilistic data structure described in `BLOOM FILTER.pptx`.

A Bloom filter answers one question very quickly:

- **Definitely not present** — at least one required bit is zero.
- **Probably present** — all required bits are one, although this can be a false positive.

Inserted values do not produce false negatives as long as the filter is not modified outside this library. Standard Bloom filters do not support deletion because clearing a shared bit can make another inserted value disappear.

## Features

- Automatically calculates the optimal bit-array size and number of hashes.
- Stores bits compactly in a `bytearray`.
- Uses deterministic BLAKE2b double hashing.
- Reports fill ratio and estimated false-positive probability.
- Saves and reloads filters as versioned JSON files.
- Includes a CLI, slide-inspired demo, and unit tests.

## Quick start

Python 3.10 or newer is required. No third-party runtime packages are needed.

```bash
python -m bloom_filter demo
python -m unittest discover -s tests -v
```

Create a persistent filter for 10,000 expected values with a 1% target false-positive rate:

```bash
python -m bloom_filter create usernames.bloom.json --capacity 10000 --error-rate 0.01
python -m bloom_filter add usernames.bloom.json swathi disha divyashree deeksha
python -m bloom_filter check usernames.bloom.json swathi unknown-user
python -m bloom_filter stats usernames.bloom.json
```

## Python API

```python
from bloom_filter import BloomFilter

bloom = BloomFilter(capacity=1_000, error_rate=0.01)
bloom.add("geeks")

if "geeks" in bloom:
    print("probably present")
else:
    print("definitely not present")
```

## Mathematics

For expected insertions `n` and desired false-positive probability `p`, the implementation uses:

```text
m = ceil(-(n * ln(p)) / (ln(2)^2))
k = round((m / n) * ln(2))
```

where `m` is the number of bits and `k` is the number of hash positions. The estimated current false-positive rate after `n` add operations is:

```text
(1 - exp(-k * n / m))^k
```

Capacity is a design target, not a hard limit. Adding substantially more values increases false positives.

## Good project extensions

- A counting Bloom filter for safe deletion.
- A scalable Bloom filter that adds layers as capacity grows.
- FastAPI endpoints for username or malicious-URL prechecks.
- Benchmarks against a Python `set` for memory and lookup speed.

## LLM integration ideas

The Bloom filter should remain the deterministic, fast gate. An LLM is useful after that gate:

1. **RAG duplicate-document guard:** normalize a document, query the filter by content fingerprint, and skip embedding likely duplicates. Ask an LLM to explain duplicate clusters or propose canonical titles.
2. **Prompt/cache routing:** use a filter to check whether a normalized request may already be cached. On a probable match, verify against the real cache before returning anything.
3. **Security triage:** use a filter as a local precheck for known risky URLs, package names, or leaked-password fingerprints. Send only probable matches to a classifier or analyst-facing LLM explanation; never treat the Bloom result alone as proof.
4. **Natural-language observability:** give an LLM the output of `stats()` and recent insertion counts so it can explain saturation and recommend a larger capacity or lower target error rate.

Because Bloom filters can produce false positives, every consequential LLM workflow must verify probable matches against an authoritative store.


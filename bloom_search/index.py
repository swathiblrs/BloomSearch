"""Segmented inverted index with Bloom-filter term skipping and BM25."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Iterable

from bloom_filter import BloomFilter


TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "can", "do", "does", "for", "from",
    "how", "i", "in", "is", "it", "me", "my", "of", "on", "or", "the", "this", "to",
    "was", "what", "when", "where", "which", "who", "why", "will", "with", "you", "your",
}


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def has_meaningful_match(query: str, document: "Document") -> bool:
    """Reject results supported only by generic stop words."""
    query_terms = {term for term in tokenize(query) if term not in STOP_WORDS}
    if not query_terms:
        return False
    document_terms = set(tokenize(f"{document.title} {document.text}"))
    return bool(query_terms & document_terms)


@dataclass(frozen=True, slots=True)
class Document:
    id: str
    title: str
    text: str
    url: str = ""


@dataclass(slots=True)
class IndexSegment:
    documents: list[Document]
    term_frequencies: dict[str, dict[str, int]]
    document_lengths: dict[str, int]
    term_filter: BloomFilter

    @classmethod
    def build(cls, documents: list[Document], error_rate: float) -> "IndexSegment":
        frequencies: dict[str, dict[str, int]] = {}
        lengths: dict[str, int] = {}
        vocabulary: set[str] = set()
        for document in documents:
            counts = Counter(tokenize(f"{document.title} {document.text}"))
            frequencies[document.id] = dict(counts)
            lengths[document.id] = sum(counts.values())
            vocabulary.update(counts)
        term_filter = BloomFilter(max(1, len(vocabulary)), error_rate)
        term_filter.add_many(vocabulary)
        return cls(documents, frequencies, lengths, term_filter)


class SearchIndex:
    FORMAT_VERSION = 1

    def __init__(self, segments: list[IndexSegment], error_rate: float = 0.01) -> None:
        self.segments = segments
        self.error_rate = error_rate

    @classmethod
    def build(
        cls,
        documents: Iterable[Document],
        *,
        segment_size: int = 100,
        error_rate: float = 0.01,
    ) -> "SearchIndex":
        if segment_size <= 0:
            raise ValueError("segment_size must be greater than zero")
        unique_documents: list[Document] = []
        fingerprints = BloomFilter(capacity=100_000, error_rate=error_rate)
        confirmed: set[str] = set()
        for document in documents:
            normalized = f"{document.title.strip().lower()}\n{document.text.strip().lower()}"
            fingerprint = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            if fingerprint in fingerprints and fingerprint in confirmed:
                continue
            fingerprints.add(fingerprint)
            confirmed.add(fingerprint)
            unique_documents.append(document)
        segments = [
            IndexSegment.build(unique_documents[start : start + segment_size], error_rate)
            for start in range(0, len(unique_documents), segment_size)
        ]
        return cls(segments, error_rate)

    @property
    def document_count(self) -> int:
        return sum(len(segment.documents) for segment in self.segments)

    def search_bm25(self, query: str, limit: int = 20) -> tuple[list[tuple[Document, float]], int]:
        terms = tokenize(query)
        if not terms:
            return [], len(self.segments)
        selected_indices: list[int] = []
        skipped = 0
        for index, segment in enumerate(self.segments):
            if not any(term in segment.term_filter for term in terms):
                skipped += 1
                continue
            selected_indices.append(index)
        return self.search_bm25_selected(query, selected_indices, limit), skipped

    @staticmethod
    def snippet(document: Document, query: str, *, length: int = 240) -> str:
        """Return a compact result snippet centered on the first matching term."""
        text = " ".join(document.text.split())
        if len(text) <= length:
            return text
        lowered = text.lower()
        positions = [lowered.find(term) for term in tokenize(query)]
        positions = [position for position in positions if position >= 0]
        center = min(positions, default=0)
        start = max(0, center - length // 3)
        end = min(len(text), start + length)
        prefix = "..." if start else ""
        suffix = "..." if end < len(text) else ""
        return prefix + text[start:end].strip() + suffix

    def search_bm25_selected(
        self, query: str, segment_indices: Iterable[int], limit: int = 20
    ) -> list[tuple[Document, float]]:
        """Score documents from explicitly selected segments using global IDF."""
        terms = tokenize(query)
        if not terms:
            return []
        selected = [self.segments[index] for index in segment_indices]
        candidates = [
            (segment, document) for segment in selected for document in segment.documents
        ]

        total_documents = max(1, self.document_count)
        document_frequency: dict[str, int] = defaultdict(int)
        for segment in self.segments:
            for term in terms:
                document_frequency[term] += sum(
                    term in segment.term_frequencies[document.id] for document in segment.documents
                )
        all_lengths = [length for segment in self.segments for length in segment.document_lengths.values()]
        average_length = sum(all_lengths) / max(1, len(all_lengths))
        scored: list[tuple[Document, float]] = []
        k1, b = 1.5, 0.75
        for segment, document in candidates:
            frequencies = segment.term_frequencies[document.id]
            length = segment.document_lengths[document.id]
            score = 0.0
            for term in terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                df = document_frequency[term]
                inverse_frequency = math.log(1 + (total_documents - df + 0.5) / (df + 0.5))
                denominator = frequency + k1 * (1 - b + b * length / max(1, average_length))
                score += inverse_frequency * frequency * (k1 + 1) / denominator
            if score:
                scored.append((document, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:limit]

    def save(self, path: str | Path) -> None:
        data = {
            "format_version": self.FORMAT_VERSION,
            "error_rate": self.error_rate,
            "segments": [
                {
                    "documents": [asdict(document) for document in segment.documents],
                    "term_frequencies": segment.term_frequencies,
                    "document_lengths": segment.document_lengths,
                    "term_filter": segment.term_filter.to_dict(),
                }
                for segment in self.segments
            ],
        }
        Path(path).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SearchIndex":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("format_version") != cls.FORMAT_VERSION:
            raise ValueError("unsupported search index version")
        segments: list[IndexSegment] = []
        for stored in data["segments"]:
            bloom_data = stored["term_filter"]
            config = bloom_data["config"]
            import base64

            term_filter = BloomFilter(
                int(config["capacity"]),
                float(config["error_rate"]),
                _bits=base64.b64decode(bloom_data["bits"], validate=True),
                _items_added=int(bloom_data["items_added"]),
            )
            segments.append(
                IndexSegment(
                    documents=[Document(**document) for document in stored["documents"]],
                    term_frequencies=stored["term_frequencies"],
                    document_lengths={key: int(value) for key, value in stored["document_lengths"].items()},
                    term_filter=term_filter,
                )
            )
        return cls(segments, float(data["error_rate"]))

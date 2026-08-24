"""Certainty-aware learned segment filtering and reproducible evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Iterable, Sequence

from bloom_filter import CountingBloomFilter

from .index import Document, SearchIndex, tokenize


@dataclass(frozen=True, slots=True)
class QueryCase:
    query: str
    relevant_ids: frozenset[str]


@dataclass(frozen=True, slots=True)
class SegmentDecision:
    segment_index: int
    bloom_positive: bool
    certainty: float
    relevance_probability: float


class LogisticSegmentClassifier:
    """Small dependency-free logistic regression for segment relevance."""

    def __init__(self, feature_count: int) -> None:
        if feature_count <= 0:
            raise ValueError("feature_count must be positive")
        self.weights = [0.0] * (feature_count + 1)

    @staticmethod
    def _sigmoid(value: float) -> float:
        if value >= 0:
            decay = math.exp(-value)
            return 1.0 / (1.0 + decay)
        growth = math.exp(value)
        return growth / (1.0 + growth)

    def predict_probability(self, features: Sequence[float]) -> float:
        if len(features) + 1 != len(self.weights):
            raise ValueError("feature length does not match classifier")
        value = self.weights[0] + sum(
            weight * feature for weight, feature in zip(self.weights[1:], features)
        )
        return self._sigmoid(value)

    def fit(
        self,
        rows: Sequence[tuple[Sequence[float], int]],
        *,
        learning_rate: float = 0.15,
        epochs: int = 500,
        l2: float = 0.001,
        positive_weight: float = 1.0,
    ) -> None:
        if not rows:
            raise ValueError("training rows cannot be empty")
        for _ in range(epochs):
            gradient = [0.0] * len(self.weights)
            total_weight = 0.0
            for features, label in rows:
                sample_weight = positive_weight if label else 1.0
                error = (self.predict_probability(features) - label) * sample_weight
                gradient[0] += error
                for index, feature in enumerate(features, start=1):
                    gradient[index] += error * feature
                total_weight += sample_weight
            scale = learning_rate / max(1.0, total_weight)
            self.weights[0] -= scale * gradient[0]
            for index in range(1, len(self.weights)):
                self.weights[index] -= scale * (gradient[index] + l2 * self.weights[index])

    @property
    def logical_byte_count(self) -> int:
        return len(self.weights) * 8


class CertaintyAwareCascade:
    """Counting filters followed by a learned segment-relevance stage."""

    def __init__(self, index: SearchIndex, error_rate: float | None = None) -> None:
        self.index = index
        self.error_rate = error_rate if error_rate is not None else index.error_rate
        self.filters: list[CountingBloomFilter] = []
        for segment in index.segments:
            vocabulary = {
                term for frequencies in segment.term_frequencies.values() for term in frequencies
            }
            counting = CountingBloomFilter(max(1, len(vocabulary)), self.error_rate)
            counting.add_many(vocabulary)
            self.filters.append(counting)
        self.classifier = LogisticSegmentClassifier(feature_count=6)
        self.threshold = 0.5

    def _features(self, query: str, segment_index: int) -> tuple[float, ...]:
        terms = tuple(dict.fromkeys(tokenize(query)))
        counting = self.filters[segment_index]
        certainties = [counting.certainty(term) for term in terms]
        positives = [value for value in certainties if value > 0]
        positive_ratio = len(positives) / max(1, len(terms))
        mean_certainty = sum(positives) / max(1, len(positives))
        max_certainty = max(positives, default=0.0)
        min_certainty = min(positives, default=0.0)
        return (
            positive_ratio,
            mean_certainty,
            max_certainty,
            min_certainty,
            counting.fill_ratio,
            min(1.0, len(terms) / 10.0),
        )

    def _relevant_segments(self, case: QueryCase) -> set[int]:
        return {
            index
            for index, segment in enumerate(self.index.segments)
            if any(document.id in case.relevant_ids for document in segment.documents)
        }

    def fit(self, cases: Sequence[QueryCase], target_segment_recall: float = 0.99) -> None:
        if not 0 < target_segment_recall <= 1:
            raise ValueError("target_segment_recall must be in (0, 1]")
        rows: list[tuple[Sequence[float], int]] = []
        positive_count = 0
        negative_count = 0
        for case in cases:
            relevant = self._relevant_segments(case)
            for segment_index in range(len(self.index.segments)):
                label = int(segment_index in relevant)
                rows.append((self._features(case.query, segment_index), label))
                positive_count += label
                negative_count += 1 - label
        self.classifier.fit(
            rows,
            positive_weight=max(1.0, negative_count / max(1, positive_count)),
        )

        probabilities = sorted(
            {
                round(self.classifier.predict_probability(features), 6)
                for features, _ in rows
            },
            reverse=True,
        )
        chosen = 0.0
        for threshold in probabilities:
            relevant_total = 0
            relevant_selected = 0
            for case in cases:
                relevant = self._relevant_segments(case)
                relevant_total += len(relevant)
                selected = set(self.select_segments(case.query, threshold=threshold))
                relevant_selected += len(relevant & selected)
            recall = relevant_selected / max(1, relevant_total)
            if recall >= target_segment_recall:
                chosen = threshold
                break
        self.threshold = chosen

    def decisions(self, query: str) -> list[SegmentDecision]:
        decisions: list[SegmentDecision] = []
        terms = tuple(dict.fromkeys(tokenize(query)))
        for index, counting in enumerate(self.filters):
            certainties = [counting.certainty(term) for term in terms]
            positive = any(value > 0 for value in certainties)
            probability = self.classifier.predict_probability(self._features(query, index))
            decisions.append(
                SegmentDecision(
                    segment_index=index,
                    bloom_positive=positive,
                    certainty=max(certainties, default=0.0),
                    relevance_probability=probability,
                )
            )
        return decisions

    def select_segments(self, query: str, threshold: float | None = None) -> list[int]:
        cutoff = self.threshold if threshold is None else threshold
        positive = [decision for decision in self.decisions(query) if decision.bloom_positive]
        positive.sort(
            key=lambda decision: (decision.relevance_probability, decision.certainty), reverse=True
        )
        selected = [
            decision.segment_index
            for decision in positive
            if decision.relevance_probability >= cutoff
        ]
        if not selected and positive:
            selected.append(positive[0].segment_index)
        return selected

    @property
    def logical_byte_count(self) -> int:
        return sum(item.logical_byte_count for item in self.filters) + self.classifier.logical_byte_count


TOPICS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("bloom", ("filter", "membership", "hash", "probabilistic")),
    ("crawler", ("url", "visited", "duplicate", "spider")),
    ("ranking", ("bm25", "relevance", "score", "retrieval")),
    ("database", ("sstable", "lookup", "storage", "index")),
    ("cache", ("latency", "hit", "miss", "memory")),
    ("python", ("package", "function", "testing", "code")),
    ("cloud", ("container", "serverless", "deployment", "runtime")),
    ("language", ("token", "query", "semantic", "rewrite")),
    ("neural", ("model", "training", "classifier", "prediction")),
    ("privacy", ("data", "encoding", "anonymous", "protection")),
    ("vector", ("embedding", "similarity", "nearest", "search")),
    ("monitoring", ("metric", "trace", "log", "alert")),
)


def build_synthetic_benchmark(
    *, documents_per_segment: int = 50, error_rate: float = 0.15
) -> tuple[SearchIndex, list[QueryCase], list[QueryCase]]:
    """Create deterministic topical documents and labeled train/test queries."""
    documents: list[Document] = []
    train: list[QueryCase] = []
    test: list[QueryCase] = []
    for topic, keywords in TOPICS:
        for number in range(documents_per_segment):
            concept = f"concept{number}"
            identifier = f"{topic}-{number}"
            text = (
                f"{topic} {' '.join(keywords)} {concept} practical guide example architecture "
                f"performance evaluation for {topic} systems"
            )
            documents.append(Document(identifier, f"{topic.title()} {concept} guide", text))
            case = QueryCase(
                query=f"{topic} {keywords[number % len(keywords)]} {concept} guide",
                relevant_ids=frozenset({identifier}),
            )
            (train if number % 2 == 0 else test).append(case)
    index = SearchIndex.build(
        documents, segment_size=documents_per_segment, error_rate=error_rate
    )
    return index, train, test


def _ranking_metrics(results: Sequence[tuple[Document, float]], relevant: frozenset[str]) -> tuple[float, float, float]:
    ids = [document.id for document, _ in results[:10]]
    hits = [index for index, identifier in enumerate(ids, start=1) if identifier in relevant]
    recall = len(hits) / max(1, len(relevant))
    reciprocal_rank = 1.0 / hits[0] if hits else 0.0
    dcg = sum(1.0 / math.log2(rank + 1) for rank in hits)
    ideal_hits = min(len(relevant), 10)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    ndcg = dcg / max(1.0, ideal)
    return recall, reciprocal_rank, ndcg


def _estimated_tokens(results: Sequence[tuple[Document, float]]) -> int:
    characters = sum(len(document.title) + len(document.text) + 32 for document, _ in results)
    return math.ceil(characters / 4)


def evaluate_benchmark() -> dict[str, object]:
    index, train, test = build_synthetic_benchmark()
    cascade = CertaintyAwareCascade(index)
    cascade.fit(train, target_segment_recall=0.99)

    systems = {
        "bm25_all_segments": {"segment_selector": "all", "candidate_limit": 10},
        "standard_bloom_bm25": {"segment_selector": "standard", "candidate_limit": 10},
        "counting_bloom_bm25": {"segment_selector": "counting", "candidate_limit": 10},
        "certainty_cascade_bm25": {"segment_selector": "cascade", "candidate_limit": 5},
    }
    aggregates: dict[str, dict[str, float]] = {
        name: {
            "queries": 0,
            "segments_searched": 0,
            "documents_scored": 0,
            "recall_at_10": 0,
            "mrr": 0,
            "ndcg_at_10": 0,
            "estimated_reranker_tokens": 0,
            "latency_ms": 0,
        }
        for name in systems
    }

    for case in test:
        terms = tuple(dict.fromkeys(tokenize(case.query)))
        for name, config in systems.items():
            started = perf_counter()
            selector = config["segment_selector"]
            if selector == "all":
                selected = list(range(len(index.segments)))
            elif selector == "standard":
                selected = [
                    segment_index
                    for segment_index, segment in enumerate(index.segments)
                    if any(term in segment.term_filter for term in terms)
                ]
            elif selector == "counting":
                selected = [
                    segment_index
                    for segment_index, counting in enumerate(cascade.filters)
                    if any(term in counting for term in terms)
                ]
            else:
                selected = cascade.select_segments(case.query)
            results = index.search_bm25_selected(
                case.query, selected, limit=int(config["candidate_limit"])
            )
            elapsed_ms = (perf_counter() - started) * 1_000
            recall, reciprocal_rank, ndcg = _ranking_metrics(results, case.relevant_ids)
            row = aggregates[name]
            row["queries"] += 1
            row["segments_searched"] += len(selected)
            row["documents_scored"] += sum(
                len(index.segments[segment_index].documents) for segment_index in selected
            )
            row["recall_at_10"] += recall
            row["mrr"] += reciprocal_rank
            row["ndcg_at_10"] += ndcg
            row["estimated_reranker_tokens"] += _estimated_tokens(results)
            row["latency_ms"] += elapsed_ms

    finalized: dict[str, dict[str, float | int]] = {}
    for name, values in aggregates.items():
        count = int(values["queries"])
        finalized[name] = {
            "queries": count,
            "avg_segments_searched": round(values["segments_searched"] / count, 4),
            "avg_documents_scored": round(values["documents_scored"] / count, 4),
            "recall_at_10": round(values["recall_at_10"] / count, 4),
            "mrr": round(values["mrr"] / count, 4),
            "ndcg_at_10": round(values["ndcg_at_10"] / count, 4),
            "avg_estimated_reranker_tokens": round(
                values["estimated_reranker_tokens"] / count, 2
            ),
            "avg_local_latency_ms": round(values["latency_ms"] / count, 4),
        }

    false_positive_checks = 0
    false_positives = 0
    for case in test:
        for term in set(tokenize(case.query)):
            for segment_index, segment in enumerate(index.segments):
                exact = any(term in frequencies for frequencies in segment.term_frequencies.values())
                if not exact:
                    false_positive_checks += 1
                    false_positives += term in segment.term_filter

    standard_logical_bytes = sum(
        (segment.term_filter.config.bit_count + 7) // 8 for segment in index.segments
    )
    return {
        "benchmark": {
            "type": "deterministic synthetic topical benchmark",
            "documents": index.document_count,
            "segments": len(index.segments),
            "training_queries": len(train),
            "test_queries": len(test),
            "configured_filter_error_rate": index.error_rate,
            "observed_standard_filter_false_positive_rate": round(
                false_positives / max(1, false_positive_checks), 4
            ),
        },
        "cascade": {
            "calibrated_probability_threshold": round(cascade.threshold, 6),
            "standard_bloom_logical_bytes": standard_logical_bytes,
            "counting_cascade_logical_bytes": cascade.logical_byte_count,
            "classifier_weights": [round(value, 6) for value in cascade.classifier.weights],
        },
        "systems": finalized,
    }


def write_results(output: str | Path) -> dict[str, object]:
    results = evaluate_benchmark()
    Path(output).write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results

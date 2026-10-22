"""Selective NLP action policy for BloomSearch.

This module deliberately contains no language model.  It extracts inexpensive
query and retrieval features, provides safe deterministic decisions for the
prototype, and includes a small trainable softmax classifier that can later be
fit on QReCC/CANARD/AmbigNQ-derived action labels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Sequence

from .index import SearchIndex, STOP_WORDS, tokenize


LAYER_ONE_ACTIONS = ("keep", "rewrite", "clarify")
CONTEXT_WORDS = {
    "he", "her", "hers", "him", "his", "it", "its", "she", "that", "their",
    "them", "these", "they", "this", "those", "there", "former", "latter",
}
AMBIGUITY_MARKERS = {"best", "cheap", "fast", "good", "near", "right", "today"}


@dataclass(frozen=True, slots=True)
class PolicyFeatures:
    """Normalized inputs shared by the layer-one and layer-two policies."""

    token_count: float
    meaningful_ratio: float
    context_dependence: float
    ambiguity: float
    bm25_confidence: float
    bm25_margin: float
    bloom_coverage: float

    def values(self) -> tuple[float, ...]:
        return tuple(asdict(self).values())


@dataclass(frozen=True, slots=True)
class ActionDecision:
    action: str
    confidence: float
    reason: str
    source: str
    features: PolicyFeatures

    def to_dict(self) -> dict[str, object]:
        return {
            "action": self.action,
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
            "source": self.source,
            "features": {key: round(value, 4) for key, value in asdict(self.features).items()},
        }


@dataclass(frozen=True, slots=True)
class RerankDecision:
    should_rerank: bool
    confidence: float
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "action": "rerank" if self.should_rerank else "return_bm25",
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
        }


class SoftmaxActionClassifier:
    """Dependency-free multiclass classifier for keep/rewrite/clarify labels."""

    FORMAT_VERSION = 1

    def __init__(self, feature_count: int = 7) -> None:
        if feature_count <= 0:
            raise ValueError("feature_count must be positive")
        self.feature_count = feature_count
        self.weights = [
            [0.0] * (feature_count + 1) for _ in LAYER_ONE_ACTIONS
        ]
        self.fitted = False

    @staticmethod
    def _probabilities(scores: Sequence[float]) -> list[float]:
        ceiling = max(scores)
        exponentials = [math.exp(score - ceiling) for score in scores]
        total = sum(exponentials)
        return [value / total for value in exponentials]

    def predict_proba(self, features: Sequence[float]) -> dict[str, float]:
        if len(features) != self.feature_count:
            raise ValueError("feature length does not match classifier")
        scores = [
            row[0] + sum(weight * value for weight, value in zip(row[1:], features))
            for row in self.weights
        ]
        return dict(zip(LAYER_ONE_ACTIONS, self._probabilities(scores)))

    def predict(self, features: Sequence[float]) -> tuple[str, float]:
        probabilities = self.predict_proba(features)
        action = max(probabilities, key=probabilities.get)  # type: ignore[arg-type]
        return action, probabilities[action]

    def fit(
        self,
        rows: Sequence[tuple[Sequence[float], str]],
        *,
        learning_rate: float = 0.2,
        epochs: int = 500,
        l2: float = 0.001,
    ) -> None:
        if not rows:
            raise ValueError("training rows cannot be empty")
        for features, label in rows:
            if len(features) != self.feature_count:
                raise ValueError("feature length does not match classifier")
            if label not in LAYER_ONE_ACTIONS:
                raise ValueError(f"unsupported action label: {label}")
        for _ in range(epochs):
            gradients = [[0.0] * (self.feature_count + 1) for _ in LAYER_ONE_ACTIONS]
            for features, label in rows:
                probabilities = self.predict_proba(features)
                for action_index, action in enumerate(LAYER_ONE_ACTIONS):
                    error = probabilities[action] - float(action == label)
                    gradients[action_index][0] += error
                    for feature_index, value in enumerate(features, start=1):
                        gradients[action_index][feature_index] += error * value
            scale = learning_rate / len(rows)
            for action_index, weight_row in enumerate(self.weights):
                weight_row[0] -= scale * gradients[action_index][0]
                for feature_index in range(1, len(weight_row)):
                    regularized = gradients[action_index][feature_index] + l2 * weight_row[feature_index]
                    weight_row[feature_index] -= scale * regularized
        self.fitted = True

    def save(self, path: str | Path) -> None:
        payload = {
            "format_version": self.FORMAT_VERSION,
            "actions": list(LAYER_ONE_ACTIONS),
            "feature_count": self.feature_count,
            "weights": self.weights,
            "fitted": self.fitted,
        }
        Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SoftmaxActionClassifier":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format_version") != cls.FORMAT_VERSION:
            raise ValueError("unsupported action-classifier format")
        if tuple(payload.get("actions", ())) != LAYER_ONE_ACTIONS:
            raise ValueError("action labels do not match this classifier")
        classifier = cls(int(payload["feature_count"]))
        weights = payload.get("weights")
        if not isinstance(weights, list) or len(weights) != len(LAYER_ONE_ACTIONS):
            raise ValueError("invalid action-classifier weights")
        classifier.weights = [[float(value) for value in row] for row in weights]
        if any(len(row) != classifier.feature_count + 1 for row in classifier.weights):
            raise ValueError("invalid action-classifier weight dimensions")
        classifier.fitted = bool(payload.get("fitted", True))
        return classifier


class JointActionPolicy:
    """Choose layer-one query handling and layer-two reranking actions."""

    def __init__(self, classifier: SoftmaxActionClassifier | None = None) -> None:
        self.classifier = classifier

    @staticmethod
    def extract_features(index: SearchIndex, query: str, context: str = "") -> PolicyFeatures:
        terms = tokenize(query)
        unique_terms = tuple(dict.fromkeys(terms))
        meaningful = [term for term in terms if term not in STOP_WORDS]
        context_hits = sum(term in CONTEXT_WORDS for term in terms)
        marker_hits = sum(term in AMBIGUITY_MARKERS for term in terms)
        context_dependence = min(1.0, context_hits / max(1, len(terms) / 3))
        underspecified = 1.0 if len(meaningful) <= 1 else 0.0
        missing_context = 0.0 if context.strip() else context_dependence
        ambiguity = min(1.0, 0.55 * underspecified + 0.35 * missing_context + 0.1 * marker_hits)

        initial, _ = index.search_bm25(query, limit=2)
        top_score = initial[0][1] if initial else 0.0
        second_score = initial[1][1] if len(initial) > 1 else 0.0
        bm25_confidence = 1.0 - math.exp(-max(0.0, top_score))
        bm25_margin = (top_score - second_score) / max(1.0, top_score)

        evidence_terms = tuple(dict.fromkeys(meaningful)) or unique_terms
        if evidence_terms and index.segments:
            present = sum(
                any(term in segment.term_filter for segment in index.segments)
                for term in evidence_terms
            )
            bloom_coverage = present / len(evidence_terms)
        else:
            bloom_coverage = 0.0

        return PolicyFeatures(
            token_count=min(1.0, len(terms) / 12.0),
            meaningful_ratio=len(meaningful) / max(1, len(terms)),
            context_dependence=context_dependence,
            ambiguity=ambiguity,
            bm25_confidence=bm25_confidence,
            bm25_margin=max(0.0, min(1.0, bm25_margin)),
            bloom_coverage=bloom_coverage,
        )

    def decide_layer_one(
        self, index: SearchIndex, query: str, *, context: str = ""
    ) -> ActionDecision:
        features = self.extract_features(index, query, context)
        if self.classifier is not None and self.classifier.fitted:
            action, confidence = self.classifier.predict(features.values())
            return ActionDecision(
                action,
                confidence,
                "trained action classifier selected the highest-probability action",
                "trained_classifier",
                features,
            )

        if features.ambiguity >= 0.6 and features.bm25_confidence < 0.45:
            return ActionDecision(
                "clarify",
                max(features.ambiguity, 1.0 - features.bm25_confidence),
                "the query is underspecified and retrieval confidence is low",
                "safe_baseline",
                features,
            )
        if (
            features.context_dependence >= 0.35
            or features.bloom_coverage < 0.5
            or features.bm25_confidence < 0.2
        ):
            return ActionDecision(
                "rewrite",
                max(0.55, 1.0 - features.bm25_confidence),
                "the query is context-dependent or has weak lexical evidence",
                "safe_baseline",
                features,
            )
        return ActionDecision(
            "keep",
            max(0.55, features.bm25_confidence),
            "the original query has sufficient lexical evidence",
            "safe_baseline",
            features,
        )

    @staticmethod
    def decide_layer_two(candidates: Sequence[tuple[object, float]]) -> RerankDecision:
        if len(candidates) < 2:
            return RerankDecision(False, 0.9, "fewer than two candidates can be reordered")
        top_score = max(0.0, candidates[0][1])
        second_score = max(0.0, candidates[1][1])
        normalized_margin = (top_score - second_score) / max(1.0, top_score)
        weak_top = 1.0 - (1.0 - math.exp(-top_score))
        uncertainty = max(0.0, min(1.0, 0.65 * (1.0 - normalized_margin) + 0.35 * weak_top))
        should_rerank = uncertainty >= 0.55
        reason = (
            "the leading BM25 candidates are difficult to separate"
            if should_rerank
            else "BM25 has a sufficiently clear leading result"
        )
        return RerankDecision(should_rerank, uncertainty if should_rerank else 1.0 - uncertainty, reason)

    @staticmethod
    def clarification_question(query: str) -> str:
        meaningful = [term for term in tokenize(query) if term not in STOP_WORDS]
        subject = " ".join(meaningful[:3]) or "that topic"
        return f"Could you clarify what you want to know about {subject}?"

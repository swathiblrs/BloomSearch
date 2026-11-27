"""Runtime adapter for the trained ambiguity detector.

The research model is loaded lazily so BloomSearch can still run with its
dependency-free baseline when the optional research dependencies or model
artifact are unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True, slots=True)
class AmbiguityResult:
    score: float
    needs_clarification: bool
    source: str


class AmbiguityDetector(Protocol):
    def predict(self, query: str) -> AmbiguityResult: ...


class SklearnAmbiguityDetector:
    """Use the team's persisted TF-IDF/logistic-regression detector."""

    def __init__(self, vectorizer: object, model: object, threshold: float) -> None:
        self.vectorizer = vectorizer
        self.model = model
        self.threshold = min(1.0, max(0.0, float(threshold)))

    @classmethod
    def load(cls, path: str | Path) -> "SklearnAmbiguityDetector":
        import joblib  # Optional research dependency; intentionally lazy.

        payload = joblib.load(Path(path))
        if not isinstance(payload, dict):
            raise ValueError("ambiguity model must contain a dictionary")
        try:
            return cls(payload["vectorizer"], payload["model"], payload["threshold"])
        except KeyError as error:
            raise ValueError("ambiguity model is missing a required component") from error

    def predict(self, query: str) -> AmbiguityResult:
        transform = getattr(self.vectorizer, "transform", None)
        predict_proba = getattr(self.model, "predict_proba", None)
        if not callable(transform) or not callable(predict_proba):
            raise ValueError("ambiguity model has incompatible components")
        matrix = transform([query])
        score = float(predict_proba(matrix)[0, 1])
        score = min(1.0, max(0.0, score))
        return AmbiguityResult(score, score >= self.threshold, "trained_ambiguity_detector")


@lru_cache(maxsize=4)
def load_ambiguity_detector(path: str | Path) -> AmbiguityDetector | None:
    """Load the optional trained detector, returning ``None`` for safe fallback."""
    try:
        model_path = Path(path)
        if not model_path.is_file():
            return None
        return SklearnAmbiguityDetector.load(model_path)
    except (AttributeError, ImportError, OSError, TypeError, ValueError):
        return None

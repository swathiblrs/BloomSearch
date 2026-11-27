import math

from research.evaluate_research_questions import (
    _estimated_input_tokens,
    _ndcg,
    _rank,
    _recall,
)


class Document:
    def __init__(self, document_id: str):
        self.id = document_id


def test_rank_and_single_relevant_document_metrics():
    results = [(Document("a"), 2.0), (Document("target"), 1.0)]
    rank = _rank(results, "target")
    assert rank == 2
    assert _recall(rank) == 1.0
    assert math.isclose(_ndcg(rank), 1.0 / math.log2(3))


def test_missing_relevant_document_has_zero_recall_and_ndcg():
    results = [(Document("a"), 2.0)]
    rank = _rank(results, "target")
    assert rank is None
    assert _recall(rank) == 0.0
    assert _ndcg(rank) == 0.0


def test_input_token_estimate_includes_context_and_is_never_zero():
    assert _estimated_input_tokens("", "") == 1
    assert _estimated_input_tokens("abcd", "efgh") == 3

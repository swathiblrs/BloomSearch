import math
from pathlib import Path

from research.evaluate_research_questions import (
    _estimated_input_tokens,
    _ndcg,
    _rank,
    _recall,
    automatic_intent_harm_comparison,
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


def test_automatic_intent_comparison_uses_same_guard_for_both_conditions(
    tmp_path: Path,
):
    study = tmp_path / "study.csv"
    study.write_text(
        "item_id,original_query,context,system_action,proposed_rewrite\n"
        "Q01,What happened in 2020?,,rewrite,What happened?\n"
        "Q02,Who won?,,keep,\n",
        encoding="utf-8",
    )
    always = tmp_path / "always.csv"
    always.write_text(
        "item_id,original_query,context,always_rewrite_output\n"
        "Q01,What happened in 2020?,,What happened?\n"
        "Q02,Who won?,,Who won?\n",
        encoding="utf-8",
    )

    result = automatic_intent_harm_comparison(study, always)

    assert result["always_rewrite"]["harmful_rewrite_rate"] == 0.5
    assert result["selective_rewrite"]["harmful_rewrite_rate"] == 1.0
    assert result["selective_minus_always_rate"] == 0.5

from research.human_study.analyze_always_rewrite_baseline import analyze
from research.human_study.prepare_always_rewrite_baseline import baseline_rows


def test_baseline_rows_include_only_qrecc_items():
    study = [
        {
            "item_id": "Q01",
            "original_query": "Original?",
            "context": "Context",
            "source_dataset": "QReCC",
            "source_record_id": "7",
        },
        {
            "item_id": "Q02",
            "original_query": "Ambiguous?",
            "context": "",
            "source_dataset": "AmbigNQ-derived validation",
            "source_record_id": "9",
        },
    ]
    rows = baseline_rows(study, {7: {"rewritten_query": "Focused rewrite?"}})
    assert rows == [
        {
            "item_id": "Q01",
            "original_query": "Original?",
            "context": "Context",
            "always_rewrite_output": "Focused rewrite?",
        }
    ]


def test_analysis_compares_selective_and_always_rewrite_per_query():
    study = [
        {"item_id": "Q01", "system_action": "rewrite", "source_dataset": "QReCC"},
        {"item_id": "Q02", "system_action": "keep", "source_dataset": "QReCC"},
    ]
    original = [
        {
            "participant_id": "P01",
            "item_id": "Q01",
            "rewrite_preserves_intent": "no",
        }
    ]
    baseline = [
        {
            "participant_id": "P01",
            "item_id": "Q01",
            "always_rewrite_preserves_intent": "no",
        },
        {
            "participant_id": "P01",
            "item_id": "Q02",
            "always_rewrite_preserves_intent": "no",
        },
    ]
    result = analyze(baseline, original, study)
    assert result["selective_harmful_rewrite_rate_per_query"] == 0.5
    assert result["always_rewrite_harmful_rate_per_query"] == 1.0
    assert result["absolute_rate_reduction"] == 0.5

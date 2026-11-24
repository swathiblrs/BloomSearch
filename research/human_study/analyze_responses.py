"""Analyze completed BloomSearch human-study response CSV files."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import mean


ACTIONS = ("keep", "rewrite", "clarify")


def macro_f1(actual: list[str], predicted: list[str]) -> tuple[float, dict[str, float]]:
    scores: dict[str, float] = {}
    for action in ACTIONS:
        true_positive = sum(a == action and p == action for a, p in zip(actual, predicted))
        false_positive = sum(a != action and p == action for a, p in zip(actual, predicted))
        false_negative = sum(a == action and p != action for a, p in zip(actual, predicted))
        denominator = 2 * true_positive + false_positive + false_negative
        scores[action] = 2 * true_positive / denominator if denominator else 0.0
    return mean(scores.values()), scores


def optional_int(row: dict[str, str], field: str) -> int | None:
    value = row.get(field, "").strip()
    return int(value) if value else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("responses", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    with args.responses.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("response file contains no participant responses")

    actual = [row["correct_action"].strip().lower() for row in rows]
    predicted = [row["system_action"].strip().lower() for row in rows]
    if any(value not in ACTIONS for value in actual + predicted):
        raise ValueError("actions must be keep, rewrite, or clarify")
    overall_macro_f1, per_action_f1 = macro_f1(actual, predicted)

    rewrite_rows = [
        row for row in rows if row["system_action"].strip().lower() == "rewrite"
    ]
    intent_labels = [
        row["rewrite_preserves_intent"].strip().lower()
        for row in rewrite_rows
        if row["rewrite_preserves_intent"].strip()
    ]
    if any(value not in {"yes", "no"} for value in intent_labels):
        raise ValueError("rewrite_preserves_intent must be yes, no, or blank")

    clarify_rows = [
        row for row in rows if row["system_action"].strip().lower() == "clarify"
    ]
    usefulness = [
        value
        for row in clarify_rows
        if (value := optional_int(row, "clarification_usefulness_1_to_5")) is not None
    ]
    task_success = [
        value
        for row in clarify_rows
        if (value := optional_int(row, "task_success_0_or_1")) is not None
    ]
    trust = [
        value
        for row in rows
        if (value := optional_int(row, "trust_1_to_5")) is not None
    ]
    if any(value not in range(1, 6) for value in usefulness + trust):
        raise ValueError("usefulness and trust scores must be between 1 and 5")
    if any(value not in {0, 1} for value in task_success):
        raise ValueError("task_success must be 0, 1, or blank")

    annotations_by_item: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        annotations_by_item[row["item_id"]].add(row["correct_action"].strip().lower())
    agreement = _safe_rate(
        sum(len(labels) == 1 for labels in annotations_by_item.values()),
        len(annotations_by_item),
    )

    result = {
        "participants": len({row["participant_id"] for row in rows}),
        "responses": len(rows),
        "items": len(annotations_by_item),
        "action_policy": {
            "macro_f1": overall_macro_f1,
            "per_action_f1": per_action_f1,
            "unanimous_action_agreement_rate": agreement,
        },
        "intent_preservation": {
            "evaluated_rewrites": len(intent_labels),
            "harmful_rewrite_rate": _safe_rate(intent_labels.count("no"), len(intent_labels)),
        },
        "clarification": {
            "evaluated_usefulness": len(usefulness),
            "mean_usefulness_1_to_5": mean(usefulness) if usefulness else None,
            "useful_rating_at_least_4_rate": _safe_rate(
                sum(value >= 4 for value in usefulness), len(usefulness)
            ),
            "evaluated_task_success": len(task_success),
            "task_success_rate": _safe_rate(sum(task_success), len(task_success)),
        },
        "trust": {
            "evaluated": len(trust),
            "mean_1_to_5": mean(trust) if trust else None,
        },
    }
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


if __name__ == "__main__":
    raise SystemExit(main())

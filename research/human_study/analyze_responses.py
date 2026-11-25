"""Analyze blinded BloomSearch human-study response CSV files."""

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


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def join_private_key(
    responses: list[dict[str, str]], study_items: list[dict[str, str]]
) -> list[dict[str, str]]:
    """Join hidden system actions after participant responses are collected."""
    by_id = {row["item_id"]: row for row in study_items}
    if len(by_id) != len(study_items):
        raise ValueError("study key contains duplicate item IDs")
    merged: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for response in responses:
        participant = response.get("participant_id", "").strip()
        item_id = response.get("item_id", "").strip()
        if not participant or not item_id:
            raise ValueError("participant_id and item_id are required")
        pair = (participant, item_id)
        if pair in seen:
            raise ValueError(f"duplicate response for participant/item: {pair}")
        seen.add(pair)
        if item_id not in by_id:
            raise ValueError(f"response item is not present in the study key: {item_id}")
        merged.append({**by_id[item_id], **response, "system_action": by_id[item_id]["system_action"]})
    return merged


def _task_success_summary(rows: list[dict[str, str]]) -> dict[str, int | float | None]:
    values = [
        value
        for row in rows
        if (value := optional_int(row, "task_success_0_or_1")) is not None
    ]
    if any(value not in {0, 1} for value in values):
        raise ValueError("task_success must be 0, 1, or blank")
    return {
        "evaluated": len(values),
        "success_rate": _safe_rate(sum(values), len(values)),
    }


def analyze(
    response_rows: list[dict[str, str]], study_rows: list[dict[str, str]]
) -> dict[str, object]:
    if not response_rows:
        raise ValueError("response file contains no participant responses")
    rows = join_private_key(response_rows, study_rows)
    actual = [row["correct_action"].strip().lower() for row in rows]
    predicted = [row["system_action"].strip().lower() for row in rows]
    if any(value not in ACTIONS for value in actual + predicted):
        raise ValueError("actions must be keep, rewrite, or clarify")
    overall_macro_f1, per_action_f1 = macro_f1(actual, predicted)

    rewrite_rows = [row for row in rows if row["system_action"] == "rewrite"]
    intent_labels = [
        row["rewrite_preserves_intent"].strip().lower()
        for row in rewrite_rows
        if row.get("rewrite_preserves_intent", "").strip()
    ]
    if any(value not in {"yes", "no"} for value in intent_labels):
        raise ValueError("rewrite_preserves_intent must be yes, no, or blank")

    clarify_rows = [row for row in rows if row["system_action"] == "clarify"]
    usefulness = [
        value
        for row in clarify_rows
        if (value := optional_int(row, "clarification_usefulness_1_to_5")) is not None
    ]
    trust = [
        value
        for row in rows
        if (value := optional_int(row, "trust_1_to_5")) is not None
    ]
    if any(value not in range(1, 6) for value in usefulness + trust):
        raise ValueError("usefulness and trust scores must be between 1 and 5")

    annotations_by_item: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        annotations_by_item[row["item_id"]].add(row["correct_action"].strip().lower())
    agreement = _safe_rate(
        sum(len(labels) == 1 for labels in annotations_by_item.values()),
        len(annotations_by_item),
    )
    participants = {row["participant_id"].strip() for row in rows}
    expected_items = {row["item_id"] for row in study_rows}
    completion_by_participant = {
        participant: _safe_rate(
            len({row["item_id"] for row in rows if row["participant_id"].strip() == participant}),
            len(expected_items),
        )
        for participant in sorted(participants)
    }

    task_by_action = {
        action: _task_success_summary([row for row in rows if row["system_action"] == action])
        for action in ACTIONS
    }
    return {
        "participants": len(participants),
        "responses": len(rows),
        "items": len(annotations_by_item),
        "completion_rate_by_participant": completion_by_participant,
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
            "primary_perceived_task_success": task_by_action["clarify"],
        },
        "perceived_task_success": {
            "overall": _task_success_summary(rows),
            "by_system_action": task_by_action,
        },
        "trust": {
            "evaluated": len(trust),
            "mean_1_to_5": mean(trust) if trust else None,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("responses", type=Path)
    parser.add_argument(
        "--study-key",
        type=Path,
        default=Path("research/human_study/study_items.csv"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    with args.responses.open(newline="", encoding="utf-8") as handle:
        responses = list(csv.DictReader(handle))
    with args.study_key.open(newline="", encoding="utf-8") as handle:
        study_items = list(csv.DictReader(handle))
    result = analyze(responses, study_items)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

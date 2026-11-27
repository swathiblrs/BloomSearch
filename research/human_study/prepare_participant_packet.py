"""Create a blinded, participant-specific BloomSearch study packet."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import random


ACTION_SELECTION_FIELDS = (
    "item_id",
    "original_query",
    "context",
)
OUTPUT_EVALUATION_FIELDS = (
    "item_id",
    "original_query",
    "context",
    "proposed_rewrite",
    "clarification_question",
)
RESPONSE_FIELDS = (
    "participant_id",
    "item_id",
    "correct_action",
    "rewrite_preserves_intent",
    "clarification_usefulness_1_to_5",
    "task_success_0_or_1",
    "trust_1_to_5",
    "notes",
)


def packet_rows(
    study_rows: list[dict[str, str]], participant_id: str, *, seed: int
) -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
]:
    if not participant_id.strip():
        raise ValueError("participant_id cannot be blank")
    rows = list(study_rows)
    random.Random(f"{seed}:{participant_id.strip()}").shuffle(rows)
    action_selection = [
        {field: row.get(field, "") for field in ACTION_SELECTION_FIELDS}
        for row in rows
    ]
    output_evaluation = [
        {field: row.get(field, "") for field in OUTPUT_EVALUATION_FIELDS}
        for row in rows
    ]
    responses = [
        {
            "participant_id": participant_id.strip(),
            "item_id": row["item_id"],
            "correct_action": "",
            "rewrite_preserves_intent": "",
            "clarification_usefulness_1_to_5": "",
            "task_success_0_or_1": "",
            "trust_1_to_5": "",
            "notes": "",
        }
        for row in rows
    ]
    return action_selection, output_evaluation, responses


def write_csv(path: Path, rows: list[dict[str, str]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fields),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("participant_id")
    parser.add_argument(
        "--study-key",
        type=Path,
        default=Path("research/human_study/study_items.csv"),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("research/human_study/participant_packets"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with args.study_key.open(newline="", encoding="utf-8") as handle:
        study_rows = list(csv.DictReader(handle))
    if not study_rows:
        raise ValueError("study key contains no items")
    action_selection, output_evaluation, responses = packet_rows(
        study_rows, args.participant_id, seed=args.seed
    )
    prefix = args.participant_id.strip()
    action_path = args.output_directory / f"{prefix}_phase_a_actions.csv"
    output_path = args.output_directory / f"{prefix}_phase_b_outputs.csv"
    responses_path = args.output_directory / f"{prefix}_responses.csv"
    write_csv(action_path, action_selection, ACTION_SELECTION_FIELDS)
    write_csv(output_path, output_evaluation, OUTPUT_EVALUATION_FIELDS)
    write_csv(responses_path, responses, RESPONSE_FIELDS)
    print(f"Wrote action-selection packet to {action_path}")
    print(f"Wrote output-evaluation packet to {output_path}")
    print(f"Wrote response sheet to {responses_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

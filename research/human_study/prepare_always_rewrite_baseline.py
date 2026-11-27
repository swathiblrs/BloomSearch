"""Prepare the RQ3 always-rewrite baseline for existing P01/P02 participants."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import subprocess


ITEM_FIELDS = (
    "item_id",
    "original_query",
    "context",
    "always_rewrite_output",
)
RESPONSE_FIELDS = (
    "participant_id",
    "item_id",
    "always_rewrite_preserves_intent",
    "notes",
)


def historical_records(commit: str) -> dict[int, dict[str, object]]:
    payload = json.loads(
        subprocess.check_output(
            [
                "git",
                "show",
                f"{commit}:data/qrecc/system_rewrite_evaluation_final.json",
            ]
        )
    )
    return {int(row["query_index"]): row for row in payload["records"]}


def baseline_rows(
    study_rows: list[dict[str, str]], records: dict[int, dict[str, object]]
) -> list[dict[str, str]]:
    rows = []
    for study_row in study_rows:
        if study_row["source_dataset"] != "QReCC":
            continue
        record = records[int(study_row["source_record_id"])]
        rewrite = " ".join(str(record["rewritten_query"]).split()).strip()
        if not rewrite:
            raise ValueError(f"stored rewrite is blank for {study_row['item_id']}")
        rows.append(
            {
                "item_id": study_row["item_id"],
                "original_query": study_row["original_query"],
                "context": study_row["context"],
                "always_rewrite_output": rewrite,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("participant_id")
    parser.add_argument("--historical-commit", default="19459c6c")
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

    participant = args.participant_id.strip()
    if not participant:
        raise ValueError("participant_id cannot be blank")
    with args.study_key.open(newline="", encoding="utf-8") as handle:
        study_rows = list(csv.DictReader(handle))
    rows = baseline_rows(study_rows, historical_records(args.historical_commit))
    random.Random(f"{args.seed}:{participant}:always-rewrite").shuffle(rows)
    responses = [
        {
            "participant_id": participant,
            "item_id": row["item_id"],
            "always_rewrite_preserves_intent": "",
            "notes": "",
        }
        for row in rows
    ]
    item_path = args.output_directory / f"{participant}_phase_c_always_rewrite.csv"
    response_path = (
        args.output_directory / f"{participant}_phase_c_always_rewrite_responses.csv"
    )
    write_csv(item_path, rows, ITEM_FIELDS)
    write_csv(response_path, responses, RESPONSE_FIELDS)
    print(f"Wrote {len(rows)} always-rewrite items to {item_path}")
    print(f"Wrote response sheet to {response_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

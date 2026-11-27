"""Prepare a blinded, reproducible 30-item BloomSearch pilot-study set."""

from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path
import random
import subprocess
import tempfile
from typing import Iterable

from bloom_search.index import SearchIndex
from bloom_search.nlp_policy import JointActionPolicy


PRIVATE_FIELDS = (
    "item_id",
    "original_query",
    "context",
    "system_action",
    "proposed_rewrite",
    "clarification_question",
    "source_dataset",
    "source_record_id",
)


def git_json(commit: str, path: str) -> object:
    return json.loads(subprocess.check_output(["git", "show", f"{commit}:{path}"]))


def git_index(commit: str, path: str) -> SearchIndex:
    payload = subprocess.check_output(["git", "show", f"{commit}:{path}"])
    with tempfile.NamedTemporaryFile(suffix=".json") as handle:
        handle.write(payload)
        handle.flush()
        return SearchIndex.load(handle.name)


def parse_interpretations(value: str) -> list[str]:
    """Parse and deduplicate AmbigNQ interpretation questions."""
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    unique: list[str] = []
    seen: set[str] = set()
    for raw in parsed:
        text = " ".join(str(raw).split()).strip()
        normalized = text.casefold()
        if text and normalized not in seen:
            seen.add(normalized)
            unique.append(text)
    return unique


def targeted_clarification(interpretations: Iterable[str], *, maximum: int = 3) -> str:
    """Create a dataset-backed clarification with concrete alternatives."""
    options = [" ".join(value.split()).strip() for value in interpretations if value.strip()]
    options = options[:maximum]
    if len(options) < 2:
        raise ValueError("a targeted clarification needs at least two interpretations")
    numbered = [
        f"({number}) {option.rstrip('?')}"
        for number, option in enumerate(options, start=1)
    ]
    if len(numbered) == 2:
        rendered = " or ".join(numbered)
    else:
        rendered = "; ".join(numbered[:-1]) + f"; or {numbered[-1]}"
    return f"Do you mean {rendered}?"


def select_diverse(
    candidates: list[dict[str, str]],
    count: int,
    *,
    rng: random.Random,
) -> list[dict[str, str]]:
    """Randomize candidates and prefer one item from each source conversation."""
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    selected: list[dict[str, str]] = []
    used_groups: set[str] = set()
    for row in shuffled:
        group = row["source_group"]
        if group in used_groups:
            continue
        selected.append(row)
        used_groups.add(group)
        if len(selected) == count:
            return selected
    for row in shuffled:
        if row in selected:
            continue
        selected.append(row)
        if len(selected) == count:
            return selected
    raise ValueError(f"could select only {len(selected)} of {count} requested items")


def _write_csv(path: Path, rows: list[dict[str, str]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fieldnames} for row in rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-commit", default="19459c6c")
    parser.add_argument("--items-per-action", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/human_study/study_items.csv"),
        help="Private study key. Do not show this file to participants.",
    )
    args = parser.parse_args()
    if args.items_per_action < 1:
        raise ValueError("items-per-action must be positive")

    rewrite_payload = git_json(
        args.historical_commit,
        "data/qrecc/system_rewrite_evaluation_final.json",
    )
    qrecc = git_json(args.historical_commit, "data/qrecc/test_3000.json")
    if not isinstance(rewrite_payload, dict) or not isinstance(qrecc, list):
        raise ValueError("unexpected historical study-data format")
    records = rewrite_payload.get("records")
    if not isinstance(records, list):
        raise ValueError("rewrite records are missing")

    policy = JointActionPolicy.runtime()
    qrecc_index = git_index(
        args.historical_commit,
        "data/qrecc/qrecc_retrieval_index.json",
    )
    action_candidates: dict[str, list[dict[str, str]]] = {"keep": [], "rewrite": []}
    for row in records:
        query_index = int(row["query_index"])
        example = qrecc[query_index]
        context_value = example.get("context", [])
        context = "\n".join(str(turn) for turn in context_value)
        decision = policy.decide_layer_one(
            qrecc_index,
            str(row["original_query"]),
            context=context,
        )
        action = decision.action
        if action not in action_candidates:
            continue
        if action == "rewrite" and (
            not row.get("accepted")
            or row.get("rewritten_query") == row.get("original_query")
        ):
            continue
        action_candidates[action].append(
            {
                "original_query": str(row["original_query"]),
                "context": " | ".join(str(turn) for turn in context_value[-4:]),
                "system_action": action,
                "proposed_rewrite": str(row["rewritten_query"]) if action == "rewrite" else "",
                "clarification_question": "",
                "source_dataset": "QReCC",
                "source_record_id": str(query_index),
                "source_group": f"conversation-{example['conversation_id']}",
            }
        )

    rng = random.Random(args.seed)
    selected: list[dict[str, str]] = []
    for action in ("keep", "rewrite"):
        selected.extend(
            select_diverse(
                action_candidates[action],
                args.items_per_action,
                rng=rng,
            )
        )

    validation_path = Path("ambiguity/data/processed/validation.csv")
    with validation_path.open(newline="", encoding="utf-8") as handle:
        ambiguous_rows = [
            row
            for row in csv.DictReader(handle)
            if row["ambiguity_label"] == "1" and len(parse_interpretations(row["interpretations"])) >= 2
        ]
    rng.shuffle(ambiguous_rows)
    empty_index = SearchIndex.build([])
    clarify_candidates: list[dict[str, str]] = []
    for row in ambiguous_rows:
        query = row["query"]
        decision = policy.decide_layer_one(empty_index, query)
        if decision.action != "clarify":
            continue
        interpretations = parse_interpretations(row["interpretations"])
        clarify_candidates.append(
            {
                "original_query": query,
                "context": "",
                "system_action": "clarify",
                "proposed_rewrite": "",
                "clarification_question": targeted_clarification(interpretations),
                "source_dataset": "AmbigNQ-derived validation",
                "source_record_id": str(row["id"]),
                "source_group": f"ambiguity-{row['id']}",
            }
        )
    selected.extend(select_diverse(clarify_candidates, args.items_per_action, rng=rng))

    expected = 3 * args.items_per_action
    if len(selected) != expected:
        raise ValueError(f"could prepare only {len(selected)} of {expected} requested items")

    # Assign opaque identifiers after mixing actions so IDs cannot reveal labels.
    rng.shuffle(selected)
    width = max(2, len(str(expected)))
    for number, row in enumerate(selected, start=1):
        row["item_id"] = f"Q{number:0{width}d}"
        row.pop("source_group", None)

    _write_csv(args.output, selected, PRIVATE_FIELDS)
    counts = {
        action: sum(row["system_action"] == action for row in selected)
        for action in ("keep", "rewrite", "clarify")
    }
    print(f"Wrote {len(selected)} blinded-key items to {args.output}: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

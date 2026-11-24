"""Prepare a balanced 30-item pilot-study set from existing project data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import tempfile

from bloom_search.index import SearchIndex
from bloom_search.nlp_policy import JointActionPolicy


def git_json(commit: str, path: str) -> object:
    return json.loads(subprocess.check_output(["git", "show", f"{commit}:{path}"]))


def git_index(commit: str, path: str) -> SearchIndex:
    payload = subprocess.check_output(["git", "show", f"{commit}:{path}"])
    with tempfile.NamedTemporaryFile(suffix=".json") as handle:
        handle.write(payload)
        handle.flush()
        return SearchIndex.load(handle.name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-commit", default="19459c6c")
    parser.add_argument("--items-per-action", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/human_study/study_items.csv"),
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

    items: list[dict[str, object]] = []
    selected_counts = {"keep": 0, "rewrite": 0, "clarify": 0}
    policy = JointActionPolicy.runtime()
    qrecc_index = git_index(
        args.historical_commit,
        "data/qrecc/qrecc_retrieval_index.json",
    )
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
        if action not in {"keep", "rewrite"}:
            continue
        if selected_counts[action] >= args.items_per_action:
            continue
        # A rewrite study item needs a real stored proposal, not a gold answer.
        if action == "rewrite" and (
            not row.get("accepted")
            or row.get("rewritten_query") == row.get("original_query")
        ):
            continue
        selected_counts[action] += 1
        items.append(
            {
                "item_id": f"{action}-{selected_counts[action]:02d}",
                "original_query": row["original_query"],
                "context": " | ".join(str(turn) for turn in context_value[-4:]),
                "system_action": action,
                "proposed_rewrite": row["rewritten_query"] if action == "rewrite" else "",
                "clarification_question": "",
            }
        )
        if all(selected_counts[action] >= args.items_per_action for action in ("keep", "rewrite")):
            break

    with Path("ambiguity/data/processed/validation.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        ambiguous = [row for row in csv.DictReader(handle) if row["ambiguity_label"] == "1"]
    empty_index = SearchIndex.build([])
    for row in ambiguous:
        query = row["query"]
        decision = policy.decide_layer_one(empty_index, query)
        if decision.action != "clarify":
            continue
        selected_counts["clarify"] += 1
        items.append(
            {
                "item_id": f"clarify-{selected_counts['clarify']:02d}",
                "original_query": query,
                "context": "",
                "system_action": "clarify",
                "proposed_rewrite": "",
                "clarification_question": JointActionPolicy.clarification_question(query),
            }
        )
        if selected_counts["clarify"] >= args.items_per_action:
            break

    expected = 3 * args.items_per_action
    if len(items) != expected:
        raise ValueError(f"could prepare only {len(items)} of {expected} requested items")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(items[0]))
        writer.writeheader()
        writer.writerows(items)
    print(f"Wrote {len(items)} balanced study items to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Compare selective and always-rewrite human intent-preservation labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from research.evaluate_nonhuman_completion import _mcnemar_exact


def analyze(
    baseline_rows: list[dict[str, str]],
    original_rows: list[dict[str, str]],
    study_rows: list[dict[str, str]],
) -> dict[str, object]:
    actions = {row["item_id"]: row["system_action"] for row in study_rows}
    qrecc_ids = {
        row["item_id"] for row in study_rows if row["source_dataset"] == "QReCC"
    }
    original_by_pair = {
        (row["participant_id"], row["item_id"]): row for row in original_rows
    }
    always_harm: list[bool] = []
    selective_harm: list[bool] = []
    seen: set[tuple[str, str]] = set()
    for row in baseline_rows:
        pair = (row["participant_id"].strip(), row["item_id"].strip())
        if pair in seen:
            raise ValueError(f"duplicate baseline response: {pair}")
        seen.add(pair)
        if pair[1] not in qrecc_ids:
            raise ValueError(f"baseline item is not a QReCC study item: {pair[1]}")
        label = row["always_rewrite_preserves_intent"].strip().lower()
        if label not in {"yes", "no"}:
            raise ValueError("always_rewrite_preserves_intent must be yes or no")
        always_harm.append(label == "no")

        if actions[pair[1]] == "rewrite":
            original = original_by_pair.get(pair)
            if original is None:
                raise ValueError(f"missing original response for {pair}")
            selected_label = original["rewrite_preserves_intent"].strip().lower()
            if selected_label not in {"yes", "no"}:
                raise ValueError(f"missing selective intent label for {pair}")
            selective_harm.append(selected_label == "no")
        else:
            # The selective policy kept the original query, so no rewrite harm occurred.
            selective_harm.append(False)

    if not always_harm:
        raise ValueError("no always-rewrite responses were provided")
    test = _mcnemar_exact(selective_harm, always_harm)
    return {
        "participants": len({row["participant_id"].strip() for row in baseline_rows}),
        "paired_judgments": len(always_harm),
        "selective_harmful_rewrite_rate_per_query": sum(selective_harm)
        / len(selective_harm),
        "always_rewrite_harmful_rate_per_query": sum(always_harm) / len(always_harm),
        "absolute_rate_reduction": (sum(always_harm) - sum(selective_harm))
        / len(always_harm),
        "mcnemar": test,
    }


def read_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows = []
    for path in paths:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_responses", type=Path, nargs="+")
    parser.add_argument(
        "--original-responses", type=Path, nargs="+", required=True
    )
    parser.add_argument(
        "--study-key",
        type=Path,
        default=Path("research/human_study/study_items.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/results/always_rewrite_human_baseline.json"),
    )
    args = parser.parse_args()
    with args.study_key.open(newline="", encoding="utf-8") as handle:
        study_rows = list(csv.DictReader(handle))
    result = analyze(
        read_rows(args.baseline_responses),
        read_rows(args.original_responses),
        study_rows,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

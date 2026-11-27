"""Calculate remaining automatic metrics from stored rewrite records.

This script makes no model/API calls. It replays the historical QReCC hosted
rewrite records retained in Git history and reports retrieval degradation as a
proxy for harmful rewriting. Semantic intent harm still requires annotation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess


def load_historical_json(commit: str, path: str) -> object:
    payload = subprocess.check_output(["git", "show", f"{commit}:{path}"])
    return json.loads(payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-commit", default="19459c6c")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/results/remaining_metrics.json"),
    )
    args = parser.parse_args()

    payload = load_historical_json(
        args.historical_commit,
        "data/qrecc/system_rewrite_evaluation_final.json",
    )
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise ValueError("unexpected historical rewrite-result format")
    records = payload["records"]
    accepted = [row for row in records if row.get("accepted")]
    hit_harm = [
        row for row in accepted if row.get("original_hit") and not row.get("rewritten_hit")
    ]
    rank_harm = [
        row
        for row in accepted
        if float(row.get("rewritten_rr", 0.0)) < float(row.get("original_rr", 0.0))
    ]
    rank_help = [
        row
        for row in accepted
        if float(row.get("rewritten_rr", 0.0)) > float(row.get("original_rr", 0.0))
    ]

    result = {
        "dataset": "QReCC stored rewrite evaluation",
        "queries": len(records),
        "accepted_rewrites": len(accepted),
        "retrieval_harm_proxy": {
            "lost_hit_at_5_count": len(hit_harm),
            "lost_hit_at_5_rate_per_accepted_rewrite": len(hit_harm) / len(accepted),
            "reduced_reciprocal_rank_count": len(rank_harm),
            "reduced_reciprocal_rank_rate_per_accepted_rewrite": len(rank_harm)
            / len(accepted),
            "improved_reciprocal_rank_count": len(rank_help),
            "improved_reciprocal_rank_rate_per_accepted_rewrite": len(rank_help)
            / len(accepted),
        },
        "warning": (
            "Retrieval degradation is not proof of changed user intent. The semantic "
            "harmful-rewrite rate must be measured with human intent-preservation labels."
        ),
        "remaining_human_metrics": [
            "three-class keep/rewrite/clarify Macro-F1",
            "semantic harmful-rewrite rate",
            "clarification usefulness and task success",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

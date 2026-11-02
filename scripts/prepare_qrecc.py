import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data" / "qrecc" / "qrecc-training.json"
DEFAULT_OUTPUT = ROOT / "data" / "qrecc" / "training_sample.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = json.loads(args.source.read_text(encoding="utf-8"))
    prepared = []

    for row in rows[:args.limit]:
        context = row.get("Context", [])
        if isinstance(context, str):
            context = [context]

        prepared.append(
            {
                "conversation_id": row.get("Conversation_no"),
                "turn_id": row.get("Turn_no"),
                "context": context,
                "original_query": row.get("Question", ""),
                "gold_rewrite": row.get("Rewrite", ""),
                "answer": row.get("Answer", ""),
                "answer_url": row.get("Answer_URL", ""),
            }
        )

    args.output.write_text(
        json.dumps(prepared, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Prepared {len(prepared)} examples")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
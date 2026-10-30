import json
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, classification_report

from scripts.train_rewrite_benefit_predictor import features


ROOT = Path(__file__).resolve().parents[1]
TEST_FILE = ROOT / "data" / "qrecc" / "test_3000.json"
RESULTS_FILE = ROOT / "data" / "qrecc" / "test_3000_results.json"
MODEL_FILE = ROOT / "data" / "qrecc" / "rewrite_benefit_12000.joblib"
OUTPUT_FILE = ROOT / "data" / "qrecc" / "rewrite_benefit_test_3000_metrics.json"


def main() -> None:
    test_rows = json.loads(TEST_FILE.read_text(encoding="utf-8"))

    retrieval_results = json.loads(
        RESULTS_FILE.read_text(encoding="utf-8")
    )["results"]

    labels = {
        (row["conversation_id"], row["turn_id"]): row["rewrite_helped"]
        for row in retrieval_results
    }

    rows = [
        row for row in test_rows
        if (row["conversation_id"], row["turn_id"]) in labels
    ]

    bundle = joblib.load(MODEL_FILE)
    vectorizer = bundle["vectorizer"]
    model = bundle["model"]

    x = vectorizer.transform([features(row) for row in rows])
    y_true = [labels[(row["conversation_id"], row["turn_id"])] for row in rows]
    y_pred = model.predict(x).tolist()

    metrics = {
        "examples": len(rows),
        "test_accuracy": accuracy_score(y_true, y_pred),
        "classification_report": classification_report(
            y_true,
            y_pred,
            output_dict=True,
            zero_division=0,
        ),
        "rewrite_recommended": sum(y_pred),
    }

    OUTPUT_FILE.write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Test examples: {len(rows)}")
    print(f"Predictor test accuracy: {metrics['test_accuracy']:.1%}")
    print(f"Rewrites recommended: {metrics['rewrite_recommended']}")
    print(f"Saved metrics: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
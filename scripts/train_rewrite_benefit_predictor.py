import json
import re
from pathlib import Path

import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_FILE = ROOT / "data" / "qrecc" / "training_12000.json"
RESULTS_FILE = ROOT / "data" / "qrecc" / "training_12000_results.json"
MODEL_FILE = ROOT / "data" / "qrecc" / "rewrite_benefit_12000.joblib"
METRICS_FILE = ROOT / "data" / "qrecc" / "rewrite_benefit_metrics_12000.json"

REFERENCE_WORDS = {
    "it", "its", "they", "them", "their", "he", "him", "his",
    "she", "her", "this", "that", "these", "those",
}


def features(row: dict) -> dict:
    query = row["original_query"].lower()
    words = re.findall(r"[a-z]+", query)
    context = row.get("context", [])

    return {
        "has_context": int(bool(context)),
        "context_turns": len(context),
        "query_word_count": len(words),
        "short_query": int(len(words) <= 6),
        "has_reference_word": int(any(word in REFERENCE_WORDS for word in words)),
        "starts_with_what_about": int(query.startswith("what about")),
        "starts_with_how_about": int(query.startswith("how about")),
        "starts_with_and": int(query.startswith("and ")),
    }


def main() -> None:
    sample_rows = json.loads(SAMPLE_FILE.read_text(encoding="utf-8"))
    evaluation = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))["results"]

    labels = {
        (row["conversation_id"], row["turn_id"]): row["rewrite_helped"]
        for row in evaluation
    }

    rows = [
        row for row in sample_rows
        if (row["conversation_id"], row["turn_id"]) in labels
    ]

    x = [features(row) for row in rows]
    y = [labels[(row["conversation_id"], row["turn_id"])] for row in rows]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    vectorizer = DictVectorizer(sparse=False)
    x_train_vector = vectorizer.fit_transform(x_train)
    x_test_vector = vectorizer.transform(x_test)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(x_train_vector, y_train)

    predictions = model.predict(x_test_vector)

    metrics = {
        "examples": len(rows),
        "test_accuracy": accuracy_score(y_test, predictions),
        "classification_report": classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0,
        ),
    }

    joblib.dump(
        {"vectorizer": vectorizer, "model": model},
        MODEL_FILE,
    )

    METRICS_FILE.write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Examples used: {len(rows)}")
    print(f"Test accuracy: {metrics['test_accuracy']:.1%}")
    print(f"Saved model: {MODEL_FILE}")
    print(f"Saved metrics: {METRICS_FILE}")


if __name__ == "__main__":
    main()

import json
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from scripts.train_rewrite_benefit_predictor import features

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "qrecc"

TRAIN_FILE = DATA / "training_12000.json"
TRAIN_RESULTS = DATA / "training_12000_results.json"
TEST_FILE = DATA / "test_3000.json"
TEST_RESULTS = DATA / "test_3000_results.json"

MODEL_FILE = DATA / "rewrite_hybrid_model.joblib"
METRICS_FILE = DATA / "rewrite_hybrid_metrics.json"


def load_data(sample_file, results_file):
    samples = json.loads(sample_file.read_text(encoding="utf-8"))
    results = json.loads(
        results_file.read_text(encoding="utf-8")
    )["results"]

    labels = {
        (str(r["conversation_id"]), str(r["turn_id"])):
            int(r["rewrite_helped"])
        for r in results
    }

    texts = []
    numeric_features = []
    targets = []

    for row in samples:
        key = (
            str(row["conversation_id"]),
            str(row["turn_id"]),
        )

        if key not in labels:
            continue

        query = row["original_query"]
        context = row.get("context", [])

        # Use recent conversational context.
        if isinstance(context, list):
            context_text = " ".join(map(str, context[-3:]))
        else:
            context_text = str(context)

        text = context_text[-500:] + " [QUERY] " + query

        # Reuse features from the original model.
        numeric = features(row)
        if isinstance(numeric, dict):
            numeric = list(numeric.values())

        texts.append(text)
        numeric_features.append(numeric)
        targets.append(labels[key])

    return (
        texts,
        np.asarray(numeric_features, dtype=float),
        np.asarray(targets, dtype=int),
    )


def main():
    print("Loading datasets...")

    X_train_text, X_train_num, y_train = load_data(
        TRAIN_FILE, TRAIN_RESULTS
    )
    X_test_text, X_test_num, y_test = load_data(
        TEST_FILE, TEST_RESULTS
    )

    print("Training examples:", len(y_train))
    print("Test examples:", len(y_test))

    # Convert query text into TF-IDF features.
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=15000,
        min_df=2,
        sublinear_tf=True,
    )

    train_tfidf = vectorizer.fit_transform(X_train_text)
    test_tfidf = vectorizer.transform(X_test_text)

    # Scale handcrafted features using training data only.
    scaler = StandardScaler()
    train_num = scaler.fit_transform(X_train_num)
    test_num = scaler.transform(X_test_num)

    # Combine text features and handcrafted features.
    X_train = hstack(
        [train_tfidf, csr_matrix(train_num)],
        format="csr",
    )
    X_test = hstack(
        [test_tfidf, csr_matrix(test_num)],
        format="csr",
    )

    # Train the hybrid classifier.
    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0,
    )

    print("\nHybrid Model Results")
    print("--------------------")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(
        classification_report(
            y_test, predictions, zero_division=0
        )
    )
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))
    print("Rewrites recommended:", int(predictions.sum()))

    # Save all preprocessing components with the model.
    joblib.dump(
        {
            "vectorizer": vectorizer,
            "scaler": scaler,
            "model": model,
        },
        MODEL_FILE,
    )

    metrics = {
        "training_examples": len(y_train),
        "test_examples": len(y_test),
        "accuracy": float(accuracy),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(
            y_test, predictions
        ).tolist(),
        "rewrites_recommended": int(predictions.sum()),
    }

    METRICS_FILE.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print("\nSaved model:", MODEL_FILE)
    print("Saved metrics:", METRICS_FILE)


if __name__ == "__main__":
    main()


import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "qrecc"

TRAIN_FILE = DATA / "training_12000.json"
TRAIN_RESULTS = DATA / "training_12000_results.json"
TEST_FILE = DATA / "test_3000.json"
TEST_RESULTS = DATA / "test_3000_results.json"

MODEL_FILE = DATA / "rewrite_tfidf_model.joblib"


def load_data(sample_file, results_file):
    rows = json.loads(sample_file.read_text(encoding="utf-8"))
    results = json.loads(results_file.read_text(encoding="utf-8"))["results"]

    labels = {
        (r["conversation_id"], r["turn_id"]): r["rewrite_helped"]
        for r in results
    }

    X, y = [], []

    for row in rows:
        key = (row["conversation_id"], row["turn_id"])
        if key not in labels:
            continue

        context = " ".join(row.get("context", []))
        query = row["original_query"]

        X.append(context[-500:] + " [QUERY] " + query)
        y.append(int(labels[key]))

    return X, y


X_train, y_train = load_data(TRAIN_FILE, TRAIN_RESULTS)
X_test, y_test = load_data(TEST_FILE, TEST_RESULTS)

model = make_pipeline(
    TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=15000,
        min_df=2
    ),
    LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Training examples:", len(X_train))
print("Test examples:", len(X_test))
print("Accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")
print(classification_report(y_test, predictions, zero_division=0))

joblib.dump(model, MODEL_FILE)
print("Saved model:", MODEL_FILE)

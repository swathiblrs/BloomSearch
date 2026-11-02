
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.model_selection import train_test_split

from scripts.train_rewrite_benefit_predictor import features

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "qrecc"

SAMPLE_FILE = DATA / "training_12000.json"
RESULTS_FILE = DATA / "training_12000_results.json"
MODEL_FILE = DATA / "rewrite_hybrid_model.joblib"

THRESHOLDS = [0.3, 0.4, 0.5, 0.6]


def main():
    samples = json.loads(SAMPLE_FILE.read_text(encoding="utf-8"))
    results = json.loads(
        RESULTS_FILE.read_text(encoding="utf-8")
    )["results"]

    lookup = {
        (str(r["conversation_id"]), str(r["turn_id"])): r
        for r in results
    }

    texts = []
    numeric_features = []
    original_hits = []
    rewrite_hits = []

    for row in samples:
        key = (
            str(row["conversation_id"]),
            str(row["turn_id"]),
        )

        if key not in lookup:
            continue

        result = lookup[key]
        query = row["original_query"]
        context = row.get("context", [])

        if isinstance(context, list):
            context_text = " ".join(map(str, context[-3:]))
        else:
            context_text = str(context)

        texts.append(
            context_text[-500:] + " [QUERY] " + query
        )

        numeric = features(row)
        if isinstance(numeric, dict):
            numeric = list(numeric.values())

        numeric_features.append(numeric)
        original_hits.append(int(result["original_hit_at_10"]))
        rewrite_hits.append(int(result["rewrite_hit_at_10"]))

    # Use a reproducible subset for threshold exploration.
    indices = np.arange(len(texts))
    _, validation_indices = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
    )

    bundle = joblib.load(MODEL_FILE)

    validation_texts = [texts[i] for i in validation_indices]
    validation_numeric = np.asarray(
        [numeric_features[i] for i in validation_indices],
        dtype=float,
    )

    X_text = bundle["vectorizer"].transform(validation_texts)
    X_numeric = bundle["scaler"].transform(validation_numeric)

    X = hstack(
        [X_text, csr_matrix(X_numeric)],
        format="csr",
    )

    model = bundle["model"]
    class_index = list(model.classes_).index(1)
    probabilities = model.predict_proba(X)[:, class_index]

    original = np.asarray(original_hits)[validation_indices]
    rewritten = np.asarray(rewrite_hits)[validation_indices]

    print("\nThreshold Validation")
    print("------------------------------")

    for threshold in THRESHOLDS:
        decisions = probabilities >= threshold
        selected_hits = np.where(decisions, rewritten, original)

        print(
            f"Threshold: {threshold:.2f} | "
            f"Hit@10: {np.mean(selected_hits) * 100:.2f}% | "
            f"Rewrite rate: {np.mean(decisions) * 100:.2f}%"
        )


if __name__ == "__main__":
    main()

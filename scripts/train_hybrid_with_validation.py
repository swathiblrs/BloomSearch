
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from scripts.train_rewrite_benefit_predictor import features

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "qrecc"

TRAIN_FILE = DATA / "training_12000.json"
RESULTS_FILE = DATA / "training_12000_results.json"

MODEL_FILE = DATA / "rewrite_hybrid_validated.joblib"
METRICS_FILE = DATA / "rewrite_hybrid_validation_metrics.json"

THRESHOLDS = [0.30, 0.40, 0.50, 0.60]


def load_data():
    samples = json.loads(
        TRAIN_FILE.read_text(encoding="utf-8")
    )

    results = json.loads(
        RESULTS_FILE.read_text(encoding="utf-8")
    )["results"]

    lookup = {
        (str(r["conversation_id"]), str(r["turn_id"])): r
        for r in results
    }

    texts = []
    numeric = []
    original_hits = []
    rewrite_hits = []
    groups = []

    for row in samples:
        key = (
            str(row["conversation_id"]),
            str(row["turn_id"]),
        )

        if key not in lookup:
            continue

        result = lookup[key]

        context = row.get("context", [])
        if isinstance(context, list):
            context_text = " ".join(map(str, context[-3:]))
        else:
            context_text = str(context)

        texts.append(
            context_text[-500:]
            + " [QUERY] "
            + row["original_query"]
        )

        f = features(row)
        if isinstance(f, dict):
            f = list(f.values())

        numeric.append(f)
        original_hits.append(
            int(result["original_hit_at_10"])
        )
        rewrite_hits.append(
            int(result["rewrite_hit_at_10"])
        )
        groups.append(str(row["conversation_id"]))

    return (
        texts,
        np.asarray(numeric, dtype=float),
        np.asarray(original_hits),
        np.asarray(rewrite_hits),
        np.asarray(groups),
    )


def build_model(texts, numeric, indices):
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=15000,
        min_df=2,
        sublinear_tf=True,
    )

    scaler = StandardScaler()

    X_text = vectorizer.fit_transform(
        [texts[i] for i in indices]
    )
    X_numeric = scaler.fit_transform(numeric[indices])

    X = hstack(
        [X_text, csr_matrix(X_numeric)],
        format="csr",
    )

    return vectorizer, scaler, X


def main():
    texts, numeric, original, rewritten, groups = load_data()

    indices = np.arange(len(texts))

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )

    train_idx, val_idx = next(
        splitter.split(indices, groups=groups)
    )

    print("Training examples:", len(train_idx))
    print("Validation examples:", len(val_idx))

    vectorizer, scaler, X_train = build_model(
        texts, numeric, train_idx
    )

    # Label 1 when rewriting improves retrieval.
    y_train = (
        rewritten[train_idx] > original[train_idx]
    ).astype(int)

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )

    model.fit(X_train, y_train)

    X_val_text = vectorizer.transform(
        [texts[i] for i in val_idx]
    )
    X_val_numeric = scaler.transform(numeric[val_idx])

    X_val = hstack(
        [X_val_text, csr_matrix(X_val_numeric)],
        format="csr",
    )

    class_index = list(model.classes_).index(1)
    probabilities = model.predict_proba(X_val)[:, class_index]

    print("\nValidation Threshold Results")
    print("-" * 55)

    results = []

    for threshold in THRESHOLDS:
        decisions = probabilities >= threshold

        hits = np.where(
            decisions,
            rewritten[val_idx],
            original[val_idx],
        )

        hit_rate = float(np.mean(hits))
        rewrite_rate = float(np.mean(decisions))

        print(
            f"Threshold {threshold:.2f} | "
            f"Hit@10: {hit_rate * 100:.2f}% | "
            f"Rewrite rate: {rewrite_rate * 100:.2f}%"
        )

        results.append({
            "threshold": threshold,
            "hit_at_10": hit_rate,
            "rewrite_rate": rewrite_rate,
        })

    # Select the best threshold under a 70% rewrite budget.
    eligible = [
        r for r in results
        if r["rewrite_rate"] <= 0.70
    ]

    if not eligible:
        raise ValueError(
            "No threshold satisfies the rewrite budget."
        )

    best = max(
        eligible,
        key=lambda r: r["hit_at_10"]
    )

    print("\nSelected threshold:", best["threshold"])
    print(
        "Validation Hit@10:",
        f"{best['hit_at_10'] * 100:.2f}%"
    )

    # Save the validation-trained model and selected threshold.
    joblib.dump(
        {
            "vectorizer": vectorizer,
            "scaler": scaler,
            "model": model,
            "threshold": best["threshold"],
        },
        MODEL_FILE,
    )

    metrics = {
        "training_examples": len(train_idx),
        "validation_examples": len(val_idx),
        "rewrite_budget": 0.70,
        "threshold_results": results,
        "selected_threshold": best["threshold"],
    }

    METRICS_FILE.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print("Saved model:", MODEL_FILE)
    print("Saved metrics:", METRICS_FILE)


if __name__ == "__main__":
    main()
